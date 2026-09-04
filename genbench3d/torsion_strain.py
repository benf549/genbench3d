"""Torsion-library strain score for a docked/generated ligand pose.

A geometry-only, license-clean strain metric: each rotatable (heavy, non-ring) torsion of the pose is
compared against an empirical distribution of that torsion's dihedral angle in a reference set of
experimental structures. Torsions sitting at low-probability angles are "strained". The per-pose score is
the MAX over torsions (a pose is as strained as its worst torsion).

Validated against AIMNet2 (wB97M+CPCM) relaxation: the pose max-torsion energy tracks the largest
per-torsion angular relaxation |Δθ| with Spearman ~0.5, and generalises across chemistry and the
bound/unbound domain (CSD-unbound reference -> PDBBind-bound crystal poses). See README (Methods).

Both the reference-building and the scoring use the helpers here so patterns are defined identically.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --- fixed discretisation (do NOT change without rebuilding libraries) ---
NBINS = 36                 # 10-degree bins over [-180, 180)
BINW = 360.0 / NBINS
SMOOTH_SIGMA = 1.5         # Gaussian KDE smoothing, in bins (~15 deg), wrapped
EPS = 1e-4                 # floor inside the log; caps energy at -log(EPS) ~ 9.21
DEFAULT_MIN_N = 50         # per-pattern support threshold (count thresholding)

_BUNDLED_DIR = os.path.join(os.path.dirname(__file__), "data", "torsion_libs")


# --------------------------------------------------------------------------- #
# torsion enumeration + pattern keys (shared by build and score)
# --------------------------------------------------------------------------- #
def _atom_key(atom) -> Tuple[int, int, int]:
    return (atom.GetAtomicNum(), int(atom.GetIsAromatic()), int(atom.GetHybridization()))


def bin_of(theta_deg: float) -> int:
    return int(((theta_deg + 180.0) % 360.0) // BINW)


def pattern_keys(mol, a: int, b: int, c: int, d: int) -> Tuple[str, str]:
    """Coarse torsion patterns: a 4-atom key (a-b-c-d neighbourhood) and a 2-atom central-bond backoff.
    Both are canonicalised so a-b-c-d and d-c-b-a map to the same key. Returned as repr() strings so the
    library is a plain dict keyed by hashable text."""
    K = [_atom_key(mol.GetAtomWithIdx(i)) for i in (a, b, c, d)]
    quad = min((K[0], K[1], K[2], K[3]), (K[3], K[2], K[1], K[0]))
    pair = min((K[1], K[2]), (K[2], K[1]))
    return repr(quad), repr(pair)


def heavy_nonring_torsions(mol) -> Iterable[Tuple[int, int, int, int]]:
    """Every heavy-atom, non-ring, single-bond torsion a-b-c-d (all heavy neighbour combinations of the
    central b-c bond). Hydrogens and ring bonds are excluded; double/triple/aromatic central bonds too."""
    from rdkit import Chem
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        b, c = bond.GetBeginAtom(), bond.GetEndAtom()
        if b.GetAtomicNum() == 1 or c.GetAtomicNum() == 1:
            continue
        bn = [n for n in b.GetNeighbors() if n.GetIdx() != c.GetIdx() and n.GetAtomicNum() > 1]
        cn = [n for n in c.GetNeighbors() if n.GetIdx() != b.GetIdx() and n.GetAtomicNum() > 1]
        for a in bn:
            for d in cn:
                yield a.GetIdx(), b.GetIdx(), c.GetIdx(), d.GetIdx()


# --------------------------------------------------------------------------- #
# library construction
# --------------------------------------------------------------------------- #
def _finalize(hist: np.ndarray) -> Tuple[np.ndarray, float, float]:
    from scipy.ndimage import gaussian_filter1d
    dens = gaussian_filter1d(hist, SMOOTH_SIGMA, mode="wrap")
    dens = dens / dens.sum()
    return dens.astype(np.float32), float(dens.max()), float(hist.sum())


def build_library(mol_iter: Iterable, source_name: str, min_n_for_neutral: int = DEFAULT_MIN_N) -> Dict:
    """Accumulate symmetrised dihedral histograms for every 4-atom and 2-atom pattern over a reference set
    of RDKit molecules (each with a single 3D conformer), and return a scorer-ready library. ALL patterns
    are kept with their support count N; the support threshold is applied at *score* time (so MIN_N can be
    tuned without rebuilding). `meta.neutral_energy` is the median per-torsion energy of the reference under
    its own library (used as the default for out-of-distribution torsions).
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolTransforms
    h4: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(NBINS))
    h2: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(NBINS))
    n_mol = 0
    for mol in mol_iter:
        if mol is None or mol.GetNumConformers() == 0:
            continue
        n_mol += 1
        conf = mol.GetConformer()
        for a, b, c, d in heavy_nonring_torsions(mol):
            try:
                th = rdMolTransforms.GetDihedralDeg(conf, a, b, c, d)
            except Exception:
                continue
            if th != th:
                continue
            k4, k2 = pattern_keys(mol, a, b, c, d)
            for t in (th, -th):                      # symmetrise: torsion sign is arbitrary
                i = bin_of(t)
                h4[k4][i] += 1.0
                h2[k2][i] += 1.0
    d4 = {k: _finalize(v) for k, v in h4.items()}
    d2 = {k: _finalize(v) for k, v in h2.items()}
    lib = {"d4": d4, "d2": d2,
           "meta": {"source": source_name, "n_molecules": n_mol,
                    "nbins": NBINS, "smooth_sigma": SMOOTH_SIGMA, "eps": EPS,
                    "n_patterns_4atom": len(d4), "n_patterns_2atom": len(d2)}}
    # neutral energy = median per-torsion energy of the reference under its own library
    scorer = TorsionStrainLibrary([lib], min_n=min_n_for_neutral, neutral_energy=0.0)
    es: List[float] = []
    for k4, (dens, mode, _n) in d4.items():
        es.extend(float(-np.log(dens[i] / mode + EPS)) for i in range(NBINS))
    lib["meta"]["neutral_energy"] = float(np.median(es)) if es else 4.0
    return lib


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
class TorsionStrainLibrary:
    """One or more torsion libraries scored together by union(max): a torsion's energy is the LARGEST
    (most-strained) verdict across the libraries that resolve it (each reference's notion of 'unusual')."""

    def __init__(self, libs: Sequence[Dict], min_n: int = DEFAULT_MIN_N,
                 neutral_energy: Optional[float] = None):
        self.libs = list(libs)
        self.min_n = int(min_n)
        if neutral_energy is None:
            ns = [L.get("meta", {}).get("neutral_energy") for L in self.libs]
            ns = [x for x in ns if x is not None]
            neutral_energy = float(np.mean(ns)) if ns else 4.0
        self.neutral_energy = float(neutral_energy)

    # ---- library resolution ----
    @staticmethod
    def _resolve_path(name_or_path: str) -> str:
        if os.path.exists(name_or_path):
            return name_or_path
        cand = os.path.join(_BUNDLED_DIR, name_or_path if name_or_path.endswith(".pkl")
                            else f"{name_or_path}.pkl")
        if os.path.exists(cand):
            return cand
        raise FileNotFoundError(
            f"torsion library '{name_or_path}' not found (looked in {_BUNDLED_DIR}). "
            f"Bundled: {available_references()}")

    @classmethod
    def load(cls, references: Sequence[str], min_n: int = DEFAULT_MIN_N,
             neutral_energy: Optional[float] = None) -> "TorsionStrainLibrary":
        libs = [pickle.load(open(cls._resolve_path(r), "rb")) for r in references]
        return cls(libs, min_n=min_n, neutral_energy=neutral_energy)

    # ---- per-torsion energy with count thresholding + backoff ----
    def _energy_one(self, lib: Dict, k4: str, k2: str, b: int) -> Optional[float]:
        for key, tier in ((k4, "d4"), (k2, "d2")):
            ent = lib[tier].get(key)
            if ent is not None and ent[2] >= self.min_n:     # support N >= MIN_N
                dens, mode, _n = ent
                return float(-np.log(dens[b] / mode + EPS))
        return None                                          # uncovered by this library

    def torsion_energy(self, k4: str, k2: str, b: int) -> Optional[float]:
        vals = [e for e in (self._energy_one(L, k4, k2, b) for L in self.libs) if e is not None]
        return max(vals) if vals else None                   # union(max); None -> uncovered by all

    # ---- pose score ----
    def score_mol(self, mol) -> Dict:
        """Return per-torsion energies and the pose strain = max over torsions (max-pool). Torsions
        uncovered by every library get the neutral energy (not the ceiling), so novel chemistry is neither
        spuriously penalised nor able to dodge the reward."""
        from rdkit.Chem import rdMolTransforms
        conf = mol.GetConformer()
        per = []
        for a, b, c, d in heavy_nonring_torsions(mol):
            try:
                th = rdMolTransforms.GetDihedralDeg(conf, a, b, c, d)
            except Exception:
                continue
            if th != th:
                continue
            k4, k2 = pattern_keys(mol, a, b, c, d)
            e = self.torsion_energy(k4, k2, bin_of(th))
            per.append({"atoms": (a, b, c, d), "angle": float(th),
                        "energy": self.neutral_energy if e is None else float(e),
                        "covered": e is not None})
        vals = [p["energy"] for p in per]
        pose = float(max(vals)) if vals else float("nan")
        n_cov = int(sum(p["covered"] for p in per))
        return {"pose_strain": pose, "n_torsions": len(per), "n_covered": n_cov, "per_torsion": per}


# --------------------------------------------------------------------------- #
# pose strain -> [0, 1] design reward  (saturating softplus shoulder)
# --------------------------------------------------------------------------- #
# Label-free calibration: real crystal ligands are, by definition, essentially unstrained, so we score a
# large set of them and place the reward so a typical crystal earns ~full reward. The reward SATURATES at
# 1 across the crystallographic region and only falls for poses more strained than real crystals:
#
#   r(s) = exp( -LAM * softplus(BETA*(s - REWARD_PLATEAU)) / BETA ),   LAM = ln2 / excess(REWARD_KNEE)
#
# i.e. r = 1.0 (flat) for s <= REWARD_PLATEAU, r = 0.5 at REWARD_KNEE, and r -> 0 for highly-strained
# poses (with a gentle residual gradient in the tail, unlike a hard cut). Flattening the crystallographic
# region keeps the reward from spending gradient — or trading against other objectives — on strain
# differences that are below the noise floor of real crystals; it bites only on genuine outliers.
#
# The anchors below were calibrated on 4,788 PDBBind crystal ligands scored with the validated
# union(custom-CSD + LigBoundConf) library at MIN_N=50 (PDBBind held out of the library, so non-circular):
# REWARD_PLATEAU = 3.0 sits at ~p76 of that crystal pose-strain distribution and REWARD_KNEE = 4.95 at p95.
# NOTE: the anchors live on the pose-strain scale of the reference actually used, so they are
# library-specific. Scoring with a different --reference (e.g. `ligboundconf` alone, whose max-pool scale
# runs lower than the union's) warrants recalibration — see calibrate_shoulder().
REWARD_PLATEAU = 3.0     # strain <= this is "crystallographic" (~p76 of crystals) -> full reward
REWARD_KNEE = 4.95       # crystal p95 -> r = 0.5
REWARD_BETA = 8.0        # softplus corner sharpness (larger = sharper plateau edge)


def _shoulder_excess(x, plateau: float, beta: float):
    """softplus(beta*(x - plateau)) / beta — a smooth max(0, x - plateau). NaN propagates quietly."""
    with np.errstate(invalid="ignore"):
        return np.logaddexp(0.0, beta * (np.asarray(x, dtype=float) - plateau)) / beta


def pose_reward(pose_strain, plateau: float = REWARD_PLATEAU, knee: float = REWARD_KNEE,
                beta: float = REWARD_BETA):
    """Map a pose strain energy to a smooth reward in [0, 1] with a plateau at 1 in the crystallographic
    region (see the calibration note above). Full reward for strain <= `plateau`, r = 0.5 at `knee`,
    decaying toward 0 for highly-strained poses. Accepts a scalar or array; NaN in -> NaN out. Higher
    reward = lower strain."""
    lam = np.log(2.0) / _shoulder_excess(knee, plateau, beta)
    r = np.exp(-lam * _shoulder_excess(pose_strain, plateau, beta))
    return float(r) if np.ndim(r) == 0 else r


def calibrate_shoulder(crystal_energies, plateau_pct: float = 0.76, knee_pct: float = 0.95,
                       beta: float = REWARD_BETA) -> Dict:
    """Derive shoulder anchors for a chosen reference from the pose strains of REAL crystal ligands scored
    with THAT reference: plateau = the `plateau_pct` percentile (default p76), knee = the `knee_pct`
    percentile (default p95). Returns a dict usable directly as **kwargs to pose_reward()."""
    s = np.asarray(crystal_energies, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("no finite crystal energies to calibrate on")
    return {"plateau": float(np.quantile(s, plateau_pct)),
            "knee": float(np.quantile(s, knee_pct)), "beta": float(beta)}


def available_references() -> List[str]:
    if not os.path.isdir(_BUNDLED_DIR):
        return []
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(_BUNDLED_DIR, "*.pkl")))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _quiet_rdkit():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def _load_pose(path: str, smiles: str):
    from .pose_scorer import load_any
    return load_any(path, smiles)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Torsion-library strain score for a ligand pose (PDB/SDF/MOL2 + SMILES).")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--pdb", help="single pose file (PDB/SDF/MOL2)")
    src.add_argument("--batch", help="text file of newline-separated pose paths (one shared --smiles)")
    p.add_argument("--smiles", help="ligand SMILES (bond-order template; shared by all --batch poses)")
    p.add_argument("--reference", default="ligboundconf",
                   help="'+'-joined bundled names or library paths, scored by union(max) "
                        f"(default: ligboundconf; available: {','.join(available_references()) or 'none'})")
    p.add_argument("--min-n", type=int, default=DEFAULT_MIN_N,
                   help=f"per-pattern support threshold (default {DEFAULT_MIN_N})")
    p.add_argument("--neutral-energy", type=float, default=None,
                   help="energy for torsions uncovered by all references (default: library median)")
    p.add_argument("--reward-plateau", type=float, default=REWARD_PLATEAU,
                   help=f"reward: strain plateau edge, full reward below (default {REWARD_PLATEAU})")
    p.add_argument("--reward-knee", type=float, default=REWARD_KNEE,
                   help=f"reward: strain at which reward = 0.5 (default {REWARD_KNEE}, crystal p95)")
    p.add_argument("--reward-beta", type=float, default=REWARD_BETA,
                   help=f"reward: softplus corner sharpness (default {REWARD_BETA})")
    p.add_argument("--list-references", action="store_true", help="list bundled libraries and exit")
    p.add_argument("--json", action="store_true", help="emit JSON (includes per-torsion detail)")
    p.add_argument("--out", help="batch mode: output CSV path (default stdout)")
    args = p.parse_args(argv)
    _quiet_rdkit()

    if args.list_references:
        for r in available_references():
            print(r)
        return 0
    if not (args.pdb or args.batch):
        p.error("one of --pdb or --batch is required")
    if not args.smiles:
        p.error("--smiles is required")

    lib = TorsionStrainLibrary.load(args.reference.split("+"), min_n=args.min_n,
                                    neutral_energy=args.neutral_energy)

    def score_path(path: str) -> Dict:
        try:
            mol = _load_pose(path, args.smiles)
            r = lib.score_mol(mol)
            r["path"] = path
        except Exception as e:  # unloadable pose -> NaN, never crash a batch
            r = {"path": path, "pose_strain": float("nan"), "n_torsions": 0,
                 "n_covered": 0, "error": str(e)}
        r["reward"] = pose_reward(r["pose_strain"], plateau=args.reward_plateau,
                                  knee=args.reward_knee, beta=args.reward_beta)
        return r

    if args.pdb:
        r = score_path(args.pdb)
        if args.json:
            print(json.dumps(r))
        else:
            print(f"pose_strain={r['pose_strain']:.4f}  reward={r['reward']:.4f}  "
                  f"n_torsions={r['n_torsions']}  n_covered={r['n_covered']}")
        return 0

    paths = [ln.strip() for ln in open(args.batch) if ln.strip()]
    rows = [score_path(pt) for pt in paths]
    fields = ["path", "pose_strain", "reward", "n_torsions", "n_covered", "error"]
    out = open(args.out, "w", newline="") if args.out else None
    w = csv.DictWriter(out or __import__("sys").stdout, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})
    if out:
        out.close()
        print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
