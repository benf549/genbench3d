"""Standalone torsion-strain pose scorer built on GenBench3D geometry references.

Computes the **s-value** of Baillif et al. (2024, arXiv:2407.04424): the geometric mean of
per-geometry *q-values* (a value's KDE likelihood normalised by the density's mode), against a
reference conformer library. The paper's headline s-value covers bond lengths + valence angles
(it deliberately *excludes* torsions, since binding-induced torsional strain is legitimate);
this tool's default is the mirror image — a **torsion s-value** over the freely-varying
dihedrals — validated as a torsional-strain reward for protein-ligand design poses.

Recommended (default) recipe:
    geometry = "torsion", torsions = "nonring", tiers = ("exact", "gen_outer"),
    reference = "pdbbind_full"

Input is a **PDB complex + the ligand SMILES**: the ligand (resname ``LIG`` by default) is
extracted from the PDB and its bond orders assigned from the SMILES template
(``AssignBondOrdersFromTemplate``), with a halogen/hydrogen over-bond prune for robustness.

Library use::

    from genbench3d.pose_scorer import load_ligand, s_score
    mol = load_ligand("complex.pdb", "Cc1ccccc1")
    print(s_score(mol)["s_value"])

CLI::

    sscore --pdb complex.pdb --smiles "Cc1ccccc1"
    sscore --batch poses.csv --reference pdbbind_full+/data/refs/csd_drug --json
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, BondType

from .geometry.geometry_extractor import GeometryExtractor
from .references import (ReferenceSet, resolve, available_references,
                         DEFAULT_REFERENCE, MERGE_POLICIES)
from .robust_ligand import prune_terminal_overbonds

_EXTRACTOR = GeometryExtractor()


def _quiet_rdkit():
    """Silence RDKit's app logs (e.g. the benign 'More than one matching pattern' from
    AssignBondOrdersFromTemplate on symmetric ligands). CLI/worker-only — never on import,
    so library callers keep control of RDKit logging."""
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


TORSION_SETS = ("nonring", "rotatable", "all")
GEOMETRIES = ("torsion", "bond_angle", "all")
TIER_CHOICES = ("exact", "gen_outer", "gen_inner")
DEFAULT_TIERS = ("exact", "gen_outer")


# --------------------------------------------------------------------------- ligand loading
def _n_heavy(mol):
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def _heavy_only(mol):
    """A copy of `mol` with hydrogens dropped, conformer preserved (no sanitization)."""
    rw = Chem.RWMol(mol)
    for idx in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1), reverse=True):
        rw.RemoveAtom(idx)
    return rw.GetMol()


def load_ligand(pdb_path, smiles, resname="LIG"):
    """Load a ligand conformer from a PDB (complex or ligand-only) + its SMILES.

    ``resname`` selects the ligand residue (default ``"LIG"``); if it is absent, or ``"auto"``,
    the residue fragment whose heavy-atom count best matches the SMILES template is used (the
    template match then validates the choice). Coordinates are taken as-is; bond orders come from
    the template. Robust to arbitrary poses: the H-bearing template match is tried first, then a
    heavy-atom-only match (which survives bad/missing H placement on distorted predicted poses).
    Raises ValueError only if no template match can be found at all.
    """
    full = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
    if full is None:
        raise ValueError(f"RDKit could not parse PDB: {pdb_path}")
    frags = Chem.SplitMolByPDBResidues(full)
    if not frags:
        raise ValueError(f"no residues parsed from {pdb_path}")
    graph = Chem.MolFromSmiles(smiles)
    if graph is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    if resname != "auto" and resname in frags:
        lig = frags[resname]
    else:  # fragment closest in heavy-atom count to the template (validated by the match below)
        target = graph.GetNumHeavyAtoms()
        lig = frags[min(frags, key=lambda r: abs(_n_heavy(frags[r]) - target))]
    lig = prune_terminal_overbonds(lig)

    try:  # attempt 1: full (H-bearing) template
        bound = AllChem.AssignBondOrdersFromTemplate(Chem.AddHs(graph), lig)
        Chem.SanitizeMol(bound)
        return bound
    except Exception:
        pass
    try:  # attempt 2: heavy-atom-only template (robust to H placement on distorted poses)
        bound = AllChem.AssignBondOrdersFromTemplate(graph, _heavy_only(lig))
        Chem.SanitizeMol(bound)
        return bound
    except Exception as e:
        raise ValueError(f"no template match for ligand in {pdb_path}: {e}")


def _sidecar_smiles(path):
    smi_path = os.path.splitext(path)[0] + ".smi"
    if os.path.exists(smi_path):
        parts = open(smi_path).read().strip().split()
        return parts[0] if parts else None
    return None


def load_any(path, smiles=None, resname="LIG"):
    """Load a molecule from a path. SDF/MOL/MOL2 load with bond orders directly; a PDB uses the
    given SMILES (or a ``<stem>.smi`` sidecar) for template bond-order assignment."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".sdf", ".mol"):
        m = next((x for x in Chem.SDMolSupplier(path, removeHs=False, sanitize=True) if x), None)
        if m is None:
            raise ValueError(f"no valid molecule in {path}")
        return m
    if ext == ".mol2":
        m = Chem.MolFromMol2File(path, removeHs=False, sanitize=True)
        if m is None:
            raise ValueError(f"could not parse MOL2 {path}")
        return m
    if ext in (".pdb", ".ent"):
        smi = smiles or _sidecar_smiles(path)
        if not smi:
            raise ValueError(f"PDB input needs a SMILES (arg/2nd column or {os.path.splitext(path)[0]}.smi)")
        return load_ligand(path, smi, resname=resname)
    raise ValueError(f"unsupported extension {ext!r} for {path} (use .sdf/.mol2/.pdb)")


def _read_paths(txt_path):
    """Parse a batch .txt: one entry per line, ``path`` or ``path,smiles`` /
    ``path<whitespace>smiles``. Blank lines and lines starting with '#' are ignored."""
    rows = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                path, smi = line.split(",", 1)
                path, smi = path.strip(), (smi.strip() or None)
            else:
                parts = line.split(None, 1)
                path, smi = parts[0], (parts[1].strip() if len(parts) > 1 else None)
            rows.append((path, smi))
    return rows


# --------------------------------------------------------------------------- scoring
def _gmean(qs):
    q = np.asarray(qs, dtype=float)
    q = q[~np.isnan(q)]
    if not len(q):
        return float("nan")
    return float(np.exp(np.log(np.clip(q, 1e-9, 1.0)).mean()))


def _torsion_selected(bond, mode):
    if mode == "all":
        return True
    if bond.IsInRing():
        return False
    if mode == "nonring":
        return True
    if mode == "rotatable":
        return bond.GetBondType() == BondType.SINGLE
    raise ValueError(f"unknown torsions mode {mode!r} (expected one of {TORSION_SETS})")


def s_score(mol, reference=DEFAULT_REFERENCE, geometry="torsion", torsions="nonring",
            tiers=DEFAULT_TIERS, policy="primary", per_geometry=False):
    """Score one conformer and return a result dict with ``s_value`` and ``n_geometries``.

    ``reference`` may be a spec string (``resolve``d here) or a pre-built
    :class:`~genbench3d.references.ReferenceSet` (reuse across many poses / pool workers).
    ``geometry``: ``"torsion"`` (default), ``"bond_angle"`` (the paper's validity s-value), or
    ``"all"``. ``torsions``: ``"nonring"`` (default), ``"rotatable"``, or ``"all"`` — applies
    only to torsion geometries. ``per_geometry=True`` attaches the per-geometry q breakdown.
    """
    refset = reference if isinstance(reference, ReferenceSet) else resolve(reference, policy=policy)
    tiers = frozenset(tiers)
    conf = mol.GetConformer()
    qs = []
    details = [] if per_geometry else None

    def emit(gtype, pattern, value, atom_ids):
        if value is None or (isinstance(value, float) and value != value):
            return
        q, label, tier = refset.geometry_q(pattern, value, gtype, tiers)
        if q == q:  # not nan
            qs.append(q)
            if per_geometry:
                details.append(dict(geometry=gtype, q=q, tier=tier, reference=label,
                                    atom_ids=list(atom_ids), pattern=pattern.to_string()))

    if geometry in ("bond_angle", "all"):
        for bond in mol.GetBonds():
            emit("bond", _EXTRACTOR.get_bond_pattern(bond),
                 _EXTRACTOR.get_bond_length(conf, bond),
                 (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        for i, j, k in _EXTRACTOR.get_angles_atom_ids(mol):
            emit("angle", _EXTRACTOR.get_angle_pattern(mol, i, j, k),
                 _EXTRACTOR.get_angle_value(conf, i, j, k), (i, j, k))
    if geometry in ("torsion", "all"):
        for b, s, t, e in _EXTRACTOR.get_torsions_atom_ids(mol):
            bond = mol.GetBondBetweenAtoms(int(s), int(t))
            if not _torsion_selected(bond, torsions):
                continue
            emit("torsion", _EXTRACTOR.get_torsion_pattern(mol, b, s, t, e),
                 _EXTRACTOR.get_torsion_value(conf, b, s, t, e), (b, s, t, e))

    result = dict(s_value=_gmean(qs), n_geometries=len(qs), geometry=geometry,
                  torsions=torsions, tiers=sorted(tiers), references=refset.labels,
                  policy=refset.policy)
    if per_geometry:
        result["per_geometry"] = details
    return result


# --------------------------------------------------------------------------- batch (pool)
_WORKER = {}


def _winit(spec, policy, gen, geometry, torsions, tiers, resname, batch_smiles):
    _quiet_rdkit()
    _WORKER["refset"] = resolve(spec, policy=policy, use_generalized_patterns=gen)
    _WORKER.update(geometry=geometry, torsions=torsions, tiers=tiers, resname=resname,
                   batch_smiles=batch_smiles)


def _wscore(row):
    path, smi = row
    smi = smi or _WORKER.get("batch_smiles")   # per-line smiles overrides the shared batch --smiles
    try:
        mol = load_any(path, smi, resname=_WORKER["resname"])
        r = s_score(mol, reference=_WORKER["refset"], geometry=_WORKER["geometry"],
                    torsions=_WORKER["torsions"], tiers=_WORKER["tiers"])
        return dict(path=path, s_value=r["s_value"], n_geometries=r["n_geometries"], error="")
    except Exception as e:  # noqa: BLE001 — an unloadable/unscoreable pose gets the worst score (0)
        return dict(path=path, s_value=0.0, n_geometries=0, error=f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- CLI
def _build_parser():
    p = argparse.ArgumentParser(
        prog="sscore",
        description="Torsion-strain (s-value) pose scorer over GenBench3D geometry references.")
    # not required at the argparse level so --list-references can run alone; validated in main()
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--pdb", help="complex/ligand PDB file (with --smiles) for a single pose")
    src.add_argument("--batch", help=".txt of newline-separated file paths (path or 'path,smiles' per line)")
    p.add_argument("--smiles", help="ligand SMILES; required with --pdb. With --batch it is applied to "
                   "EVERY path (a per-line 'path,smiles' overrides it) — for a batch of one target ligand.")
    p.add_argument("--reference", default=DEFAULT_REFERENCE,
                   help="reference spec; '+' merges (e.g. 'pdbbind_full+/data/refs/csd_drug'). "
                        "Presets: pdbbind_full, ligboundconf. Default: %(default)s")
    p.add_argument("--policy", default="primary", choices=MERGE_POLICIES,
                   help="merge policy when >1 reference (default: %(default)s)")
    p.add_argument("--geometry", default="torsion", choices=GEOMETRIES,
                   help="which geometries enter the s-value (default: %(default)s)")
    p.add_argument("--torsions", default="nonring", choices=TORSION_SETS,
                   help="torsion subset for torsion geometries (default: %(default)s)")
    p.add_argument("--tiers", default=",".join(DEFAULT_TIERS),
                   help="comma-separated generalization tiers from "
                        f"{TIER_CHOICES} (default: %(default)s)")
    p.add_argument("--no-generalize", action="store_true",
                   help="disable generalized-pattern lookup entirely (exact patterns only)")
    p.add_argument("--resname", default="LIG",
                   help="ligand residue name in the PDB, or 'auto' (default: %(default)s)")
    p.add_argument("--per-geometry", action="store_true",
                   help="include the per-geometry q breakdown (single-pose only)")
    p.add_argument("--json", action="store_true", help="emit JSON instead of text")
    p.add_argument("--out", help="write batch results to this CSV (default: stdout)")
    p.add_argument("--nproc", type=int, default=1, help="worker processes for --batch")
    p.add_argument("--list-references", action="store_true",
                   help="print available reference aliases and exit")
    return p


def main(argv=None):
    _quiet_rdkit()
    args = _build_parser().parse_args(argv)
    if args.list_references:
        print(json.dumps(available_references(), indent=2))
        return 0
    if not args.pdb and not args.batch:
        _build_parser().error("one of the arguments --pdb --batch is required")

    tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip())
    bad = [t for t in tiers if t not in TIER_CHOICES]
    if bad:
        _build_parser().error(f"unknown tier(s) {bad}; choose from {TIER_CHOICES}")
    gen = not args.no_generalize

    if args.pdb:
        if not args.smiles:
            _build_parser().error("--smiles is required with --pdb")
        refset = resolve(args.reference, policy=args.policy, use_generalized_patterns=gen)
        err = ""
        try:
            mol = load_any(args.pdb, args.smiles, resname=args.resname)
            r = s_score(mol, reference=refset, geometry=args.geometry, torsions=args.torsions,
                        tiers=tiers, per_geometry=args.per_geometry)
        except Exception as e:  # unloadable/unscoreable pose -> worst score (0)
            err = f"{type(e).__name__}: {e}"
            r = dict(s_value=0.0, n_geometries=0, geometry=args.geometry, torsions=args.torsions,
                     tiers=sorted(frozenset(tiers)), references=[args.reference], policy=args.policy)
        if args.json:
            print(json.dumps({**r, "error": err} if err else r, indent=2))
        else:
            print(f"s_value        {r['s_value']:.4f}")
            print(f"n_geometries   {r['n_geometries']}")
            print(f"geometry       {r['geometry']} ({r['torsions']} torsions, "
                  f"tiers={'+'.join(r['tiers'])})")
            print(f"reference      {'+'.join(r['references'])} (policy={r['policy']})")
            if err:
                print(f"note           could not load pose -> s_value 0  ({err})")
        return 0

    # batch: a .txt of file paths (one target ligand via --smiles; per-line 'path,smiles' overrides)
    rows = _read_paths(args.batch)
    winit_args = (args.reference, args.policy, gen, args.geometry, args.torsions, tiers,
                  args.resname, args.smiles)
    if args.nproc > 1:
        from multiprocessing import Pool
        with Pool(args.nproc, initializer=_winit, initargs=winit_args) as pool:
            results = list(pool.imap(_wscore, rows, chunksize=8))
    else:
        _winit(*winit_args)
        results = [_wscore(row) for row in rows]

    fields = ["path", "s_value", "n_geometries", "error"]
    if args.json:
        payload = json.dumps(results, indent=2)
        (open(args.out, "w").write(payload) if args.out else sys.stdout.write(payload + "\n"))
    else:
        out = open(args.out, "w", newline="") if args.out else sys.stdout
        w = csv.DictWriter(out, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)
        if args.out:
            out.close()
    n_loaded = sum(1 for r in results if not r["error"])
    print(f"# scored {len(results)} poses ({n_loaded} loaded, {len(results) - n_loaded} -> s_value 0)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
