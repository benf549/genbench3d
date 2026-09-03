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
def load_ligand(pdb_path, smiles, resname="LIG"):
    """Load a ligand pose from a complex PDB + its SMILES.

    ``resname`` selects the ligand residue (default ``"LIG"``); ``"auto"`` picks the residue
    fragment whose heavy-atom count best matches the SMILES template. Coordinates are taken
    as-is; bond orders/protonation come from the template.
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
    template = Chem.AddHs(graph)

    if resname == "auto":
        target = graph.GetNumHeavyAtoms()
        def heavy(fr):
            return sum(1 for a in fr.GetAtoms() if a.GetAtomicNum() > 1)
        rn = min(frags, key=lambda r: abs(heavy(frags[r]) - target))
        lig = frags[rn]
    else:
        if resname not in frags:
            raise ValueError(
                f"residue {resname!r} not in {pdb_path}; found {sorted(frags)[:12]} "
                f"(try --resname auto)")
        lig = frags[resname]

    lig = prune_terminal_overbonds(lig)
    bound = AllChem.AssignBondOrdersFromTemplate(template, lig)
    Chem.SanitizeMol(bound)
    return bound


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


def _winit(spec, policy, gen, geometry, torsions, tiers, resname):
    _quiet_rdkit()
    _WORKER["refset"] = resolve(spec, policy=policy, use_generalized_patterns=gen)
    _WORKER.update(geometry=geometry, torsions=torsions, tiers=tiers, resname=resname)


def _wscore(row):
    idx, pdb, smi = row
    try:
        mol = load_ligand(pdb, smi, resname=_WORKER["resname"])
        r = s_score(mol, reference=_WORKER["refset"], geometry=_WORKER["geometry"],
                    torsions=_WORKER["torsions"], tiers=_WORKER["tiers"])
        return dict(id=idx, pdb=pdb, s_value=r["s_value"], n_geometries=r["n_geometries"], error="")
    except Exception as e:  # noqa: BLE001 — one bad pose must not kill the batch
        return dict(id=idx, pdb=pdb, s_value=float("nan"), n_geometries=0,
                    error=f"{type(e).__name__}: {e}")


def _read_batch(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        if "pdb" not in cols or "smiles" not in cols:
            raise ValueError(f"batch CSV must have 'pdb' and 'smiles' columns; got {reader.fieldnames}")
        rows = []
        for n, r in enumerate(reader):
            idx = r[cols["id"]] if "id" in cols else str(n)
            rows.append((idx, r[cols["pdb"]], r[cols["smiles"]]))
    return rows


# --------------------------------------------------------------------------- CLI
def _build_parser():
    p = argparse.ArgumentParser(
        prog="sscore",
        description="Torsion-strain (s-value) pose scorer over GenBench3D geometry references.")
    # not required at the argparse level so --list-references can run alone; validated in main()
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--pdb", help="complex PDB file (with --smiles) for a single pose")
    src.add_argument("--batch", help="CSV with columns pdb,smiles[,id] for many poses")
    p.add_argument("--smiles", help="ligand SMILES (required with --pdb)")
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
        mol = load_ligand(args.pdb, args.smiles, resname=args.resname)
        r = s_score(mol, reference=refset, geometry=args.geometry, torsions=args.torsions,
                    tiers=tiers, per_geometry=args.per_geometry)
        if args.json:
            print(json.dumps(r, indent=2))
        else:
            print(f"s_value        {r['s_value']:.4f}")
            print(f"n_geometries   {r['n_geometries']}")
            print(f"geometry       {r['geometry']} ({r['torsions']} torsions, "
                  f"tiers={'+'.join(r['tiers'])})")
            print(f"reference      {'+'.join(r['references'])} (policy={r['policy']})")
        return 0

    # batch
    rows = _read_batch(args.batch)
    winit_args = (args.reference, args.policy, gen, args.geometry, args.torsions, tiers, args.resname)
    if args.nproc > 1:
        from multiprocessing import Pool
        with Pool(args.nproc, initializer=_winit, initargs=winit_args) as pool:
            results = list(pool.imap(_wscore, rows, chunksize=8))
    else:
        _winit(*winit_args)
        results = [_wscore(row) for row in rows]

    fields = ["id", "pdb", "s_value", "n_geometries", "error"]
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
    n_ok = sum(1 for r in results if r["s_value"] == r["s_value"])
    print(f"# scored {n_ok}/{len(results)} poses", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
