"""Standalone GenBench3D **Validity3D** pose checker (+ batch CLI).

Validity3D (Baillif et al. 2024) flags a conformer as 3D-valid when, against a reference
conformer library:
  * every **bond length** is geometrically valid  (q-value > threshold, or an unseen pattern),
  * every **valence angle** is valid,
  * no **steric clash** (intramolecular, van der Waals * safety ratio, PoseBusters-style), and
  * no **puckered aromatic/planar ring** (max atom-to-mean-plane distance < tolerance).
**Torsions are deliberately excluded** from the gate (binding-induced torsional strain is
legitimate) — they are available as a separate strain s-value (see genbench3d.pose_scorer).

This reimplements Validity3D on the geometry primitives (GeometryExtractor + ClashChecker +
ReferenceGeometry.geometry_is_valid) so it stays dependency-light (no conf-ensemble / molvs /
nglview), matching upstream's logic and default params. Generalized patterns are OFF for
validity (the upstream author's 12-2025 fix: generalization can underestimate Validity3D).

Batch CLI takes a **.txt of newline-separated file paths** (SDF/MOL2 load directly; PDB needs a
SMILES via a 2nd column or a `<stem>.smi` sidecar) and writes a CSV of validity results + a few
useful extra features (validity s-value, per-type invalid counts, clash count, torsion strain).
"""
import argparse
import csv
import os
import json
import sys

import numpy as np
from rdkit import Chem

from .geometry.geometry_extractor import GeometryExtractor
from .geometry.clash_checker import ClashChecker
from .references import load_reference, ReferenceSet, DEFAULT_REFERENCE
from .pose_scorer import load_ligand, load_any, _read_paths, s_score, _gmean, _quiet_rdkit

# Validity3D defaults — genbench3d/params.py (Q_VALUE_THRESHOLD, CLASH_SAFETY_RATIO, ...).
Q_VALUE_THRESHOLD = 0.001
CLASH_SAFETY_RATIO = 0.75
MAX_RING_PLANE_DISTANCE = 0.1
CONSIDER_HYDROGENS = False
INCLUDE_TORSIONS = False

_EXTRACTOR = GeometryExtractor()


# --------------------------------------------------------------------------- geometry checks
def _max_distance_to_plane(conf, ring_atom_ids):
    """Max atom distance to the ring's best-fit plane (SVD), per PoseBusters flatness."""
    coords = conf.GetPositions()[list(ring_atom_ids), :]
    centred = coords - coords.mean(axis=0)
    _, _, V = np.linalg.svd(centred)
    normal = V[-1]
    return float(np.max(np.dot(centred, normal)))


def _bond_angle_validity(ref, mol, consider_hs):
    conf = mol.GetConformer()
    res = {"bond": {"invalid": [], "qs": []}, "angle": {"invalid": [], "qs": []}}
    new_patterns = []
    for bond in _EXTRACTOR.get_bonds(mol, consider_hydrogens=consider_hs):
        P = _EXTRACTOR.get_bond_pattern(bond)
        v = _EXTRACTOR.get_bond_length(conf, bond)
        q, new = ref.geometry_is_valid(P, v, geometry="bond")
        if new:
            new_patterns.append(("bond", P.to_string()))
        else:
            res["bond"]["qs"].append(q)
            if not (q > Q_VALUE_THRESHOLD):
                res["bond"]["invalid"].append({"pattern": P.to_string(), "value": v, "q": q})
    for i, j, k in _EXTRACTOR.get_angles_atom_ids(mol, consider_hydrogens=consider_hs):
        P = _EXTRACTOR.get_angle_pattern(mol, i, j, k)
        v = _EXTRACTOR.get_angle_value(conf, i, j, k)
        q, new = ref.geometry_is_valid(P, v, geometry="angle")
        if new:
            new_patterns.append(("angle", P.to_string()))
        else:
            res["angle"]["qs"].append(q)
            if not (q > Q_VALUE_THRESHOLD):
                res["angle"]["invalid"].append({"pattern": P.to_string(), "value": v, "q": q})
    return res, new_patterns


def _puckered_rings(mol):
    conf = mol.GetConformer()
    out = []
    for atom_ids in _EXTRACTOR.get_planar_rings_atom_ids(mol):
        d = _max_distance_to_plane(conf, atom_ids)
        if not (d < MAX_RING_PLANE_DISTANCE):
            out.append({"ring_size": len(atom_ids), "max_plane_distance": d})
    return out


def _clashes(mol, consider_hs, clash_checker):
    # Mirror Validity3D.analyze_clashes exactly, incl. RemoveHs + 1-2/1-3/1-4 exclusions.
    m = mol if consider_hs else Chem.RemoveHs(mol)
    bond_idxs = [tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
                 for b in _EXTRACTOR.get_bonds(m, consider_hydrogens=consider_hs)]
    two_hop = [tuple(sorted((t[0], t[2])))
               for t in _EXTRACTOR.get_angles_atom_ids(m, consider_hydrogens=consider_hs)]
    three_hop = [tuple(sorted((t[0], t[3])))
                 for t in _EXTRACTOR.get_torsions_atom_ids(m, consider_hydrogens=consider_hs)]
    excluded = bond_idxs + two_hop + three_hop
    conf_id = m.GetConformer().GetId()
    return clash_checker.get_clashes(m, conf_id, excluded)


def evaluate_validity(mol, reference=DEFAULT_REFERENCE, consider_hydrogens=CONSIDER_HYDROGENS,
                      include_torsions=INCLUDE_TORSIONS, with_strain=True, per_geometry=False):
    """Validity3D result for one conformer. ``reference`` is a spec string, a
    ``(label, ReferenceGeometry)`` tuple, or a ReferenceGeometry. Returns a result dict."""
    if isinstance(reference, str):
        reference = load_reference(reference.split("+")[0], use_generalized_patterns=False)
    label, ref = reference if isinstance(reference, tuple) else (getattr(reference, "source", None) and reference.source.name, reference)

    clash_checker = ClashChecker(safety_ratio=CLASH_SAFETY_RATIO, consider_hs=consider_hydrogens)
    prev_gen = ref.use_generalized_patterns
    ref.use_generalized_patterns = False  # validity: no generalization (author's 12-2025 fix)

    ba, new_patterns = _bond_angle_validity(ref, mol, consider_hydrogens)
    puckered = _puckered_rings(mol)
    clashes = _clashes(mol, consider_hydrogens, clash_checker)

    tors_invalid = []
    if include_torsions:
        conf = mol.GetConformer()
        for b, s, t, e in _EXTRACTOR.get_torsions_atom_ids(mol, consider_hydrogens=consider_hydrogens):
            P = _EXTRACTOR.get_torsion_pattern(mol, b, s, t, e)
            v = _EXTRACTOR.get_torsion_value(conf, b, s, t, e)
            q, new = ref.geometry_is_valid(P, v, geometry="torsion")
            if not new and not (q > Q_VALUE_THRESHOLD):
                tors_invalid.append({"pattern": P.to_string(), "value": v, "q": q})

    n_ib, n_ia = len(ba["bond"]["invalid"]), len(ba["angle"]["invalid"])
    n_ring, n_clash, n_it = len(puckered), len(clashes), len(tors_invalid)
    valid = (n_ib == 0 and n_ia == 0 and n_ring == 0 and n_clash == 0
             and (n_it == 0 if include_torsions else True))

    strain = float("nan")
    if with_strain:
        ref.use_generalized_patterns = True  # strain reward uses exact+gen_outer
        strain = s_score(mol, reference=ReferenceSet([(label or "ref", ref)]),
                         geometry="torsion", torsions="nonring",
                         tiers=("exact", "gen_outer"))["s_value"]
    ref.use_generalized_patterns = prev_gen

    result = {
        "valid": valid,
        "n_invalid_bonds": n_ib,
        "n_invalid_angles": n_ia,
        "n_puckered_rings": n_ring,
        "n_clashes": n_clash,
        "n_new_patterns": len(new_patterns),
        "validity_s_value": _gmean(ba["bond"]["qs"] + ba["angle"]["qs"]),
        "bond_s_value": _gmean(ba["bond"]["qs"]),
        "angle_s_value": _gmean(ba["angle"]["qs"]),
        "min_bond_q": min(ba["bond"]["qs"]) if ba["bond"]["qs"] else float("nan"),
        "min_angle_q": min(ba["angle"]["qs"]) if ba["angle"]["qs"] else float("nan"),
        "torsion_strain_s_value": strain,
        "n_heavy_atoms": mol.GetNumHeavyAtoms(),
        "reference": label,
    }
    if include_torsions:
        result["n_invalid_torsions"] = n_it
    if per_geometry:
        result["invalid"] = {"bond": ba["bond"]["invalid"], "angle": ba["angle"]["invalid"],
                             "ring": puckered, "torsion": tors_invalid,
                             "clashes": [c._asdict() for c in clashes],
                             "new_patterns": new_patterns}
    return result


# --------------------------------------------------------------------------- batch worker
_W = {}


def _winit(spec, consider_hs, include_torsions, with_strain, resname, batch_smiles):
    _quiet_rdkit()
    _W["reference"] = load_reference(spec.split("+")[0], use_generalized_patterns=False)
    _W.update(consider_hs=consider_hs, include_torsions=include_torsions,
              with_strain=with_strain, resname=resname, batch_smiles=batch_smiles)


CSV_FIELDS = ["path", "valid", "n_invalid_bonds", "n_invalid_angles", "n_puckered_rings",
              "n_clashes", "n_new_patterns", "validity_s_value", "min_bond_q", "min_angle_q",
              "torsion_strain_s_value", "n_heavy_atoms", "error"]


def _weval(row):
    path, smi = row
    smi = smi or _W.get("batch_smiles")   # per-line smiles overrides the shared batch --smiles
    base = {k: "" for k in CSV_FIELDS}
    base["path"] = path
    try:
        mol = load_any(path, smi, resname=_W["resname"])
        r = evaluate_validity(mol, reference=_W["reference"], consider_hydrogens=_W["consider_hs"],
                              include_torsions=_W["include_torsions"], with_strain=_W["with_strain"])
        for k in CSV_FIELDS:
            if k in r:
                base[k] = r[k]
        base["error"] = ""
    except Exception as e:  # noqa: BLE001 — one bad file must not kill the batch
        base["error"] = f"{type(e).__name__}: {e}"
        base["valid"] = ""
    return base


# --------------------------------------------------------------------------- CLI
def _build_parser():
    from .references import available_references  # noqa: F401 (kept for --list-references)
    p = argparse.ArgumentParser(
        prog="pose-validity",
        description="GenBench3D Validity3D pose checker (bonds+angles valid, no clash, no "
                    "puckered ring). Torsions are excluded from validity (reported as strain).")
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--batch", help=".txt of newline-separated file paths (path[,smiles] per line)")
    src.add_argument("--pdb", help="single complex PDB (with --smiles)")
    src.add_argument("--sdf", help="single SDF/MOL2 file (bond orders from the file)")
    p.add_argument("--smiles", help="ligand SMILES; required with --pdb. With --batch it is applied to "
                   "EVERY path (a per-line 'path,smiles' overrides it) — for a batch of one target ligand.")
    p.add_argument("--reference", default=DEFAULT_REFERENCE,
                   help="reference for bond/angle validity (single; default: %(default)s)")
    p.add_argument("--consider-hydrogens", action="store_true",
                   help="include hydrogens in bond/angle/clash checks (default: heavy atoms only)")
    p.add_argument("--include-torsions", action="store_true",
                   help="also gate validity on torsions (default off — the paper excludes them)")
    p.add_argument("--no-strain", action="store_true",
                   help="skip the torsion strain s-value extra feature")
    p.add_argument("--resname", default="LIG", help="ligand residue name in PDB inputs, or 'auto'")
    p.add_argument("--per-geometry", action="store_true", help="list the invalid geometries (single/JSON)")
    p.add_argument("--json", action="store_true", help="emit JSON")
    p.add_argument("--out", help="write batch CSV/JSON here (default: stdout)")
    p.add_argument("--nproc", type=int, default=1, help="worker processes for --batch")
    p.add_argument("--list-references", action="store_true")
    return p


def main(argv=None):
    _quiet_rdkit()
    args = _build_parser().parse_args(argv)
    if args.list_references:
        from .references import available_references
        print(json.dumps(available_references(), indent=2))
        return 0
    if not (args.batch or args.pdb or args.sdf):
        _build_parser().error("one of --batch, --pdb, or --sdf is required")

    with_strain = not args.no_strain

    if args.pdb or args.sdf:
        path = args.pdb or args.sdf
        if args.pdb and not args.smiles:
            _build_parser().error("--smiles is required with --pdb")
        reference = load_reference(args.reference.split("+")[0], use_generalized_patterns=False)
        mol = load_any(path, args.smiles, resname=args.resname)
        r = evaluate_validity(mol, reference=reference, consider_hydrogens=args.consider_hydrogens,
                              include_torsions=args.include_torsions, with_strain=with_strain,
                              per_geometry=args.per_geometry)
        if args.json:
            print(json.dumps(r, indent=2, default=str))
        else:
            print(f"valid                    {r['valid']}")
            print(f"invalid bonds/angles     {r['n_invalid_bonds']} / {r['n_invalid_angles']}")
            print(f"puckered rings           {r['n_puckered_rings']}")
            print(f"steric clashes           {r['n_clashes']}")
            print(f"new (unseen) patterns    {r['n_new_patterns']}")
            print(f"validity s-value (b+a)   {r['validity_s_value']:.4f}  "
                  f"(min bond q {r['min_bond_q']:.4g}, min angle q {r['min_angle_q']:.4g})")
            if with_strain:
                print(f"torsion strain s-value   {r['torsion_strain_s_value']:.4f}")
            print(f"reference                {r['reference']}")
        return 0

    # batch
    rows = _read_paths(args.batch)
    winit = (args.reference, args.consider_hydrogens, args.include_torsions, with_strain,
             args.resname, args.smiles)
    if args.nproc > 1:
        from multiprocessing import Pool
        with Pool(args.nproc, initializer=_winit, initargs=winit) as pool:
            results = list(pool.imap(_weval, rows, chunksize=8))
    else:
        _winit(*winit)
        results = [_weval(r) for r in rows]

    if args.json:
        payload = json.dumps(results, indent=2, default=str)
        (open(args.out, "w").write(payload) if args.out else sys.stdout.write(payload + "\n"))
    else:
        out = open(args.out, "w", newline="") if args.out else sys.stdout
        w = csv.DictWriter(out, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
        if args.out:
            out.close()
    n_ok = sum(1 for r in results if not r["error"])
    n_valid = sum(1 for r in results if r.get("valid") is True)
    print(f"# evaluated {n_ok}/{len(results)} files; {n_valid} valid", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
