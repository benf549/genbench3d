#!/usr/bin/env python
"""Build a torsion-strain library from a reference set of 3D small-molecule structures.

Input may be an SDF, a MOL2 (multi-record), or a directory/glob of SDFs (e.g. PDBBind '*_ligand.sdf').
Output is a pickle consumable by genbench3d.torsion_strain (bundle it under
genbench3d/data/torsion_libs/<name>.pkl, or point --reference at the path).

The library stores, per coarse torsion pattern, a smoothed dihedral-angle distribution plus its support
count N; the support threshold (MIN_N) is applied at score time, so it need not be chosen here.

Examples
--------
  # LigBoundConf (public) -> bundled default
  build_torsion_lib.py --in S2_LigBoundConf_minimized.sdf --name ligboundconf \
      --out genbench3d/data/torsion_libs/ligboundconf.pkl

  # PDBBind crystal ligands (open, per its own terms)
  build_torsion_lib.py --glob '/path/PDBBind/**/*_ligand.sdf' --name pdbbind \
      --out genbench3d/data/torsion_libs/pdbbind.pkl

  # A CCDC-licensed CSD subset produced by scripts/curate_csd_subset.py (NOT redistributable)
  build_torsion_lib.py --in refdata/csd_custom/CSD_custom.mol2 --name csd_custom \
      --out refdata/csd_custom/csd_custom.pkl
"""
import argparse
import glob
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genbench3d.torsion_strain import build_library  # noqa: E402


def iter_sdf(path):
    from rdkit import Chem
    for m in Chem.SDMolSupplier(path, removeHs=False, sanitize=True):
        if m is not None:
            yield m


def iter_sdf_glob(pattern):
    from rdkit import Chem
    for f in sorted(glob.glob(pattern, recursive=True)):
        m = next((x for x in Chem.SDMolSupplier(f, removeHs=False, sanitize=True) if x is not None), None)
        if m is not None:
            yield m


def iter_mol2(path):
    """Stream records from a multi-molecule MOL2 (RDKit has no multi-MOL2 supplier)."""
    from rdkit import Chem
    buf = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("@<TRIPOS>MOLECULE") and buf:
                m = Chem.MolFromMol2Block("".join(buf), removeHs=False)
                if m is not None and m.GetNumConformers():
                    yield m
                buf = [line]
            else:
                buf.append(line)
    if buf:
        m = Chem.MolFromMol2Block("".join(buf), removeHs=False)
        if m is not None and m.GetNumConformers():
            yield m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--in", dest="inp", help="single SDF or multi-record MOL2")
    g.add_argument("--glob", help="glob of SDFs (recursive), one molecule taken per file")
    ap.add_argument("--name", required=True, help="library name recorded in meta")
    ap.add_argument("--out", required=True, help="output .pkl path")
    args = ap.parse_args()

    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    if args.glob:
        it = iter_sdf_glob(args.glob)
    elif args.inp.lower().endswith(".mol2"):
        it = iter_mol2(args.inp)
    else:
        it = iter_sdf(args.inp)

    lib = build_library(it, source_name=args.name)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pickle.dump(lib, open(args.out, "wb"))
    m = lib["meta"]
    print(f"built '{args.name}' from {m['n_molecules']} molecules: "
          f"{m['n_patterns_4atom']} 4-atom + {m['n_patterns_2atom']} 2-atom patterns, "
          f"neutral_energy={m['neutral_energy']:.2f} -> {args.out}")


if __name__ == "__main__":
    main()
