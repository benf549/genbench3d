#!/usr/bin/env python
"""Build a GenBench3D geometry reference (values + KDE pickles + pattern counts) from a molecule
set, so the pose scorer can use it — bundled, or via ``--reference <dir>`` / a registered alias.

Requires the package installed (`pip install .`), so `import genbench3d` works with no PYTHONPATH.

Inputs (choose one):
    --sdf FILE           one multi-molecule SDF            (e.g. LigBoundConf S2_*.sdf)
    --sdf-glob 'PAT'     glob of many SDFs                 (e.g. PDBBind '<db>/**/*_ligand.sdf')
    --mol2 FILE          one multi-molecule MOL2           (e.g. a CSD dump; use --csd-prep)

Required:
    --name NAME          reference source name -> <NAME>_geometry_*.p
    --out DIR            output directory

Options:
    --exclude-holo-dir D (sdf-glob) hold out any ligand whose parent-dir basename matches a
                         '<id>_holo.pdb' in D (how the scored complexes were held out)
    --csd-prep           LargestFragmentChooser + drop zero-order-bond mols (CSD/MOL2 prep)
    --min-pattern N      minimum examples per pattern for a KDE (default 50)

Then chunk the KDE pickle for git with:  python scripts/chunk_weights.py --src DIR --name NAME --out <weights_subdir>
"""
import argparse
import glob
import json
import os
import time

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from genbench3d.data.source import MolListSource
from genbench3d.geometry import ReferenceGeometry


def load_sdf(path):
    return [m for m in Chem.SDMolSupplier(path, removeHs=False, sanitize=True) if m is not None]


def load_sdf_glob(pattern, exclude_holo_dir=None):
    held = set()
    if exclude_holo_dir:
        held = {os.path.basename(p)[: -len("_holo.pdb")]
                for p in glob.glob(f"{exclude_holo_dir}/*_holo.pdb")}
    files = [f for f in glob.glob(pattern, recursive=True)
             if os.path.basename(os.path.dirname(f)) not in held]
    mols = []
    for i, f in enumerate(files):
        m = next((x for x in Chem.SDMolSupplier(f, removeHs=False, sanitize=True) if x), None)
        if m is not None:
            mols.append(m)
        if (i + 1) % 3000 == 0:
            print(f"  {i+1}/{len(files)} ({len(mols)} ok)", flush=True)
    print(f"held out {len(held)} ids; loaded {len(mols)}/{len(files)} sdfs", flush=True)
    return mols


def load_mol2(path, csd_prep):
    txt = open(path).read()
    blocks = ["@<TRIPOS>MOLECULE" + b for b in txt.split("@<TRIPOS>MOLECULE") if b.strip()]
    lfc = None
    if csd_prep:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        lfc = rdMolStandardize.LargestFragmentChooser()
    mols = []
    for b in blocks:
        m = Chem.MolFromMol2Block(b, removeHs=False)
        if m is None:
            continue
        if csd_prep:
            try:
                m = lfc.choose(m)
            except Exception:
                continue
            if any(bd.GetBondTypeAsDouble() == 0.0 for bd in m.GetBonds()):
                continue
        mols.append(m)
    print(f"mol2 blocks={len(blocks)} -> mols={len(mols)}", flush=True)
    return mols


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sdf")
    g.add_argument("--sdf-glob")
    g.add_argument("--mol2")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-holo-dir")
    ap.add_argument("--csd-prep", action="store_true")
    ap.add_argument("--min-pattern", type=int, default=50)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if a.sdf:
        mols = load_sdf(a.sdf)
    elif a.sdf_glob:
        mols = load_sdf_glob(a.sdf_glob, a.exclude_holo_dir)
    else:
        mols = load_mol2(a.mol2, a.csd_prep)
    print(f"loaded {len(mols)} mols in {time.time()-t0:.0f}s; building '{a.name}'...", flush=True)

    for suf in ("geometry_values.p", "geometry_kernel_densities.p"):  # force rebuild if stale
        p = os.path.join(a.out, f"{a.name}_{suf}")
        if os.path.exists(p):
            os.remove(p)

    src = MolListSource(mol_list=mols, name=a.name)
    ref = ReferenceGeometry(source=src, root=a.out, minimum_pattern_values=a.min_pattern)
    print(f"built in {time.time()-t0:.0f}s; KDE patterns kept: "
          f"{ {k: len(v) for k, v in ref.kernel_densities.items()} }", flush=True)

    vals = ref.read_values()
    counts = {gk: {p.to_string(): len(v) for p, v in vals[gk].items()}
              for gk in ("bond", "angle", "torsion")}
    with open(os.path.join(a.out, f"{a.name}_pattern_counts.json"), "w") as f:
        json.dump(counts, f)
    print("wrote pattern_counts.json; done", flush=True)


if __name__ == "__main__":
    main()
