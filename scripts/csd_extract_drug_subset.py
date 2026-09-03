#!/usr/bin/env python
"""Extract the CSD Drug Subset (from a .gcd refcode list) to a concatenated MOL2 file,
mirroring GenBench3D's CSDDrug source prep. The RDKit-side prep (largest-fragment +
zero-order-bond filter + sanitize) happens later in build_reference.py --csd-prep.

Runs in the CCDC csd-python-api python (has `ccdc`, NOT rdkit). Required env (put in .env at
the repo root, or export in the shell):
    CSDHOME=/path/to/ccdc-data
    CCDC_PYTHON_API_NO_QAPPLICATION=1
    # CCDC_LICENSING_CONFIGURATION=la-code:<key>   # ONLY if the host is not already activated;
    #                                              # setting it can override & break an activated host.

The CSD data and this MOL2 dump are CCDC-licensed and must NOT be committed to the repo.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gb3d_env import load_env, require

load_env()
require("CSDHOME")
os.environ.setdefault("CCDC_PYTHON_API_NO_QAPPLICATION", "1")

from ccdc.io import MoleculeReader  # noqa: E402 — import after env is set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcd", required=True, help=".gcd refcode list (one CSD refcode per line)")
    ap.add_argument("--out", default="refdata/csd_drug/CSD_Drug_Subset.mol2",
                    help="output MOL2 path (licensed data — keep out of git)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ids = [ln.strip() for ln in open(args.gcd) if ln.strip()]
    if args.limit:
        ids = ids[: args.limit]

    reader = MoleculeReader("CSD")
    n_ok = n_fail = 0
    missing = []
    with open(args.out, "w") as out:
        for i, rc in enumerate(ids):
            try:
                m = reader.molecule(rc)
                m.remove_atoms([a for a in m.atoms if a.atomic_number < 1])  # drop fake atoms
                block = m.to_string("mol2")
                out.write(block)
                if not block.endswith("\n"):
                    out.write("\n")
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                if len(missing) < 40:
                    missing.append(f"{rc}:{type(e).__name__}")
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(ids)} ok={n_ok} fail={n_fail}", flush=True)

    print(f"DONE ok={n_ok} fail={n_fail} total={len(ids)} -> {args.out}", flush=True)
    if missing:
        print("first failures:", ", ".join(missing), flush=True)


if __name__ == "__main__":
    main()
