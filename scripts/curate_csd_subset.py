#!/usr/bin/env python
"""Curate a custom CSD subset for a torsion-strain reference — larger, higher-quality, and
UNBOUND, unlike the 785-molecule CSD Drug Subset (Bryant 2019) that GenBench3D uses.

Rationale (from our transfer/strain-detection study):
  * The Drug Subset is too small -> sparse per-pattern data -> noisy torsion energies (it LOST
    to PDBBind on physical-strain detection despite being the "right" unbound domain).
  * It's literally approved drugs -> redundant with PDBBind's drug-like ligands (97.5% overlap).
  The fix is DATA DENSITY on drug-relevant chemistry from UNBOUND crystals: many high-quality
  small-molecule structures at ambient conditions, so distributions are smooth AND reflect true
  (unstrained) equilibrium geometry that a protein-bound reference (PDBBind) can't.

Filters (all CLI-tunable):
  organic, has 3D coords, not polymeric, no disorder, R-factor <= max, ambient (skip high-pressure),
  elements within an organic/drug-relevant set, molecular-weight window, >= a few rotatable bonds.
  Only the largest fragment of each structure is kept (drops counter-ions / solvent).

Runs in the CCDC csd-python-api python (has `ccdc`, NOT rdkit). Env via .env (see .env.example):
  CSDHOME=..., CCDC_PYTHON_API_NO_QAPPLICATION=1  (+ CCDC_LICENSING_CONFIGURATION only if not
  already activated). Output MOL2 + refcode list are CSD-licensed -> keep them out of git.

Next step: build the torsion library from the MOL2 with the RDKit-side build (build_lib), then
re-run the strain-detection / transfer comparison vs PDBBind.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gb3d_env import load_env
load_env()
os.environ.setdefault("CCDC_PYTHON_API_NO_QAPPLICATION", "1")

from ccdc.io import EntryReader  # noqa: E402


DEFAULT_ELEMENTS = "H,B,C,N,O,F,Si,P,S,Cl,Se,Br,I"


def largest_component(mol):
    comps = mol.components
    if not comps:
        return mol
    return max(comps, key=lambda c: sum(1 for a in c.atoms if a.atomic_number > 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="refdata/csd_custom/CSD_custom.mol2", help="output MOL2 (licensed; keep out of git)")
    ap.add_argument("--gcd", default="refdata/csd_custom/CSD_custom.gcd", help="output refcode list")
    ap.add_argument("--max-r-factor", type=float, default=7.5, help="max R-factor %% (default 7.5)")
    ap.add_argument("--mw-min", type=float, default=150.0)
    ap.add_argument("--mw-max", type=float, default=800.0)
    ap.add_argument("--min-rot-bonds", type=int, default=2, help="min rotatable bonds (need torsions to learn)")
    ap.add_argument("--allowed-elements", default=DEFAULT_ELEMENTS)
    ap.add_argument("--allow-disorder", action="store_true", help="keep disordered structures (default: drop)")
    ap.add_argument("--max", type=int, default=250000, help="cap on kept structures")
    ap.add_argument("--scan-limit", type=int, default=0, help="stop after scanning N entries (0=all; for testing)")
    ap.add_argument("--stride", type=int, default=1, help="process 1 in every N entries (even full-DB sampling)")
    ap.add_argument("--report-every", type=int, default=50000)
    args = ap.parse_args()

    allowed = {e.strip() for e in args.allowed_elements.split(",") if e.strip()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    csd = EntryReader("CSD")
    kept = 0
    reasons = {k: 0 for k in ("no3d", "not_organic", "polymeric", "disorder", "rfactor",
                              "pressure", "elements", "mw", "rotbonds", "error")}
    n = 0
    with open(args.out, "w") as out, open(args.gcd, "w") as gcd:
        for entry in csd:
            n += 1
            if args.report_every and n % args.report_every == 0:
                print(f"  scanned {n:,} | kept {kept:,}", flush=True)
            if kept >= args.max:
                break
            if args.scan_limit and n > args.scan_limit:
                break
            if args.stride > 1 and (n % args.stride) != 0:
                continue
            try:
                if not entry.has_3d_structure: reasons["no3d"] += 1; continue
                if not entry.is_organic: reasons["not_organic"] += 1; continue
                if entry.is_polymeric: reasons["polymeric"] += 1; continue
                if (not args.allow_disorder) and entry.has_disorder: reasons["disorder"] += 1; continue
                r = entry.r_factor
                if r is None or r > args.max_r_factor: reasons["rfactor"] += 1; continue
                # ambient only: skip high-pressure (strained geometry). Pressure text is unstructured;
                # keep entries with no pressure record (ambient) and drop those mentioning 'GPa'.
                pres = (getattr(entry, "pressure", None) or "")
                if "GPa" in str(pres): reasons["pressure"] += 1; continue

                mol = largest_component(entry.molecule)
                mol.remove_atoms([a for a in mol.atoms if a.atomic_number < 1])  # fake atoms
                syms = {a.atomic_symbol for a in mol.atoms}
                if not syms <= allowed: reasons["elements"] += 1; continue
                mw = mol.molecular_weight
                if mw is None or mw < args.mw_min or mw > args.mw_max: reasons["mw"] += 1; continue
                n_rot = sum(1 for b in mol.bonds
                            if (not b.is_cyclic)
                            and all(sum(1 for nb in a.neighbours if nb.atomic_number > 1) > 1 for a in b.atoms))
                if n_rot < args.min_rot_bonds: reasons["rotbonds"] += 1; continue

                block = mol.to_string("mol2")
                out.write(block if block.endswith("\n") else block + "\n")
                gcd.write(entry.identifier + "\n")
                kept += 1
            except Exception as e:  # noqa: BLE001
                reasons["error"] += 1
                continue

    print(f"\nDONE scanned {n:,} entries -> kept {kept:,}")
    print("dropped by:", {k: v for k, v in reasons.items() if v})
    print(f"MOL2 -> {args.out}\nrefcodes -> {args.gcd}")
    print("\nnext: build the torsion library from the MOL2 (RDKit side), then re-run the strain "
          "comparison vs PDBBind.")


if __name__ == "__main__":
    main()
