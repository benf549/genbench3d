#!/usr/bin/env python
"""Split a reference KDE pickle into < 50 MB parts + a manifest.json for committing to git,
and copy the small pattern-counts json alongside. This is the inverse of
``genbench3d.weights.reassemble`` (which rebuilds the pickle on first use).

Usage:
    python scripts/chunk_weights.py --src <ref_dir> --name <ref_name> --out <weights_subdir>

e.g.
    python scripts/chunk_weights.py \
        --src /path/refdata/pdbbind_full_ref --name pdbbind_full_ref \
        --out genbench3d/data/weights/pdbbind_full
"""
import argparse
import hashlib
import json
import os
import shutil

_BUF = 1 << 20
_KD_SUFFIX = "_geometry_kernel_densities.p"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_BUF), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir containing <name>_geometry_kernel_densities.p")
    ap.add_argument("--name", required=True, help="reference source name")
    ap.add_argument("--out", required=True, help="target weights subdir (created if needed)")
    ap.add_argument("--part-size", type=int, default=45_000_000, help="max bytes per part")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    kd_name = a.name + _KD_SUFFIX
    kd = os.path.join(a.src, kd_name)
    size = os.path.getsize(kd)
    digest = sha256(kd)

    parts = []
    with open(kd, "rb") as f:
        i = 0
        while True:
            chunk = f.read(a.part_size)
            if not chunk:
                break
            pn = f"{kd_name}.part{i:02d}"
            with open(os.path.join(a.out, pn), "wb") as w:
                w.write(chunk)
            parts.append(pn)
            i += 1

    manifest = {kd_name: {"sha256": digest, "bytes": size, "parts": parts}}
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    pc = f"{a.name}_pattern_counts.json"
    src_pc = os.path.join(a.src, pc)
    if os.path.exists(src_pc):
        shutil.copy(src_pc, os.path.join(a.out, pc))
        print(f"copied {pc}")

    print(f"chunked {kd_name}: {size:,} bytes -> {len(parts)} parts (sha256 {digest[:12]}...)")


if __name__ == "__main__":
    main()
