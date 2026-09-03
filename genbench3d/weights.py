"""Reassemble chunked reference-KDE pickles shipped in the repo.

GitHub hard-blocks files > 100 MB, so the large ``*_geometry_kernel_densities.p`` pickles are
committed split into < 50 MB parts plus a ``manifest.json`` (whole-file sha256 + byte size +
ordered part list). ``ensure()`` concatenates the parts back into the pickle on first use and
verifies the sha256; the reassembled ``.p`` is git-ignored. External (unchunked) reference
directories that already contain the ``.p`` are handled transparently — ``ensure`` is a no-op.
"""
import hashlib
import json
import os

_BUF = 1 << 20  # 1 MiB streaming buffer


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_BUF), b""):
            h.update(chunk)
    return h.hexdigest()


def reassemble(ref_dir):
    """Reassemble every chunked file described by ``ref_dir/manifest.json``.

    Idempotent: a target already present with the manifest's byte size is left untouched.
    Returns the list of (filename, path) that the manifest covers. No manifest -> [].
    """
    manifest_path = os.path.join(ref_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path) as f:
        manifest = json.load(f)

    out = []
    for fname, meta in manifest.items():
        target = os.path.join(ref_dir, fname)
        if os.path.exists(target) and os.path.getsize(target) == meta["bytes"]:
            out.append((fname, target))
            continue
        parts = [os.path.join(ref_dir, p) for p in meta["parts"]]
        missing = [p for p in parts if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"cannot reassemble {fname}: missing {len(missing)} part(s), "
                f"e.g. {os.path.basename(missing[0])}")
        tmp = target + ".tmp"
        with open(tmp, "wb") as w:
            for p in parts:
                with open(p, "rb") as r:
                    for chunk in iter(lambda: r.read(_BUF), b""):
                        w.write(chunk)
        got = _sha256(tmp)
        if got != meta["sha256"]:
            os.remove(tmp)
            raise ValueError(f"sha256 mismatch reassembling {fname}: {got} != {meta['sha256']}")
        os.replace(tmp, target)
        out.append((fname, target))
    return out


def ensure(ref_dir, ref_name):
    """Return the path to ``<ref_dir>/<ref_name>_geometry_kernel_densities.p``,
    reassembling it from committed chunks if it is not already present."""
    kd = os.path.join(ref_dir, f"{ref_name}_geometry_kernel_densities.p")
    if os.path.exists(kd):
        return kd
    reassemble(ref_dir)
    if not os.path.exists(kd):
        raise FileNotFoundError(
            f"{os.path.basename(kd)} not found in {ref_dir} and no chunks to reassemble it")
    return kd
