"""Resolve, load, and merge geometry references for the pose scorer.

A *reference* is a directory holding a GenBench3D KDE pickle
(``<name>_geometry_kernel_densities.p``). It can be:

* a **bundled preset** shipped (chunked) in this package — ``pdbbind_full`` (default,
  best strain discriminator) or ``ligboundconf`` (the paper's open reference);
* an **external directory** you point at by path — e.g. a CSD reference you build locally
  and cannot commit (CSD data is licensed). Register a name for it via the
  ``GB3D_EXTRA_REFERENCES`` / ``GB3D_REFERENCES_FILE`` env vars, or just pass the path.

References **compose**: a spec like ``"pdbbind_full+csd"`` builds a :class:`ReferenceSet`
that merges them. The default ``primary`` policy is a *layered lookup* — the first reference
that covers a geometry pattern wins — so PDBBind's sharp density is kept wherever it has the
pattern and the second reference only fills gaps. This extends coverage without the dilution
that blending the KDEs into one distribution causes.
"""
import glob
import json
import os

from .geometry import ReferenceGeometry
from .data.source import MolListSource
from . import weights

_HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(_HERE, "data", "weights")
MIN_PATTERN_VALUES = 50
_KD_SUFFIX = "_geometry_kernel_densities.p"

# bundled preset alias -> (source name used in the pickle filenames, subdir under WEIGHTS_DIR)
BUNDLED = {
    "pdbbind_full": ("pdbbind_full_ref", "pdbbind_full"),
    "ligboundconf": ("ligboundconf_ref", "ligboundconf"),
}
DEFAULT_REFERENCE = "pdbbind_full"
MERGE_POLICIES = ("primary", "max_q", "min_q")


def _external_registry():
    """Named external references from the environment.

    ``GB3D_REFERENCES_FILE`` -> path to a JSON object ``{alias: directory}``.
    ``GB3D_EXTRA_REFERENCES`` -> ``alias=directory`` entries separated by ``os.pathsep``.
    """
    reg = {}
    f = os.environ.get("GB3D_REFERENCES_FILE")
    if f and os.path.exists(f):
        with open(f) as fh:
            reg.update(json.load(fh))
    extra = os.environ.get("GB3D_EXTRA_REFERENCES")
    if extra:
        for item in extra.split(os.pathsep):
            if "=" in item:
                k, v = item.split("=", 1)
                reg[k.strip()] = os.path.expanduser(v.strip())
    return reg


def _infer_ref_name(ref_dir):
    """Derive the reference source name from a *_geometry_kernel_densities.p file (present or
    listed in the dir's manifest.json)."""
    hits = glob.glob(os.path.join(ref_dir, "*" + _KD_SUFFIX))
    if not hits:
        mani = os.path.join(ref_dir, "manifest.json")
        if os.path.exists(mani):
            with open(mani) as fh:
                for fname in json.load(fh):
                    if fname.endswith(_KD_SUFFIX):
                        hits = [os.path.join(ref_dir, fname)]
                        break
    if not hits:
        raise FileNotFoundError(
            f"no *{_KD_SUFFIX} (and no manifest listing one) in reference dir {ref_dir}")
    return os.path.basename(hits[0])[: -len(_KD_SUFFIX)]


def available_references():
    """List selectable reference aliases (bundled presets + registered external)."""
    return {"bundled": list(BUNDLED), "external": list(_external_registry())}


def load_reference(spec, use_generalized_patterns=True):
    """Resolve a single reference token to ``(label, ReferenceGeometry)``.

    ``spec`` is a bundled preset alias, a registered external alias, a path to a reference
    directory, or a path to a ``*_geometry_kernel_densities.p`` file.
    """
    reg = _external_registry()
    if spec in BUNDLED:
        name, sub = BUNDLED[spec]
        ref_dir, label = os.path.join(WEIGHTS_DIR, sub), spec
    elif spec in reg:
        ref_dir = os.path.expanduser(reg[spec])
        name, label = _infer_ref_name(ref_dir), spec
    elif os.path.isdir(spec):
        ref_dir = spec
        name = _infer_ref_name(ref_dir)
        label = os.path.basename(os.path.normpath(spec))
    elif os.path.isfile(spec) and spec.endswith(_KD_SUFFIX):
        ref_dir = os.path.dirname(spec) or "."
        name = os.path.basename(spec)[: -len(_KD_SUFFIX)]
        label = name
    else:
        raise ValueError(
            f"unknown reference {spec!r}. Bundled presets: {list(BUNDLED)}; "
            f"registered external: {list(reg)}; or pass a directory / .p path.")

    weights.ensure(ref_dir, name)  # reassemble committed chunks on first use; no-op otherwise
    src = MolListSource(mol_list=[], name=name)
    ref = ReferenceGeometry(source=src, root=ref_dir,
                            minimum_pattern_values=MIN_PATTERN_VALUES,
                            use_generalized_patterns=use_generalized_patterns)
    return label, ref


class ReferenceSet:
    """One or more loaded references, queried per geometry under a merge policy.

    * ``primary`` (default): the first reference (in spec order) that covers the pattern under
      the allowed tiers wins. Keeps each reference's own KDE; use to let a secondary reference
      (e.g. CSD) fill only the patterns the primary (e.g. PDBBind) lacks.
    * ``max_q``: most permissive q among covering references.
    * ``min_q``: most conservative q among covering references.
    """

    def __init__(self, members, policy="primary"):
        if policy not in MERGE_POLICIES:
            raise ValueError(f"policy must be one of {MERGE_POLICIES}, got {policy!r}")
        self.members = members  # list[(label, ReferenceGeometry)]
        self.policy = policy

    @property
    def labels(self):
        return [lbl for lbl, _ in self.members]

    @staticmethod
    def _tier(pattern, kds, allow_inner):
        if pattern in kds:
            return "exact"
        if pattern.generalize() in kds:
            return "gen_outer"
        if allow_inner and pattern.generalize(inner_neighbors=True) in kds:
            return "gen_inner"
        return "new"

    def geometry_q(self, pattern, value, geometry, tiers):
        """Return ``(q, label, tier)`` for one geometry, or ``(nan, None, 'new')`` if no
        reference covers it under the allowed generalization ``tiers``."""
        # bonds only ever generalize at the outer level upstream; torsions/angles also try
        # the (aggressive) inner-neighbour level, which we gate behind an explicit tier.
        allow_inner = (geometry != "bond") and ("gen_inner" in tiers)
        found = []
        for label, ref in self.members:
            kds = ref.kernel_densities[geometry]
            tier = self._tier(pattern, kds, allow_inner)
            if tier == "new" or tier not in tiers:
                continue
            q, _ = ref.geometry_is_valid(pattern, value, geometry=geometry)
            if q != q:  # nan
                continue
            if self.policy == "primary":
                return q, label, tier
            found.append((q, label, tier))
        if not found:
            return float("nan"), None, "new"
        return (min if self.policy == "min_q" else max)(found, key=lambda x: x[0])


def resolve(spec, policy="primary", use_generalized_patterns=True):
    """Build a :class:`ReferenceSet` from a spec string. ``'+'`` merges references,
    e.g. ``'pdbbind_full+csd'`` or ``'pdbbind_full+/data/refs/csd_drug'``."""
    tokens = [t.strip() for t in str(spec).split("+") if t.strip()]
    if not tokens:
        raise ValueError("empty reference spec")
    members = [load_reference(t, use_generalized_patterns) for t in tokens]
    return ReferenceSet(members, policy=policy)
