# genbench3d-strain — standalone torsion-strain pose scorer

A fork of [GenBench3D](https://github.com/bbaillif/genbench3d) (Baillif et al. 2024,
[arXiv:2407.04424](https://arxiv.org/abs/2407.04424)) packaged as a small, installable
**pose scorer**. It computes the paper's **s-value** — the geometric mean of per-geometry
*q-values* (a value's KDE likelihood normalised by the density's mode) against a reference
conformer library — with the default configured as a **torsional-strain reward** for
protein–ligand design poses.

Input is a **PDB complex + the ligand SMILES**. The ligand is pulled from the PDB and its bond
orders assigned from the SMILES template (`AssignBondOrdersFromTemplate`), with a halogen/H
over-bond prune for robustness on predicted/minimized poses.

> This fork adds `genbench3d.pose_scorer` (+ the `sscore` CLI), a composable reference layer,
> and the bundled PDBBind/LigBoundConf weights. The upstream benchmark code is untouched and
> stays importable. See `LICENSE` (MIT) and the upstream repo for attribution.

## What it measures

The paper's headline s-value covers **bonds + valence angles** and deliberately *excludes*
torsions (binding-induced torsional strain is legitimate, not a geometry error). This tool's
default is the mirror image — a **torsion s-value** over the freely-varying dihedrals — which
we validated as a strain reward: it separates crystal from strained design poses and, within a
single molecule, tracks an independent torsion-strain metric for 100% of ligands tested.

**Default recipe:** `geometry=torsion`, `torsions=nonring`, `tiers=exact+gen_outer`,
`reference=pdbbind_full`. `s_value ∈ [0,1]`; higher = less strained (more reference-like).

Reference choice (crystal−design gmean(q) separation, bigger = better discriminator):

| reference (`--reference`) | design coverage | separation |
|---|---|---|
| **`pdbbind_full`** (default, 19.4k, bundled) | 92.5% | **+0.222** |
| `ligboundconf` (paper's open ref, bundled) | 86.8% | +0.175 |
| CSD-drug (external, licensed) | 77.6% | +0.121 |

Generalization tiers and the ring/rotatable choice matter: `nonring` + `exact+gen_outer`
(dropping the aggressive inner-neighbour tier and geometrically-stereotyped ring torsions)
is the strongest torsion discriminator we measured.

## Install

```bash
pip install -e .        # editable is recommended — avoids copying ~420 MB of weights into site-packages
# or:  pip install .
```

The large reference KDE pickles are committed **chunked** (`*.part*`, < 50 MB each, under
GitHub's limit) and **reassembled on first use** into the package's `data/weights/` dir
(the reassembled `.p` is git-ignored). No git-lfs needed.

## Usage

Single pose:

```bash
sscore --pdb complex.pdb --smiles "Cc1ccccc1C(=O)N2CCOCC2"
# s_value        0.5834
# n_geometries   23
# geometry       torsion (nonring torsions, tiers=exact+gen_outer)
# reference      pdbbind_full (policy=primary)

sscore --pdb complex.pdb --smiles "..." --json          # machine-readable
sscore --pdb complex.pdb --smiles "..." --per-geometry  # per-torsion q breakdown
sscore --pdb complex.pdb --smiles "..." --resname auto  # ligand resname isn't LIG
```

Batch (CSV with `pdb,smiles[,id]` columns):

```bash
sscore --batch poses.csv --nproc 8 --out scores.csv
```

Library:

```python
from genbench3d.pose_scorer import load_ligand, s_score
mol = load_ligand("complex.pdb", "Cc1ccccc1C(=O)N2CCOCC2")
print(s_score(mol)["s_value"])
# the paper's validity s-value (bonds+angles) instead:
print(s_score(mol, geometry="bond_angle")["s_value"])
```

Knobs: `--geometry {torsion,bond_angle,all}`, `--torsions {nonring,rotatable,all}`,
`--tiers exact,gen_outer[,gen_inner]`, `--no-generalize`.

## Pose validity (Validity3D)

A second CLI, `pose-validity`, computes GenBench3D's **Validity3D**: a pose is 3D-valid when every
bond length and valence angle is geometrically valid (q > 0.001 against the reference), there is no
intramolecular steric clash (vdW × 0.75, PoseBusters-style), and no aromatic/planar ring is puckered
(> 0.1 Å from its mean plane). **Torsions are excluded** from the gate (binding strain is legitimate)
and reported separately as the torsion strain s-value. Bond/angle lookups use **exact patterns only**
(no generalization — the upstream author's fix against underestimating Validity3D). Verified to
reproduce upstream GenBench3D `Validity3D` exactly.

```bash
pose-validity --pdb complex.pdb --smiles "..."       # valid True/False + invalid counts + clashes
pose-validity --sdf ligand.sdf                        # SDF/MOL2 carry bond orders (no SMILES needed)
pose-validity --pdb complex.pdb --smiles "..." --json --per-geometry
```

Batch — a **.txt of newline-separated file paths** → a CSV of results:

```bash
pose-validity --batch paths.txt --nproc 8 --out validity.csv
```

Each line is a path, or `path,smiles` (a PDB needs a SMILES via the 2nd field or a `<stem>.smi`
sidecar; SDF/MOL2 load directly with their own bond orders). The reference weights load **once per
worker** — no reload per pose (and a per-process cache reuses them across library calls too). CSV
columns: `valid, n_invalid_bonds, n_invalid_angles, n_puckered_rings, n_clashes, n_new_patterns,
validity_s_value, min_bond_q, min_angle_q, torsion_strain_s_value, n_heavy_atoms, error`.

```python
from genbench3d.pose_validity import evaluate_validity
from genbench3d.pose_scorer import load_ligand
print(evaluate_validity(load_ligand("complex.pdb", "..."))["valid"])
```

## References — bundled, external, and merged

`--reference` accepts bundled presets (`pdbbind_full`, `ligboundconf`), a **path** to any
reference directory, a registered external alias, or several joined with `+` to **merge**:

```bash
sscore --reference ligboundconf --pdb c.pdb --smiles "..."
sscore --reference /data/refs/csd_drug --pdb c.pdb --smiles "..."      # external, by path
sscore --reference pdbbind_full+/data/refs/csd_drug --pdb c.pdb --smiles "..."  # merged
```

Register external references (e.g. CSD data that can't be committed) in `.env`:

```
GB3D_EXTRA_REFERENCES=csd=/data/refs/csd_drug
```
then `--reference pdbbind_full+csd`. List what's available with `sscore --list-references`.

**Merge policy** (`--policy`, default `primary`): `primary` is a *layered lookup* — the first
reference covering a pattern wins — so a primary reference (PDBBind) keeps its sharp density
and a secondary (CSD) only fills gaps. This extends coverage **without** the dilution that
blending KDEs into one distribution causes. `max_q` / `min_q` take the most permissive /
conservative q among covering references.

## Building your own reference

```bash
# LigBoundConf-style single SDF
python scripts/build_reference.py --sdf S2_LigBoundConf_minimized.sdf \
    --name ligboundconf_ref --out refdata/ligboundconf_ref
# PDBBind-style SDF glob, holding out scored complexes
python scripts/build_reference.py --sdf-glob '/db/PDBBind/**/*_ligand.sdf' \
    --exclude-holo-dir /path/pdbbind/min --name pdbbind_full_ref --out refdata/pdbbind_full_ref
# CSD MOL2 dump (see below)
python scripts/build_reference.py --mol2 CSD_Drug_Subset.mol2 --csd-prep \
    --name csd_drug_ref --out refdata/csd_drug
```

To ship a reference bundled: `python scripts/chunk_weights.py --src <ref_dir> --name <name>
--out genbench3d/data/weights/<preset>` and add the preset to `genbench3d/references.py`.

### CSD (licensed — not committed)

The CSD Drug Subset is CCDC-licensed; its MOL2 dump and any reference **derived** from it are
git-ignored and must be regenerated locally. Copy `.env.example` → `.env`, set `CSDHOME`
(and, if the host isn't already activated, `CCDC_LICENSING_CONFIGURATION`), then run
`scripts/csd_extract_drug_subset.py` in the CCDC `csd-python-api` python to produce the MOL2,
and build the reference with `build_reference.py --mol2 ... --csd-prep`.

## License

MIT (inherited from upstream GenBench3D). Bundled references derive from open data (PDBBind,
LigBoundConf). CSD-derived references are not distributed here.
