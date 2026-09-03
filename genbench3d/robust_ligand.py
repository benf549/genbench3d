"""Halogen-safe ligand building for the standalone pose scorer.

RDKit infers bonds from interatomic distance when a PDB has no explicit CONECT records.
In a predicted or minimized pose a halogen (or hydrogen) can drift close enough to another
atom to pick up a second, non-physical bond (e.g. a divalent Cl). That breaks sanitization
and the SMILES-template bond-order match, which otherwise silently NaNs the scorer for
halogen-bearing ligands. `prune_terminal_overbonds` removes those artifacts and is a no-op
on well-formed molecules, so it is always safe to apply.

Vendored verbatim from the dock-pilot analysis lib (lib/robust_ligand.py).
"""
from rdkit import Chem

# H and the halogens each form exactly one bond; any extra bond is a proximity artifact.
_MAX_BONDS_BY_ELEMENT = {"H": 1, "F": 1, "Cl": 1, "Br": 1, "I": 1}


def prune_terminal_overbonds(mol):
    """Return `mol` with spurious bonds removed so every H/halogen keeps only its
    single closest neighbour. A no-op for well-formed molecules, so it is always safe."""
    if mol is None or mol.GetNumConformers() == 0:
        return mol

    coords = mol.GetConformer()

    def bond_length(bond):
        start = coords.GetAtomPosition(bond.GetBeginAtomIdx())
        end = coords.GetAtomPosition(bond.GetEndAtomIdx())
        return start.Distance(end)

    editable = Chem.RWMol(mol)
    removed_any = False
    for atom in editable.GetAtoms():
        max_bonds = _MAX_BONDS_BY_ELEMENT.get(atom.GetSymbol())
        if max_bonds is None:
            continue
        bonds_closest_first = sorted(atom.GetBonds(), key=bond_length)
        for spurious in bonds_closest_first[max_bonds:]:      # keep the closest, drop the rest
            editable.RemoveBond(spurious.GetBeginAtomIdx(), spurious.GetEndAtomIdx())
            removed_any = True

    return editable.GetMol() if removed_any else mol


def robust_mol_from_pdb_block(block, *args, **kwargs):
    """Drop-in replacement for `Chem.MolFromPDBBlock` that survives halogen artifacts.

    The normal, sanitizing call returns None when a spurious halogen bond makes an atom
    over-valent. In that case we reparse without sanitizing, prune the bad bonds, and try
    to sanitize the cleaned molecule.
    """
    mol = Chem.MolFromPDBBlock(block, *args, **kwargs)
    if mol is not None:
        return mol

    mol = Chem.MolFromPDBBlock(block, sanitize=False, proximityBonding=True, removeHs=False)
    if mol is None:
        return None
    mol = prune_terminal_overbonds(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass          # leave it unsanitized; the caller's template match will validate it
    return mol
