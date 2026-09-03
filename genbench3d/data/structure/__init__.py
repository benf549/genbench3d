# bfry: geometry-only EDA. Real Protein/Pocket pull MDAnalysis/openmm/vina and are
# only needed for protein-ligand clash / docking, which we don't run. Provide light
# stubs so type-hint imports (e.g. clash_checker `pocket: Pocket`) resolve.
class Pocket:  # stub — never instantiated in the geometry-only path
    pass
class Protein:  # stub
    pass
VinaProtein = Protein
GlideProtein = Protein
