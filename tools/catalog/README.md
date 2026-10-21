# Molecule catalog build

Builds `molvector/catalog/catalog.json.gz`, the curated molecule set behind
**File → Open from Catalog…**. Not shipped with the package.

```bash
pip install rdkit
python tools/catalog/build_catalog.py            # rebuild from seed.csv + PubChem cache
python tools/catalog/build_catalog.py --offline  # fail rather than query PubChem
python tools/catalog/build_catalog.py --refresh  # re-download everything from PubChem
```

- `seed.csv`: hand-maintained list: name, SMILES, tags (`;`-separated) and
  synonyms (`|`-separated). Structural tags (benzenoid, catacondensed,
  pericondensed, planar, hydrocarbon) are derived automatically.
- `pubchem_cache.json.gz`: every PubChem PUG REST response the build has used
  (including "not found"). Keeping it committed means a rebuild needs no
  network access and reproduces the catalog byte for byte.

Geometry comes from the PubChem 3D conformer when one exists and its InChIKey
matches the seed. Otherwise the build uses RDKit ETKDG + MMFF94, or, for cages
ETKDG cannot embed (fullerenes), a spectral sphere embedding relaxed with UFF.
Every entry must pass three checks before the file is written: an InChI
round-trip through `molvector.render.generate_inchi`, bond-length sanity, and
planarity for fully aromatic molecules not tagged `nonplanar`.
