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
  synonyms (`|`-separated). Seed tags are only the curated collections
  (`epa16`, `astro`, `acene`, `helicene`, `bowl`) and the geometry hint
  `nonplanar`; anything else is rejected.
- Records carry `"<facet>:<value>"` tags that drive the dialog's filters
  (`molvector/catalog.py`, `FACETS`). Computed by the build: `topology`
  (acyclic / unfused / catacondensed / pericondensed / cage), `shape`
  (linear / planar / nonplanar, from the final geometry), `elements`
  (hydrocarbon / heteroatom / inorganic / metal), `rings` (benzenoid /
  nonbenzenoid / hydro / heterocycle / substituted), `charge` (neutral /
  cation / anion / open-shell) and `detected` (confirmed / tentative /
  disputed); `collection:*` comes from the seed.
- `astrochymist.json`: snapshot of The Astrochymist's list of molecules
  detected in the interstellar medium and circumstellar shells
  (<https://www.astrochymist.org/astrochymist_ism.html>, by D. E. Woon): one
  row per detection with the year, the detection paper (link, title,
  reference) and whether it is tentative or disputed. Refresh it with
  `python tools/catalog/fetch_astrochymist.py`; the build never reads the
  live page.
- `astrochymist_smiles.csv`: hand-curated structure for every row of the
  snapshot (keyed by the formula and name as written on the page): SMILES,
  catalog name, ground-state multiplicity where RDKit's radical count is not
  it (carbenes, carbon chains, metal compounds) and a charge for radical
  cations given by a neutral SMILES (C60+, HC3N+). A new row in the snapshot
  without a mapping fails the build. Detections of a molecule that is
  already in `seed.csv` (benzene, C60, the cyano-PAHs, ...) are attached to
  it; conformers (cis/trans formic acid, syn/anti methyl formate) share one
  entry. H3+ has no Lewis structure and is built from a fixed geometry.
- `pubchem_cache.json.gz`: every PubChem PUG REST response the build has used
  (including "not found"). Keeping it committed means a rebuild needs no
  network access and reproduces the catalog byte for byte.

Geometry comes from the PubChem 3D conformer when one exists and its InChIKey
matches the seed. Otherwise the build uses RDKit ETKDG + MMFF94, or, for cages
ETKDG cannot embed (fullerenes), a spectral sphere embedding relaxed with UFF.
Species no force field covers (HeH+, ArH+, VO, CrO, ...) keep the ETKDG
distance-geometry structure. Every entry must pass three checks before the
file is written: an InChI round-trip through
`molvector.render.generate_inchi` (exact for neutral closed-shell molecules,
on the connectivity block for ions and radicals, which Molvector cannot
place on a specific atom), bond lengths within covalent-radius bounds, and
planarity for fully aromatic molecules not tagged `nonplanar`.
