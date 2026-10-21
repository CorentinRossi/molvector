#!/usr/bin/env python3
"""
Build Molvector's bundled molecule catalog (molvector/catalog/catalog.json.gz).

Offline tool, not shipped with the package. Requires RDKit (and network
access to PubChem the first time a molecule is seen).

    python tools/catalog/build_catalog.py              # rebuild from seed + cache
    python tools/catalog/build_catalog.py --offline    # never touch the network
    python tools/catalog/build_catalog.py --refresh    # re-download from PubChem

For each row of seed.csv:
  1. RDKit sanitize -> canonical SMILES, InChIKey (dedupe key), formula,
     ring count, charge, spin multiplicity, facet tags ("<facet>:<value>").
  2. Geometry: PubChem's 3D conformer (looked up by InChIKey, then by name,
     accepted only if its InChIKey matches); otherwise RDKit ETKDG + MMFF94;
     cages that ETKDG cannot embed (fullerenes) start from a spectral
     sphere embedding of the bond graph and are relaxed with UFF.
  3. Validation: InChI round-trip through molvector.render.generate_inchi,
     bond-length sanity, planarity of flat aromatic molecules.

PubChem responses are cached in pubchem_cache.json.gz next to this script,
so a rebuild reproduces the same file byte for byte.
"""

import argparse
import csv
import gzip
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from rdkit import Chem, RDLogger                      # noqa: E402
from rdkit.Chem import AllChem, rdMolDescriptors      # noqa: E402
from rdkit.Geometry import Point3D                    # noqa: E402

from molvector.render import generate_inchi           # noqa: E402
from molvector.catalog import to_molecule             # noqa: E402

RDLogger.DisableLog("rdApp.*")

SEED_PATH = os.path.join(HERE, "seed.csv")
CACHE_PATH = os.path.join(HERE, "pubchem_cache.json.gz")
OUT_PATH = os.path.join(REPO, "molvector", "catalog", "catalog.json.gz")

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CATALOG_VERSION = 1
EMBED_SEED = 20240611
COORD_DECIMALS = 4

# Allowed bond lengths (Angstrom) by element pair; others use covalent radii.
BOND_RANGES = {
    ("C", "C"): (1.17, 1.60),
    ("C", "H"): (1.04, 1.14),
    ("C", "N"): (1.12, 1.50),
    ("H", "N"): (0.98, 1.06),
}
COVALENT_RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05}
PLANAR_TOL = 0.10        # max heavy-atom out-of-plane deviation (Angstrom) for "planar"


# ── PubChem ──────────────────────────────────────────────────────────────────

class PubChem:
    """PUG REST client with a persistent on-disk cache keyed by request."""

    def __init__(self, path: str, offline: bool, refresh: bool):
        self.path = path
        self.offline = offline
        self.refresh = refresh
        self.cache = {}
        if os.path.exists(path) and not refresh:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                self.cache = json.load(f)
        self.dirty = False

    def _get(self, url_path: str):
        """Response body for *url_path*, or None for a 404 (both cached)."""
        if url_path in self.cache:
            return self.cache[url_path]
        if self.offline:
            raise RuntimeError(f"--offline and not cached: {url_path}")
        body = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(f"{PUG}/{url_path}", timeout=60) as r:
                    body = r.read().decode("utf-8")
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break
                if e.code not in (500, 503, 504) or attempt == 3:
                    raise
            except urllib.error.URLError:
                if attempt == 3:
                    raise
            time.sleep(2.0 * (attempt + 1))
        time.sleep(0.25)                    # PubChem asks for <= 5 requests/s
        self.cache[url_path] = body
        self.dirty = True
        return body

    def cid(self, kind: str, value: str):
        body = self._get(f"compound/{kind}/{urllib.parse.quote(value, safe='')}/cids/TXT")
        if not body:
            return None
        first = body.split()[0]
        return int(first) if first.isdigit() and first != "0" else None

    def sdf_3d(self, cid: int):
        return self._get(f"compound/cid/{cid}/SDF?record_type=3d")

    def save(self):
        if not self.dirty:
            return
        data = json.dumps(self.cache, sort_keys=True, indent=0).encode("utf-8")
        _write_gzip(self.path, data)


def _write_gzip(path: str, data: bytes):
    """gzip with a fixed header (no mtime / filename) so output is reproducible."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as gz:
        gz.write(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(buf.getvalue())


# ── geometry ─────────────────────────────────────────────────────────────────

def pubchem_geometry(pc: PubChem, name: str, inchikey: str):
    """(mol with 3D conformer, cid) from PubChem if it has a matching 3D record."""
    cids = []
    for kind, value in (("inchikey", inchikey), ("name", name)):
        cid = pc.cid(kind, value)
        if cid and cid not in cids:
            cids.append(cid)
    for cid in cids:
        sdf = pc.sdf_3d(cid)
        if not sdf:
            continue
        mol = Chem.MolFromMolBlock(sdf, removeHs=False)
        if mol is not None and Chem.MolToInchiKey(mol) == inchikey:
            return mol, cid
    return None, (cids[0] if cids else None)


def _spectral_embed(mol) -> bool:
    """Seed a cage on a sphere from the bond graph's Laplacian eigenvectors."""
    n = mol.GetNumAtoms()
    adj = Chem.GetAdjacencyMatrix(mol).astype(float)
    _, vecs = np.linalg.eigh(np.diag(adj.sum(1)) - adj)
    xyz = vecs[:, 1:4]
    norms = np.linalg.norm(xyz, axis=1)
    if np.any(norms < 1e-6):
        return False
    xyz /= norms[:, None]
    mean_bond = np.mean([np.linalg.norm(xyz[b.GetBeginAtomIdx()] - xyz[b.GetEndAtomIdx()])
                         for b in mol.GetBonds()])
    xyz *= 1.42 / mean_bond
    conf = Chem.Conformer(n)
    for i in range(n):
        conf.SetAtomPosition(i, Point3D(*xyz[i]))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    return True


def _bend(mol, depth_per_a2: float = 0.03):
    """Curve a flat conformer into a shallow bowl (z grows with in-plane r^2), so
    the force field can leave the flat saddle point where ETKDG places bowls
    such as corannulene."""
    conf = mol.GetConformer()
    xyz = orient(conf.GetPositions())
    for i, (x, y, z) in enumerate(xyz):
        conf.SetAtomPosition(i, Point3D(x, y, z + depth_per_a2 * (x * x + y * y)))


def rdkit_geometry(mol, curved: bool = False):
    """(mol with Hs and a 3D conformer, source label). With *curved*, the
    embedded structure is bent out of plane before force-field relaxation."""
    mh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = EMBED_SEED
    params.maxIterations = 200
    if AllChem.EmbedMolecule(mh, params) == 0:
        if curved:
            # MMFF94(s) flattens bowls like corannulene back out; UFF keeps
            # the curvature (0.97 A bowl depth vs. 0.87 A experimental).
            _bend(mh)
            AllChem.UFFOptimizeMolecule(mh, maxIters=20000)
            return mh, "rdkit-etkdg-uff"
        if AllChem.MMFFHasAllMoleculeParams(mh):
            AllChem.MMFFOptimizeMolecule(mh, maxIters=20000)
            return mh, "rdkit-etkdg-mmff94"
        AllChem.UFFOptimizeMolecule(mh, maxIters=20000)
        return mh, "rdkit-etkdg-uff"
    # ETKDG gives up on fullerene cages. All-carbon cages relax cleanly with
    # UFF from a spectral sphere (MMFF distorts them: aromatic typing).
    if mh.GetNumAtoms() == mol.GetNumAtoms() and _spectral_embed(mh):
        AllChem.UFFOptimizeMolecule(mh, maxIters=20000)
        return mh, "rdkit-spectral-uff"
    raise RuntimeError("RDKit could not embed the molecule")


def orient(xyz: np.ndarray) -> np.ndarray:
    """Center on the centroid and rotate onto principal axes (normal of a flat
    molecule along z), with deterministic, right-handed axis signs."""
    c = xyz - xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    axes = []
    for k in range(2):
        a = vt[k]
        proj = c @ a
        if abs(proj.min()) > abs(proj.max()) + 1e-6:  # farthest atom on the + side
            a = -a
        axes.append(a)
    axes.append(np.cross(axes[0], axes[1]))
    out = c @ np.array(axes).T
    out[np.abs(out) < 10 ** -COORD_DECIMALS] = 0.0
    return out


# ── structural classification ────────────────────────────────────────────────

# Seed-file tags: curated collections (exported as "collection:<tag>") and
# "nonplanar", a geometry hint (bowls, helicenes, twisted biphenyls) that is
# checked against, not exported: the shape facet is computed from geometry.
COLLECTION_TAGS = {"epa16", "astro", "acene", "helicene", "bowl"}
SEED_HINTS = {"nonplanar"}


def facet_tags(mol, planar: bool) -> list:
    """Computed "<facet>:<value>" tags. Every record gets exactly one topology,
    shape and charge value, and at least one composition value."""
    rings = [set(r) for r in mol.GetRingInfo().AtomRings() if len(r) <= 8]
    per_atom = {}
    for r in rings:
        for a in r:
            per_atom[a] = per_atom.get(a, 0) + 1
    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    elements = {a.GetSymbol() for a in heavy}
    hydrocarbon = elements == {"C"}

    # Topology of the ring system (rings of <= 8 atoms: kekulene's cavity is
    # not a ring). Cage: an all-carbon, hydrogen-free closed shell, every atom
    # shared by three rings (fullerenes); cata/peri applies to the other fused
    # systems: pericondensed when some atom is shared by three rings.
    if hydrocarbon and heavy and sum(a.GetTotalNumHs() for a in heavy) == 0 \
            and all(per_atom.get(a.GetIdx(), 0) >= 3 for a in heavy):
        topology = "cage"
    elif not any(len(r1 & r2) >= 2 for i, r1 in enumerate(rings) for r2 in rings[i + 1:]):
        topology = "unfused"
    else:
        topology = "pericondensed" if max(per_atom.values()) >= 3 else "catacondensed"

    # Composition (several may apply).
    in_ring = set(per_atom)
    ring_heavy = [a for a in heavy if a.GetIdx() in in_ring]
    composition = set()
    if any(a.GetIdx() not in in_ring for a in heavy):
        composition.add("substituted")
    if any(a.GetSymbol() != "C" for a in ring_heavy):
        composition.add("heteroatom")
    # A saturated ring CH2 (not a radical CH, which stays conjugated).
    if any(a.GetSymbol() == "C" and a.GetTotalNumHs() >= 2 for a in ring_heavy):
        composition.add("hydro")
    # Plain hydrocarbon frameworks: benzenoid when every ring is a hexagon.
    if hydrocarbon and not composition:
        composition.add("benzenoid" if all(len(r) == 6 for r in rings) else "nonbenzenoid")

    charge = Chem.GetFormalCharge(mol)
    radical = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) > 0
    if radical:
        electronic = "radical"
    elif charge > 0:
        electronic = "cation"
    elif charge < 0:
        electronic = "anion"
    else:
        electronic = "closed-shell"

    return ([f"topology:{topology}", f"shape:{'planar' if planar else 'nonplanar'}",
             f"charge:{electronic}"] + [f"composition:{c}" for c in sorted(composition)])


def expect_planar(mol, seed_tags) -> bool:
    if "nonplanar" in seed_tags:
        return False
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 1:
            continue
        if not (a.GetIsAromatic() or a.GetHybridization() == Chem.HybridizationType.SP):
            return False
    return True


# ── validation ───────────────────────────────────────────────────────────────

def check_bonds(record) -> list:
    problems = []
    atoms = record["atoms"]
    for i, j, _ in record["bonds"]:
        ei, ej = atoms[i][0], atoms[j][0]
        d = float(np.linalg.norm(np.subtract(atoms[i][1:], atoms[j][1:])))
        key = tuple(sorted((ei, ej)))
        lo, hi = BOND_RANGES.get(key, (None, None))
        if lo is None:
            ref = COVALENT_RADII.get(ei, 0.8) + COVALENT_RADII.get(ej, 0.8)
            lo, hi = 0.8 * ref, 1.2 * ref
        if not lo <= d <= hi:
            problems.append(f"{ei}{i}-{ej}{j} = {d:.3f} A outside [{lo}, {hi}]")
    return problems


def inchi_roundtrip(record) -> str:
    """InChIKey of the stored geometry/bonds as Molvector itself perceives it."""
    inchi = generate_inchi(to_molecule(record))
    return Chem.InchiToInchiKey(inchi) if inchi else ""


# ── main ─────────────────────────────────────────────────────────────────────

def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def read_seed(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in rows:
            yield {
                "name": row["name"].strip(),
                "smiles": row["smiles"].strip(),
                "tags": [t.strip() for t in row["tags"].split(";") if t.strip()],
                "synonyms": [s.strip() for s in (row.get("synonyms") or "").split("|") if s.strip()],
            }


def build_record(seed, pc: PubChem):
    """(record, problems) from the first geometry source whose record validates:
    PubChem's 3D conformer, then RDKit. *problems* is empty on success."""
    mol = Chem.MolFromSmiles(seed["smiles"])
    if mol is None:
        raise ValueError(f"invalid SMILES {seed['smiles']}")
    inchikey = Chem.MolToInchiKey(mol)

    geo, cid = pubchem_geometry(pc, seed["name"], inchikey)
    problems = []
    if geo is not None:
        record, problems = make_record(seed, mol, inchikey, cid, geo, f"pubchem-3d:{cid}")
        if not problems:
            return record, []
    geo, source = rdkit_geometry(mol, curved="nonplanar" in seed["tags"])
    record, rd_problems = make_record(seed, mol, inchikey, cid, geo, source)
    if rd_problems and problems:
        rd_problems = [f"PubChem: {p}" for p in problems] + [f"RDKit: {p}" for p in rd_problems]
    return record, rd_problems


def make_record(seed, mol, inchikey, cid, geo, source):
    geo = Chem.Mol(geo)
    Chem.Kekulize(geo, clearAromaticFlags=True)

    # Molvector keeps a molecule's charge on its first heavy atom (see
    # render.generate_inchi), so put the charged / radical atom first.
    first = next((a.GetIdx() for a in geo.GetAtoms()
                  if a.GetFormalCharge() or a.GetNumRadicalElectrons()), 0)
    order = [first] + [i for i in range(geo.GetNumAtoms()) if i != first]
    geo = Chem.RenumberAtoms(geo, order)

    xyz = orient(geo.GetConformer().GetPositions())
    bond_order = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}
    unknown = set(seed["tags"]) - COLLECTION_TAGS - SEED_HINTS
    if unknown:
        raise ValueError(f"unknown seed tag(s) {sorted(unknown)}")
    heavy = [a.GetIdx() for a in geo.GetAtoms() if a.GetAtomicNum() > 1]
    out_of_plane = float(np.max(np.abs(xyz[heavy, 2]))) if heavy else 0.0
    planar = len(heavy) >= 3 and out_of_plane <= PLANAR_TOL
    tags = facet_tags(mol, planar) + [f"collection:{t}" for t in seed["tags"]
                                      if t in COLLECTION_TAGS]

    record = {
        "id": slug(seed["name"]),
        "name": seed["name"],
        "synonyms": seed["synonyms"],
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "smiles": Chem.MolToSmiles(mol),
        "inchikey": inchikey,
        "pubchem_cid": cid,
        "charge": Chem.GetFormalCharge(mol),
        "multiplicity": sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) + 1,
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "tags": sorted(set(tags)),
        "geometry_source": source,
        "atoms": [[a.GetSymbol()] + [round(float(v), COORD_DECIMALS) + 0.0 for v in xyz[a.GetIdx()]]
                  for a in geo.GetAtoms()],
        "bonds": [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_order[b.GetBondType()]]
                  for b in geo.GetBonds()],
    }

    problems = check_bonds(record)
    if expect_planar(mol, seed["tags"]) and not planar:
        problems.append(f"expected planar, heavy-atom max |z| = {out_of_plane:.3f} A")
    if "nonplanar" in seed["tags"] and planar:
        problems.append("tagged nonplanar but the geometry is flat")
    rt = inchi_roundtrip(record)
    if rt != inchikey:
        problems.append(f"InChI round-trip gave {rt or 'nothing'}, expected {inchikey}")
    return record, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", default=SEED_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--cache", default=CACHE_PATH)
    ap.add_argument("--offline", action="store_true", help="fail instead of querying PubChem")
    ap.add_argument("--refresh", action="store_true", help="ignore the PubChem cache")
    args = ap.parse_args(argv)

    pc = PubChem(args.cache, offline=args.offline, refresh=args.refresh)
    records, seen_keys, seen_ids, failures = [], {}, set(), 0
    try:
        for seed in read_seed(args.seed):
            try:
                rec, problems = build_record(seed, pc)
            except Exception as e:
                print(f"FAIL  {seed['name']}: {e}")
                failures += 1
                continue
            if rec["inchikey"] in seen_keys:
                print(f"FAIL  {seed['name']}: duplicate of {seen_keys[rec['inchikey']]}")
                failures += 1
                continue
            if rec["id"] in seen_ids:
                print(f"FAIL  {seed['name']}: duplicate id {rec['id']}")
                failures += 1
                continue
            if problems:
                print(f"FAIL  {seed['name']}: " + "; ".join(problems))
                failures += 1
                continue
            seen_keys[rec["inchikey"]] = rec["name"]
            seen_ids.add(rec["id"])
            records.append(rec)
            print(f"ok    {rec['name']:<28} {rec['formula']:<10} {len(rec['atoms']):>4} atoms  "
                  f"{rec['geometry_source']}")
    finally:
        pc.save()

    if failures:
        print(f"\n{failures} seed entr{'y' if failures == 1 else 'ies'} failed; catalog not written.")
        return 1

    payload = {"version": CATALOG_VERSION, "molecules": records}
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    _write_gzip(args.out, data)
    print(f"\nWrote {len(records)} molecules to {os.path.relpath(args.out, REPO)} "
          f"({os.path.getsize(args.out) / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
