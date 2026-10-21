#!/usr/bin/env python3
"""
Build Molvector's bundled molecule catalog (molvector/catalog/catalog.json.gz).

Offline tool, not shipped with the package. Requires RDKit (and network
access to PubChem the first time a molecule is seen).

    python tools/catalog/build_catalog.py              # rebuild from seed + cache
    python tools/catalog/build_catalog.py --offline    # never touch the network
    python tools/catalog/build_catalog.py --refresh    # re-download from PubChem

Entries come from seed.csv (curated PAHs and related molecules) and from
every species detected in space as listed by The Astrochymist
(astrochymist.json, mapped to structures in astrochymist_smiles.csv);
detections of a seed molecule are attached to it. For each entry:
  1. RDKit sanitize -> canonical SMILES, InChIKey (dedupe key, with the
     charge), formula, ring count, charge, spin multiplicity, facet tags
     ("<facet>:<value>") and, for detected species, the detection papers.
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
ASTRO_PATH = os.path.join(HERE, "astrochymist.json")
ASTRO_MAP_PATH = os.path.join(HERE, "astrochymist_smiles.csv")
CACHE_PATH = os.path.join(HERE, "pubchem_cache.json.gz")
OUT_PATH = os.path.join(REPO, "molvector", "catalog", "catalog.json.gz")

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CATALOG_VERSION = 2
EMBED_SEED = 20240611
COORD_DECIMALS = 4

# Allowed bond lengths (Angstrom) by element pair; others use covalent radii.
BOND_RANGES = {
    ("C", "C"): (1.17, 1.60),
    ("C", "H"): (1.04, 1.14),
    ("C", "N"): (1.12, 1.60),          # 2H-azirine's strained C-N: 1.52-1.57
    ("H", "N"): (0.98, 1.06),
}
# Cordero et al., Dalton Trans. 2008 (single-bond covalent radii, Angstrom).
COVALENT_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02,
    "Ar": 1.06, "K": 2.03, "Ca": 1.76, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Fe": 1.32,
}
METALS = {"Li", "Na", "K", "Mg", "Ca", "Al", "Ti", "V", "Cr", "Fe"}
LINEAR_TOL = 0.05        # max distance (Angstrom) from the principal axis for "linear"
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
    embedded = AllChem.EmbedMolecule(mh, params) == 0
    if not embedded and mh.GetNumAtoms() != mol.GetNumAtoms():
        # Small strained rings (T-shaped MgC2, c-SiC2, ...): random start.
        params.useRandomCoords = True
        embedded = AllChem.EmbedMolecule(mh, params) == 0
    if embedded:
        if curved:
            # MMFF94(s) flattens bowls like corannulene back out; UFF keeps
            # the curvature (0.97 A bowl depth vs. 0.87 A experimental).
            _bend(mh)
            AllChem.UFFOptimizeMolecule(mh, maxIters=20000)
            return mh, "rdkit-etkdg-uff"
        if AllChem.MMFFHasAllMoleculeParams(mh):
            AllChem.MMFFOptimizeMolecule(mh, maxIters=20000)
            return mh, "rdkit-etkdg-mmff94"
        if AllChem.UFFHasAllMoleculeParams(mh):
            AllChem.UFFOptimizeMolecule(mh, maxIters=20000)
            return mh, "rdkit-etkdg-uff"
        # No force field covers it (HeH+, ArH+, VO, CrO, ...): the ETKDG
        # distance-geometry structure, checked against covalent radii below.
        return mh, "rdkit-etkdg"
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
DETECTION_RANK = {"confirmed": 0, "tentative": 1, "disputed": 2}


def shape_of(xyz: np.ndarray, heavy: list) -> str:
    """"linear", "planar" or "nonplanar" for oriented coordinates (principal
    axes: x longest, z the plane normal)."""
    if len(xyz) <= 2 or float(np.max(np.hypot(xyz[:, 1], xyz[:, 2]))) <= LINEAR_TOL:
        return "linear"
    pts = heavy if len(heavy) >= 3 else list(range(len(xyz)))
    return "planar" if float(np.max(np.abs(xyz[pts, 2]))) <= PLANAR_TOL else "nonplanar"


def facet_tags(mol, shape: str, charge: int, multiplicity: int) -> list:
    """Computed "<facet>:<value>" tags. Every record gets exactly one
    topology and shape value and at least one elements and charge value;
    ring-system tags only apply to molecules with rings."""
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
    if not rings:
        topology = "acyclic"
    elif hydrocarbon and sum(a.GetTotalNumHs() for a in heavy) == 0 \
            and all(per_atom.get(a.GetIdx(), 0) >= 3 for a in heavy):
        topology = "cage"
    elif not any(len(r1 & r2) >= 2 for i, r1 in enumerate(rings) for r2 in rings[i + 1:]):
        topology = "unfused"
    else:
        topology = "pericondensed" if max(per_atom.values()) >= 3 else "catacondensed"

    # Elements (several may apply).
    elem = set()
    if "C" in elements:
        elem.add("hydrocarbon" if elements <= {"C"} else "heteroatom")
    else:
        elem.add("inorganic")
    if elements & METALS:
        elem.add("metal")

    # Ring system (several may apply; none for acyclic molecules).
    ring_sys = set()
    if rings:
        in_ring = set(per_atom)
        ring_heavy = [a for a in heavy if a.GetIdx() in in_ring]
        if any(a.GetIdx() not in in_ring for a in heavy):
            ring_sys.add("substituted")
        if any(a.GetSymbol() != "C" for a in ring_heavy):
            ring_sys.add("heterocycle")
        # A saturated ring CH2 (not a radical CH, which stays conjugated).
        if any(a.GetSymbol() == "C" and a.GetTotalNumHs() >= 2 for a in ring_heavy):
            ring_sys.add("hydro")
        # Plain hydrocarbon frameworks: benzenoid when every ring is a hexagon.
        if hydrocarbon and not ring_sys:
            ring_sys.add("benzenoid" if all(len(r) == 6 for r in rings) else "nonbenzenoid")

    electronic = {"cation" if charge > 0 else "anion" if charge < 0 else "neutral"}
    if multiplicity > 1:
        electronic.add("open-shell")

    return ([f"topology:{topology}", f"shape:{shape}"]
            + [f"elements:{v}" for v in sorted(elem)]
            + [f"rings:{v}" for v in sorted(ring_sys)]
            + [f"charge:{v}" for v in sorted(electronic)])


def expect_planar(mol, seed_tags) -> bool:
    """Fully aromatic / sp frameworks of >= 3 heavy atoms must come out flat."""
    if "nonplanar" in seed_tags:
        return False
    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if len(heavy) < 3:
        return False
    return all(a.GetIsAromatic() or a.GetHybridization() == Chem.HybridizationType.SP
               for a in heavy)


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
            # Multiple bonds run ~25 % shorter, ionic metal bonds longer.
            ref = COVALENT_RADII.get(ei, 1.0) + COVALENT_RADII.get(ej, 1.0)
            lo, hi = round(0.70 * ref, 3), round(1.30 * ref, 3)
        if not lo <= d <= hi:
            problems.append(f"{ei}{i}-{ej}{j} = {d:.3f} A outside [{lo}, {hi}]")
    return problems


def inchi_roundtrip(record, neutral: bool = False) -> str:
    """InChIKey of the stored geometry/bonds as Molvector itself perceives it
    (with *neutral*, ignoring the record's net charge)."""
    mol = to_molecule(record)
    if neutral:
        mol.charge = 0
    inchi = generate_inchi(mol)
    return Chem.InchiToInchiKey(inchi) if inchi else ""


# ── entries ──────────────────────────────────────────────────────────────────

def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _csv_rows(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(line for line in f if not line.startswith("#"))


def read_seed(path: str):
    for row in _csv_rows(path):
        yield {
            "name": row["name"].strip(),
            "smiles": row["smiles"].strip(),
            "tags": [t.strip() for t in row["tags"].split(";") if t.strip()],
            "synonyms": [s.strip() for s in (row.get("synonyms") or "").split("|") if s.strip()],
            "charge": None, "multiplicity": None, "notes": [], "detections": [],
        }


def read_astro(snapshot_path: str, map_path: str):
    """One entry per Astrochymist detection row, with its structure mapping."""
    with open(snapshot_path, encoding="utf-8") as f:
        snapshot = json.load(f)
    mapping = {(r["formula"], r["name"]): r for r in _csv_rows(map_path)}
    for det in snapshot["detections"]:
        m = mapping.get((det["formula"], det["name"]))
        if m is None:
            raise ValueError(f"{det['formula']} ({det['name'] or 'no name'}) is not mapped in "
                             f"{os.path.basename(map_path)}")
        status = "disputed" if det["disputed"] else "tentative" if det["tentative"] else "confirmed"
        # The page's common name, minus editorial asides in parentheses.
        page_name = re.sub(r"\s*\(.*?\)", "", det["name"]).strip()
        synonyms = [det["formula"]] + ([page_name] if page_name else [])
        yield {
            "name": m["catalog_name"].strip(),
            "smiles": m["smiles"].strip(),
            "tags": [],
            "synonyms": synonyms,
            "charge": int(m["charge"]) if m["charge"] else None,
            "multiplicity": int(m["multiplicity"]) if m["multiplicity"] else None,
            "notes": [m["note"]] if m["note"] else [],
            "detections": [{
                "year": det["year"], "label": det["formula"], "status": status,
                "paper_url": det["paper_url"], "paper_title": det["paper_title"],
                "paper_ref": det["paper_ref"],
            }],
        }


def entry_key(entry):
    """(InChIKey, charge): the dedupe key; radical cations given as a neutral
    SMILES plus a charge (C60+) stay apart from their neutral parent."""
    if not entry["smiles"]:
        return ("special:" + entry["name"], entry["charge"] or 0)
    mol = Chem.MolFromSmiles(entry["smiles"])
    if mol is None:
        raise ValueError(f"invalid SMILES {entry['smiles']} ({entry['name']})")
    charge = entry["charge"] if entry["charge"] is not None else Chem.GetFormalCharge(mol)
    return (Chem.MolToInchiKey(mol), charge)


def merge_entries(seeds, astro):
    """Seeds first, in file order, then species detected in space. Every
    detection of an already listed molecule (or another conformer of it) is
    attached to that entry instead of making a new one."""
    entries, by_key = [], {}
    for e in list(seeds) + list(astro):
        key = entry_key(e)
        if key in by_key:
            target = by_key[key]
            if e["detections"] or not target["detections"]:
                target["detections"] += e["detections"]
            else:
                raise ValueError(f"{e['name']}: duplicate of {target['name']}")
            target["synonyms"] += [e["name"]] + e["synonyms"]
            target["notes"] += [n for n in e["notes"] if n not in target["notes"]]
            continue
        e["synonyms"] = list(e["synonyms"])
        if e["charge"] and (key[0], 0) in by_key:
            # A radical ion of a listed molecule (C60+) has its shape.
            parent = by_key[(key[0], 0)]
            e["tags"] = [t for t in parent["tags"] if t in SEED_HINTS]
        by_key[key] = e
        entries.append(e)
    for e in entries:
        # Synonyms other than the name itself, each once (case-insensitive).
        seen, unique = {e["name"].lower()}, []
        for syn in e["synonyms"]:
            if syn.lower() not in seen:
                seen.add(syn.lower())
                unique.append(syn)
        e["synonyms"] = unique
    return entries


# ── records ──────────────────────────────────────────────────────────────────

def h3_plus_geometry():
    """H3+: an equilateral triangle (r = 0.874 A) — three-centre two-electron
    bonding has no Lewis structure, so it is built by hand."""
    r = 0.874
    xyz = np.array([[0.0, r / np.sqrt(3), 0.0],
                    [-r / 2, -r / (2 * np.sqrt(3)), 0.0],
                    [r / 2, -r / (2 * np.sqrt(3)), 0.0]])
    return [["H", *map(float, p)] for p in xyz], [[0, 1, 1], [1, 2, 1], [0, 2, 1]]


SPECIAL_GEOMETRY = {"Trihydrogen cation": h3_plus_geometry}


def _diatomic(r):
    return lambda: np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])


def _bent_triatomic(r, angle_deg):
    """Centre atom second in SMILES order (O=[Ti]=O)."""
    h = np.radians(angle_deg) / 2
    return lambda: np.array([[-r * np.sin(h), r * np.cos(h), 0.0], [0.0, 0.0, 0.0],
                             [r * np.sin(h), r * np.cos(h), 0.0]])


def _t_shaped(r_cc, r_mc):
    """Metal first in SMILES order ([Ca]1[C]=[C]1), C2 unit facing it."""
    h = np.sqrt(r_mc ** 2 - (r_cc / 2) ** 2)
    return lambda: np.array([[0.0, -h, 0.0], [-r_cc / 2, 0.0, 0.0], [r_cc / 2, 0.0, 0.0]])


# Species no force field in RDKit covers (ETKDG alone gives meaningless bond
# lengths for them): equilibrium geometries from the spectroscopy literature,
# coordinates in SMILES atom order.
FIXED_GEOMETRY = {
    "Helium hydride ion": _diatomic(0.774),
    "Argonium": _diatomic(1.280),
    "Iron carbide": _diatomic(1.596),
    "Titanium monoxide": _diatomic(1.620),
    "Vanadium monoxide": _diatomic(1.589),
    "Chromium monoxide": _diatomic(1.615),
    "Calcium oxide": _diatomic(1.822),
    "Calcium sulfide": _diatomic(2.318),
    "Titanium dioxide": _bent_triatomic(1.651, 111.6),
    "Calcium dicarbide": _t_shaped(1.27, 2.32),       # approximate
}


def fixed_geometry(mol, name):
    mh = Chem.AddHs(mol)
    xyz = FIXED_GEOMETRY[name]()
    conf = Chem.Conformer(mh.GetNumAtoms())
    for i, p in enumerate(xyz):
        conf.SetAtomPosition(i, Point3D(*map(float, p)))
    mh.AddConformer(conf, assignId=True)
    return mh


def special_record(entry):
    atoms, bonds = SPECIAL_GEOMETRY[entry["name"]]()
    atoms = [[el] + [round(v, COORD_DECIMALS) + 0.0 for v in xyz]
             for el, *xyz in [[a[0], *orient(np.array([b[1:] for b in atoms]))[k]]
                              for k, a in enumerate(atoms)]]
    return {
        "id": slug(entry["name"]), "name": entry["name"], "synonyms": entry["synonyms"],
        "formula": "H3+", "smiles": "", "inchikey": "", "pubchem_cid": None,
        "charge": entry["charge"] or 0, "multiplicity": entry["multiplicity"] or 1, "rings": 0,
        "tags": sorted(["topology:unfused", "shape:planar", "elements:inorganic", "charge:cation"]
                       + detection_tags(entry)),
        "geometry_source": "fixed", "atoms": atoms, "bonds": bonds,
        **record_extras(entry),
    }


def detection_tags(entry):
    if not entry["detections"]:
        return []
    best = min(entry["detections"], key=lambda d: DETECTION_RANK[d["status"]])
    return [f"detected:{best['status']}"]


def record_extras(entry):
    extras = {}
    if entry["detections"]:
        extras["detections"] = sorted(entry["detections"], key=lambda d: (d["year"] or 0))
    if entry["notes"]:
        extras["notes"] = entry["notes"]
    return extras


def build_record(entry, pc: PubChem):
    """(record, problems) from the first geometry source whose record validates:
    PubChem's 3D conformer, then RDKit. *problems* is empty on success."""
    if not entry["smiles"]:
        if entry["name"] not in SPECIAL_GEOMETRY:
            raise ValueError("no SMILES and no hand-built geometry")
        return special_record(entry), []
    mol = Chem.MolFromSmiles(entry["smiles"])
    inchikey = Chem.MolToInchiKey(mol)
    charge = entry["charge"] if entry["charge"] is not None else Chem.GetFormalCharge(mol)
    radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    if entry["multiplicity"] is None and radicals > 1:
        raise ValueError(f"{radicals} radical electrons: give the ground-state multiplicity")
    multiplicity = entry["multiplicity"] or radicals + 1

    # A charge override is a radical ion of a neutral SMILES: its geometry
    # is the neutral one, built here rather than taken from PubChem's record
    # of the neutral molecule.
    if entry["name"] in FIXED_GEOMETRY:
        return make_record(entry, mol, inchikey, None, charge, multiplicity,
                           fixed_geometry(mol, entry["name"]), "fixed")
    geo, cid, problems = None, None, []
    if entry["charge"] is None:
        geo, cid = pubchem_geometry(pc, entry["name"], inchikey)
    if geo is not None:
        record, problems = make_record(entry, mol, inchikey, cid, charge, multiplicity,
                                       geo, f"pubchem-3d:{cid}")
        if not problems:
            return record, []
    geo, source = rdkit_geometry(mol, curved="nonplanar" in entry["tags"])
    record, rd_problems = make_record(entry, mol, inchikey, cid, charge, multiplicity, geo, source)
    if rd_problems and problems:
        rd_problems = [f"PubChem: {p}" for p in problems] + [f"RDKit: {p}" for p in rd_problems]
    return record, rd_problems


def make_record(entry, mol, inchikey, cid, charge, multiplicity, geo, source):
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
    unknown = set(entry["tags"]) - COLLECTION_TAGS - SEED_HINTS
    if unknown:
        raise ValueError(f"unknown seed tag(s) {sorted(unknown)}")
    heavy = [a.GetIdx() for a in geo.GetAtoms() if a.GetAtomicNum() > 1]
    shape = shape_of(xyz, heavy)
    tags = (facet_tags(mol, shape, charge, multiplicity) + detection_tags(entry)
            + [f"collection:{t}" for t in entry["tags"] if t in COLLECTION_TAGS])

    formula = rdMolDescriptors.CalcMolFormula(mol)
    if entry["charge"] is not None and charge:
        formula += ("+" if charge > 0 else "-") * abs(charge)
    record = {
        "id": slug(entry["name"]),
        "name": entry["name"],
        "synonyms": entry["synonyms"],
        "formula": formula,
        "smiles": Chem.MolToSmiles(mol),
        "inchikey": inchikey,
        "pubchem_cid": cid,
        "charge": charge,
        "multiplicity": multiplicity,
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "tags": sorted(set(tags)),
        "geometry_source": source,
        "atoms": [[a.GetSymbol()] + [round(float(v), COORD_DECIMALS) + 0.0 for v in xyz[a.GetIdx()]]
                  for a in geo.GetAtoms()],
        "bonds": [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_order[b.GetBondType()]]
                  for b in geo.GetBonds()],
        **record_extras(entry),
    }

    problems = [] if source == "fixed" else check_bonds(record)
    if expect_planar(mol, entry["tags"]) and shape == "nonplanar":
        problems.append("expected planar, got a non-planar geometry")
    if "nonplanar" in entry["tags"] and shape != "nonplanar":
        problems.append("tagged nonplanar but the geometry is flat")
    # Geometry-derived InChI carries no stereo layer, and Molvector cannot
    # express a radical or a charge on a specific atom: closed-shell neutral
    # molecules must match exactly (stereo aside), others on their skeleton.
    # A radical cation given as neutral SMILES + charge is checked as that
    # neutral molecule. Metal compounds drawn with per-atom charges (MgNC,
    # HMgNC) are not checked: standard InChI disconnects the metal and
    # Molvector has nowhere to keep the charges it would need.
    flat = Chem.Mol(mol)
    Chem.RemoveStereochemistry(flat)
    expected = Chem.MolToInchiKey(flat)
    has_metal = any(a.GetSymbol() in METALS for a in mol.GetAtoms())
    if not (has_metal and any(a.GetFormalCharge() for a in mol.GetAtoms())):
        as_neutral = entry["charge"] is not None
        rt = inchi_roundtrip(record, neutral=as_neutral)
        exact = as_neutral or (charge == 0 and multiplicity == 1 and Chem.GetFormalCharge(mol) == 0)
        if (rt != expected) if exact else (rt[:14] != expected[:14]):
            problems.append(f"InChI round-trip gave {rt or 'nothing'}, expected {expected}")
    return record, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", default=SEED_PATH)
    ap.add_argument("--astro", default=ASTRO_PATH)
    ap.add_argument("--astro-map", default=ASTRO_MAP_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--cache", default=CACHE_PATH)
    ap.add_argument("--offline", action="store_true", help="fail instead of querying PubChem")
    ap.add_argument("--refresh", action="store_true", help="ignore the PubChem cache")
    args = ap.parse_args(argv)

    entries = merge_entries(read_seed(args.seed), read_astro(args.astro, args.astro_map))
    pc = PubChem(args.cache, offline=args.offline, refresh=args.refresh)
    records, seen_ids, failures = [], set(), 0
    try:
        for entry in entries:
            try:
                rec, problems = build_record(entry, pc)
            except Exception as e:
                print(f"FAIL  {entry['name']}: {e}")
                failures += 1
                continue
            if rec["id"] in seen_ids:
                print(f"FAIL  {entry['name']}: duplicate id {rec['id']}")
                failures += 1
                continue
            if problems:
                print(f"FAIL  {entry['name']}: " + "; ".join(problems))
                failures += 1
                continue
            seen_ids.add(rec["id"])
            records.append(rec)
            print(f"ok    {rec['name']:<34} {rec['formula']:<10} {len(rec['atoms']):>4} atoms  "
                  f"{rec['geometry_source']}")
    finally:
        pc.save()

    if failures:
        print(f"\n{failures} entr{'y' if failures == 1 else 'ies'} failed; catalog not written.")
        return 1

    payload = {"version": CATALOG_VERSION, "molecules": records}
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    _write_gzip(args.out, data)
    print(f"\nWrote {len(records)} molecules to {os.path.relpath(args.out, REPO)} "
          f"({os.path.getsize(args.out) / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
