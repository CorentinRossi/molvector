"""
catalog.py — Built-in molecule catalog
======================================
A curated set of PAHs and related molecules (benzene to circumcoronene, the
16 EPA priority PAHs, fullerenes, ...) plus every molecule and ion detected
in space as listed by The Astrochymist (each with its detection paper),
bundled as ``molvector/catalog/catalog.json.gz``.

The file is produced offline by ``tools/catalog/build_catalog.py`` (which
needs RDKit and network access to PubChem); nothing here needs RDKit.

  records()                       — every catalog record (loaded lazily, cached)
  facets()                        — the filter rows (FACETS plus the common atoms)
  search(query, filters, rings, modes, sort) — records matching a name /
                                    synonym / formula / SMILES, the facet
                                    filters and a ring range, sorted
  facet_counts(query, filters, rings, modes) — per-option hit counts for the UI
  mass(record)                    — molecular mass (g/mol)
  sources()                       — data sources and their update dates
  to_molecule(record)             — a render.Molecule ready to display,
                                    turned to a standard orientation

Records carry "<facet>:<value>" tags: the ones written by the build, plus a
"class:" tag and "atom:<element>" tags derived here when the file is loaded.
Within a facet the selected values combine with OR (any of them) or, in that
facet's "and" mode, AND (all of them); different facets must all match.
"""

import gzip
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from molvector.render import ATOMIC_MASSES, Atom, Bond, Molecule

CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "catalog", "catalog.json.gz")

# (facet, label, [(value, label, tooltip), ...]) in display order. Every
# record has exactly one topology and shape value and at least one elements
# and electronic value; ring-system values apply to molecules with rings and
# detection values to species detected in space (see
# tools/catalog/build_catalog.py).
FACETS = [
    ("class", "Class", [
        ("pah", "PAH", "Polycyclic aromatic hydrocarbons and their derivatives "
                       "(methyl-, cyano-, ethynyl-, hydro-PAHs; biphenyls)"),
        ("fullerene", "Fullerene", "Closed carbon cages and their ions"),
        ("heterocycle", "Heterocycle", "Rings containing N, O, S, Si, … (pyridine, quinoline, ethylene oxide)"),
        ("monocyclic", "Single ring", "One carbocyclic ring, with substituents (benzene, phenol, cyclopropenylidene)"),
        ("chain", "Carbon chain", "Linear chains of three or more carbons (polyynes, cyanopolyynes, Cn, CnH)"),
        ("organic", "Other organic", "Other carbon compounds (methanol, acetone, glycine, HCN, …)"),
        ("inorganic", "Inorganic", "No carbon (H2O, NH3, SiO, metal oxides, H3+, …)"),
    ]),
    ("topology", "Topology", [
        ("acyclic", "Acyclic", "No rings: diatomics, chains, carbon-chain radicals"),
        ("unfused", "Unfused", "A single ring or rings joined by bonds (benzene, biphenyl)"),
        ("catacondensed", "Catacondensed", "Fused rings, no atom shared by three rings"),
        ("pericondensed", "Pericondensed", "Fused rings with internal atoms shared by three rings"),
        ("cage", "Cage", "Closed carbon cages (fullerenes)"),
    ]),
    ("shape", "Shape", [
        ("linear", "Linear", "All atoms on one line (diatomics, polyynes, cyanopolyynes)"),
        ("planar", "Planar", "All heavy atoms within 0.1 Å of one plane"),
        ("nonplanar", "Non-planar", "Bowls, helicenes, cages, sp3 centers, twisted rings"),
    ]),
    # (records also carry "elements:inorganic"; the Class row offers it)
    ("elements", "Composition", [
        ("hydrocarbon", "Hydrocarbon", "Carbon and hydrogen only"),
        ("heteroatom", "C + heteroatoms", "Carbon compounds with N, O, S, P, Si, halogens, …"),
        ("metal", "Metal-bearing", "Na, K, Mg, Al, Ca, Ti, V, Cr, Fe, …"),
    ]),
    ("rings", "Rings", [
        ("benzenoid", "Benzenoid", "Hydrocarbon made only of six-membered rings"),
        ("nonbenzenoid", "Non-benzenoid", "Hydrocarbon with 3-, 4-, 5- or 7-membered rings"),
        ("hydro", "Hydro (CH₂)", "Partly saturated: a ring CH₂ group"),
        ("heterocycle", "Heterocycle", "N, O or S in a ring"),
        ("substituted", "Substituted", "Side groups on the rings: methyl, cyano, ethynyl, …"),
    ]),
    ("charge", "Electronic", [
        ("neutral", "Neutral", "No net charge"),
        ("cation", "Cation", "Positive charge"),
        ("anion", "Anion", "Negative charge"),
        ("open-shell", "Open shell", "Radicals and other multiplicity > 1 species"),
    ]),
    ("detected", "Detected in space", [
        ("confirmed", "Detected", "Detected in the interstellar medium or circumstellar shells"),
        ("tentative", "Tentative", "Detection reported as tentative"),
        ("disputed", "Disputed", "Reported detection that has since been disputed"),
    ]),
    ("collection", "Collections", [
        ("epa16", "EPA 16", "The 16 US EPA priority PAHs"),
        ("astro", "Astro PAHs", "PAHs studied in astrochemistry"),
        ("acene", "Acenes", "Linearly fused benzene rings"),
        ("helicene", "Helicenes", "Ortho-fused, helically twisted"),
        ("bowl", "Bowls", "Curved buckybowls"),
    ]),
]

Filters = Dict[str, Iterable[str]]
Modes = Dict[str, str]           # facet -> "or" (default) / "and"

ATOM_FACET_SIZE = 12             # how many of the most common elements get a button
SORTS = [("relevance", "Relevance"), ("name", "Name"),
         ("mass", "Mass ↑"), ("mass-desc", "Mass ↓")]

_records: Optional[List[dict]] = None
_sources: Dict[str, dict] = {}
_facets: Optional[list] = None


def _molecule_class(r: dict) -> str:
    """One "class" per record, from the build's tags (first rule that fits)."""
    tags = set(r.get("tags", ()))
    elements = {a[0] for a in r["atoms"]}
    n_c = sum(1 for a in r["atoms"] if a[0] == "C")
    if "topology:cage" in tags:
        return "fullerene"
    if "C" not in elements:
        return "inorganic"
    if "rings:heterocycle" in tags:
        return "heterocycle"
    fused = tags & {"topology:catacondensed", "topology:pericondensed"}
    if fused or ("rings:benzenoid" in tags and r["rings"] >= 2):
        return "pah"
    if r["rings"] >= 1:
        return "monocyclic"
    if "shape:linear" in tags and n_c >= 3:
        return "chain"
    return "organic"


def records() -> List[dict]:
    """Return every catalog record, reading the bundled file on first use
    (and adding the derived "class:" and "atom:" tags)."""
    global _records, _sources
    if _records is None:
        with gzip.open(CATALOG_PATH, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        loaded = payload["molecules"]
        _sources = payload.get("sources", {})
        for r in loaded:
            derived = [f"class:{_molecule_class(r)}"] + [f"atom:{el}" for el in sorted({a[0] for a in r["atoms"]})]
            r["tags"] = list(r.get("tags", ())) + derived
            r["mass"] = round(sum(ATOMIC_MASSES.get(a[0], 0.0) for a in r["atoms"]), 3)
        _records = loaded
    return _records


def sources() -> Dict[str, dict]:
    """Where catalog data came from, with dates: {"astrochymist": {"name",
    "url", "updated" (the list's own last update), "fetched"}}."""
    records()
    return _sources


def facets() -> list:
    """FACETS with an "Atoms" row (second) for the most common elements."""
    global _facets
    if _facets is None:
        counts: Dict[str, int] = {}
        for r in records():
            for el in {a[0] for a in r["atoms"]}:
                counts[el] = counts.get(el, 0) + 1
        common = sorted(counts, key=lambda el: (-counts[el], el))[:ATOM_FACET_SIZE]
        atoms = ("atom", "Atoms", [(el, el, f"Contains {el}") for el in common])
        _facets = FACETS[:1] + [atoms] + FACETS[1:]
    return _facets


def mass(record: dict) -> float:
    return record["mass"] if "mass" in record else \
        sum(ATOMIC_MASSES.get(a[0], 0.0) for a in record["atoms"])


def ring_range() -> Tuple[int, int]:
    counts = [r["rings"] for r in records()]
    return min(counts), max(counts)


def tag_label(tag: str) -> str:
    """Display label of a "<facet>:<value>" tag."""
    facet, _, value = tag.partition(":")
    for name, _, options in facets():
        if name == facet:
            for v, label, _ in options:
                if v == value:
                    return label
    return value or tag


def _match_rank(record: dict, q: str, q_low: str) -> Optional[int]:
    """Lower is a better match; None means no match."""
    name = record["name"].lower()
    if name == q_low:
        return 0
    if record["formula"] == q:
        return 1
    if name.startswith(q_low):
        return 2
    synonyms = [s.lower() for s in record.get("synonyms", [])]
    if q_low in synonyms:
        return 3
    if q_low in name:
        return 4
    if any(q_low in s for s in synonyms):
        return 5
    if q_low in record["formula"].lower():
        return 6
    if q in record["smiles"] or record["inchikey"] == q.upper():
        return 7
    return None


def _passes(record: dict, filters: Filters, rings: Optional[Tuple[int, int]],
            modes: Optional[Modes] = None, skip_facet: Optional[str] = None) -> bool:
    if rings is not None and not rings[0] <= record["rings"] <= rings[1]:
        return False
    tags = record.get("tags", ())
    for facet, values in filters.items():
        if facet == skip_facet or not values:
            continue
        combine = all if (modes or {}).get(facet) == "and" else any
        if not combine(f"{facet}:{v}" in tags for v in values):
            return False
    return True


def _query_hits(query: str) -> List[Tuple[int, int, dict]]:
    q = query.strip()
    q_low = q.lower()
    hits = []
    for idx, r in enumerate(records()):
        rank = _match_rank(r, q, q_low) if q else 0
        if rank is not None:
            hits.append((rank, idx, r))
    return hits


def search(query: str = "", filters: Optional[Filters] = None,
           rings: Optional[Tuple[int, int]] = None, modes: Optional[Modes] = None,
           sort: str = "relevance") -> List[dict]:
    """
    Records whose name, synonym, formula or SMILES contains *query*
    (case-insensitive, except SMILES). *filters* maps a facet to its
    selected values, combined with OR — or AND where *modes* says "and";
    every facet with a selection must match. *rings* is an inclusive (min,
    max) ring-count range. *sort*: "relevance" (best matches first, else
    catalog order), "name", "mass" or "mass-desc".
    """
    filters = filters or {}
    hits = [h for h in _query_hits(query) if _passes(h[2], filters, rings, modes)]
    if sort == "name":
        hits.sort(key=lambda h: (h[2]["name"].lower(), h[1]))
    elif sort in ("mass", "mass-desc"):
        sign = -1 if sort == "mass-desc" else 1
        hits.sort(key=lambda h: (sign * mass(h[2]), h[1]))
    else:
        hits.sort(key=lambda h: (h[0], h[1]))
    return [r for _, _, r in hits]


def facet_counts(query: str = "", filters: Optional[Filters] = None,
                 rings: Optional[Tuple[int, int]] = None,
                 modes: Optional[Modes] = None) -> Dict[str, Dict[str, int]]:
    """
    For every facet option, how many records the search would return with
    that option selected, given the query, the ring range and the other
    selections. In an OR facet the option's own facet is ignored (its values
    are alternatives, so selecting one more only adds its own hits); in an
    AND facet the option must hold together with the values already chosen.
    """
    filters = filters or {}
    modes = modes or {}
    hits = [h[2] for h in _query_hits(query)]
    counts: Dict[str, Dict[str, int]] = {}
    for facet, _, options in facets():
        own = None if modes.get(facet) == "and" else facet
        pool = [r for r in hits if _passes(r, filters, rings, modes, skip_facet=own)]
        counts[facet] = {v: sum(1 for r in pool if f"{facet}:{v}" in r.get("tags", ()))
                         for v, _, _ in options}
    return counts


def get(record_id: str) -> Optional[dict]:
    return next((r for r in records() if r["id"] == record_id), None)


def _standard_orientation(elements: List[str], xyz: np.ndarray,
                          bonds: List[Tuple[int, int]]) -> np.ndarray:
    """
    *xyz* centered and turned (a proper rotation: chirality is kept) to how
    a chemist would draw it, seen down z as the 3D view first shows it: the
    heavy-atom skeleton's longest axis along x, its plane (if any) facing
    the viewer, and a ring lattice turned so its bonds run vertical or at
    +-30 degrees (benzene with two vertical sides, naphthalene lying flat).
    """
    xyz = xyz - xyz.mean(axis=0)
    heavy = [i for i, el in enumerate(elements) if el != "H"]
    core = xyz[heavy] if len(heavy) >= 2 else xyz
    if len(core) < 2:
        return xyz
    core = core - core.mean(axis=0)
    axes = np.linalg.svd(core, full_matrices=True)[2]   # rows: longest .. shortest axis
    if np.linalg.det(axes) < 0:
        axes[2] = -axes[2]
    xyz = xyz @ axes.T

    # Ring lattice: the heavy-atom bond directions seen in the plane, with
    # 6-fold symmetry. Turn by the smallest angle that puts them at 90
    # degrees (mod 60), when they agree on a lattice at all.
    acc = 0j
    weight = 0.0
    for i, j in bonds:
        if elements[i] == "H" or elements[j] == "H":
            continue
        dx, dy = xyz[j, 0] - xyz[i, 0], xyz[j, 1] - xyz[i, 1]
        w = dx * dx + dy * dy
        acc += w * complex(math.cos(6 * math.atan2(dy, dx)), math.sin(6 * math.atan2(dy, dx)))
        weight += w
    if weight > 1e-9 and abs(acc) > 0.5 * weight:
        phi = (math.pi - math.atan2(acc.imag, acc.real)) / 6.0
        phi = (phi + math.pi / 6) % (math.pi / 3) - math.pi / 6   # smallest turn
        c, s = math.cos(phi), math.sin(phi)
        xyz = xyz @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return xyz


def to_molecule(record: dict) -> Molecule:
    """Build a Molecule (3D coordinates in Angstrom, Kekule bond orders) in
    the standard orientation of _standard_orientation."""
    elements = [el for el, _x, _y, _z in record["atoms"]]
    bonds = [Bond(int(i), int(j), int(order)) for i, j, order in record["bonds"]]
    xyz = _standard_orientation(
        elements, np.array([[x, y, z] for _el, x, y, z in record["atoms"]], dtype=float),
        [(b.i, b.j) for b in bonds])
    atoms = [Atom(el, float(x), float(y), float(z)) for el, (x, y, z) in zip(elements, xyz)]
    mult = record.get("multiplicity")
    return Molecule(name=record["name"], atoms=atoms, bonds=bonds,
                    charge=int(record.get("charge", 0)),
                    multiplicity=int(mult) if mult is not None else None)
