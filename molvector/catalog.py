"""
catalog.py — Built-in molecule catalog
======================================
A curated set of PAHs and related molecules (benzene to circumcoronene, the
16 EPA priority PAHs, fullerenes, astro-relevant cyano-PAHs, ...) bundled as
``molvector/catalog/catalog.json.gz``.

The file is produced offline by ``tools/catalog/build_catalog.py`` (which
needs RDKit and network access to PubChem); nothing here needs RDKit.

  records()                       — every catalog record (loaded lazily, cached)
  search(query, filters, rings)   — records matching a name / synonym / formula /
                                    SMILES, the facet filters and a ring range
  facet_counts(query, filters, rings) — per-option hit counts for the filter UI
  to_molecule(record)             — a render.Molecule ready to display

Records carry "<facet>:<value>" tags. Within a facet the selected values are
alternatives (OR); different facets must all match (AND).
"""

import gzip
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

from molvector.render import Atom, Bond, Molecule

CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "catalog", "catalog.json.gz")

# (facet, label, [(value, label, tooltip), ...]) in display order. Every
# record has exactly one topology, shape and charge value and at least one
# composition value (see tools/catalog/build_catalog.py).
FACETS = [
    ("topology", "Topology", [
        ("unfused", "Unfused", "A single ring or rings joined by bonds (benzene, biphenyl)"),
        ("catacondensed", "Catacondensed", "Fused rings, no atom shared by three rings"),
        ("pericondensed", "Pericondensed", "Fused rings with internal atoms shared by three rings"),
        ("cage", "Cage", "Closed carbon cages (fullerenes)"),
    ]),
    ("shape", "Shape", [
        ("planar", "Planar", "All heavy atoms within 0.1 Å of one plane"),
        ("nonplanar", "Non-planar", "Bowls, helicenes, cages, twisted or saturated rings"),
    ]),
    ("composition", "Composition", [
        ("benzenoid", "Benzenoid", "Hydrocarbon made only of six-membered rings"),
        ("nonbenzenoid", "Non-benzenoid", "Hydrocarbon with 4-, 5- or 7-membered rings"),
        ("hydro", "Hydro (CH₂)", "Partly saturated: a ring CH₂ group"),
        ("heteroatom", "Heteroatom", "N, O or S in a ring"),
        ("substituted", "Substituted", "Side groups: methyl, vinyl, ethynyl, cyano"),
    ]),
    ("charge", "Electronic", [
        ("closed-shell", "Neutral", "Neutral closed-shell molecule"),
        ("cation", "Cation", "Positive charge"),
        ("anion", "Anion", "Negative charge"),
        ("radical", "Radical", "Open shell (multiplicity > 1)"),
    ]),
    ("collection", "Collections", [
        ("epa16", "EPA 16", "The 16 US EPA priority PAHs"),
        ("astro", "Astro", "Interstellar / astrochemistry relevant"),
        ("acene", "Acenes", "Linearly fused benzene rings"),
        ("helicene", "Helicenes", "Ortho-fused, helically twisted"),
        ("bowl", "Bowls", "Curved buckybowls"),
    ]),
]

Filters = Dict[str, Iterable[str]]

_records: Optional[List[dict]] = None


def records() -> List[dict]:
    """Return every catalog record, reading the bundled file on first use."""
    global _records
    if _records is None:
        with gzip.open(CATALOG_PATH, "rt", encoding="utf-8") as f:
            _records = json.load(f)["molecules"]
    return _records


def ring_range() -> Tuple[int, int]:
    counts = [r["rings"] for r in records()]
    return min(counts), max(counts)


def tag_label(tag: str) -> str:
    """Display label of a "<facet>:<value>" tag."""
    facet, _, value = tag.partition(":")
    for name, _, options in FACETS:
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
            skip_facet: Optional[str] = None) -> bool:
    if rings is not None and not rings[0] <= record["rings"] <= rings[1]:
        return False
    tags = record.get("tags", ())
    for facet, values in filters.items():
        if facet == skip_facet or not values:
            continue
        if not any(f"{facet}:{v}" in tags for v in values):
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
           rings: Optional[Tuple[int, int]] = None) -> List[dict]:
    """
    Records whose name, synonym, formula or SMILES contains *query*
    (case-insensitive, except SMILES), best matches first; an empty query
    keeps catalog order. *filters* maps a facet to its selected values (any
    of them may match); every facet with a selection must match. *rings*
    is an inclusive (min, max) ring-count range.
    """
    filters = filters or {}
    hits = [h for h in _query_hits(query) if _passes(h[2], filters, rings)]
    hits.sort(key=lambda h: (h[0], h[1]))
    return [r for _, _, r in hits]


def facet_counts(query: str = "", filters: Optional[Filters] = None,
                 rings: Optional[Tuple[int, int]] = None) -> Dict[str, Dict[str, int]]:
    """
    For every facet option, how many records the search would return with
    that option selected, given the query, the ring range and the other
    facets' selections (the option's own facet is ignored: its values are
    alternatives, so selecting one more only adds its own hits).
    """
    filters = filters or {}
    hits = [h[2] for h in _query_hits(query)]
    counts: Dict[str, Dict[str, int]] = {}
    for facet, _, options in FACETS:
        pool = [r for r in hits if _passes(r, filters, rings, skip_facet=facet)]
        counts[facet] = {v: sum(1 for r in pool if f"{facet}:{v}" in r.get("tags", ()))
                         for v, _, _ in options}
    return counts


def get(record_id: str) -> Optional[dict]:
    return next((r for r in records() if r["id"] == record_id), None)


def to_molecule(record: dict) -> Molecule:
    """Build a Molecule (3D coordinates in Angstrom, Kekule bond orders)."""
    atoms = [Atom(el, float(x), float(y), float(z)) for el, x, y, z in record["atoms"]]
    bonds = [Bond(int(i), int(j), int(order)) for i, j, order in record["bonds"]]
    return Molecule(name=record["name"], atoms=atoms, bonds=bonds,
                    charge=int(record.get("charge", 0)))
