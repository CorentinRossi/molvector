"""
catalog.py — Built-in molecule catalog
======================================
A curated set of PAHs and related molecules (benzene to circumcoronene, the
16 EPA priority PAHs, fullerenes, astro-relevant cyano-PAHs, ...) bundled as
``molvector/catalog/catalog.json.gz``.

The file is produced offline by ``tools/catalog/build_catalog.py`` (which
needs RDKit and network access to PubChem); nothing here needs RDKit.

  records()             — every catalog record (loaded lazily, cached)
  search(query, tags)   — records matching a name / synonym / formula / SMILES
  all_tags()            — every tag used in the catalog, sorted
  to_molecule(record)   — a render.Molecule ready to display
"""

import gzip
import json
import os
from typing import Iterable, List, Optional

from molvector.render import Atom, Bond, Molecule

CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "catalog", "catalog.json.gz")

_records: Optional[List[dict]] = None


def records() -> List[dict]:
    """Return every catalog record, reading the bundled file on first use."""
    global _records
    if _records is None:
        with gzip.open(CATALOG_PATH, "rt", encoding="utf-8") as f:
            _records = json.load(f)["molecules"]
    return _records


def all_tags() -> List[str]:
    return sorted({t for r in records() for t in r.get("tags", [])})


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


def search(query: str = "", tags: Iterable[str] = ()) -> List[dict]:
    """
    Records whose name, synonym, formula or SMILES contains *query*
    (case-insensitive, except SMILES), best matches first. With *tags*,
    only records carrying every one of those tags are returned. An empty
    query returns every record (filtered by *tags*) in catalog order.
    """
    q = query.strip()
    q_low = q.lower()
    wanted = set(tags)
    hits = []
    for idx, r in enumerate(records()):
        if wanted and not wanted.issubset(r.get("tags", ())):
            continue
        rank = _match_rank(r, q, q_low) if q else 0
        if rank is not None:
            hits.append((rank, idx, r))
    hits.sort(key=lambda h: (h[0], h[1]))
    return [r for _, _, r in hits]


def get(record_id: str) -> Optional[dict]:
    return next((r for r in records() if r["id"] == record_id), None)


def to_molecule(record: dict) -> Molecule:
    """Build a Molecule (3D coordinates in Angstrom, Kekule bond orders)."""
    atoms = [Atom(el, float(x), float(y), float(z)) for el, x, y, z in record["atoms"]]
    bonds = [Bond(int(i), int(j), int(order)) for i, j, order in record["bonds"]]
    return Molecule(name=record["name"], atoms=atoms, bonds=bonds,
                    charge=int(record.get("charge", 0)))
