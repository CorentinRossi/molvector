#!/usr/bin/env python3
"""
Snapshot The Astrochymist's list of interstellar and circumstellar molecules
(https://www.astrochymist.org/astrochymist_ism.html) into astrochymist.json:
one entry per detection row with the species as written on the page, its
common name, the year, the first-listed (detection) paper and whether the
page marks it tentative (italics) or disputed (orange).

    python tools/catalog/fetch_astrochymist.py

build_catalog.py reads the snapshot, never the live page, so catalog builds
stay reproducible; re-run this script to pick up new detections, then map
any new species in astrochymist_smiles.csv.
"""

import datetime
import html
import json
import os
import re
import sys
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_URL = "https://www.astrochymist.org/astrochymist_ism.html"
OUT_PATH = os.path.join(HERE, "astrochymist.json")


def _clean(fragment: str) -> str:
    s = re.sub(r"<SUB>(.*?)</SUB>", r"\1", fragment, flags=re.I | re.S)
    s = re.sub(r"<SUP>(.*?)</SUP>", r"^\1", s, flags=re.I | re.S)
    s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    return re.sub(r"\s+", " ", s).strip()


def parse(page: str) -> list:
    # Each detection row starts with "<TR>  <!-- yyyy.mm -->"; comments are
    # only stripped after splitting on those markers.
    parts = re.split(r"(<TR>\s*<!--\s*[\d.]+\s*-->)", page)
    rows = []
    for body in parts[2::2]:
        body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
        body = re.split(r"<TR\b", body, flags=re.I)[0]
        cells = re.split(r"<T[HD]\b[^>]*>", body, flags=re.I)[1:]
        if len(cells) < 3:
            continue
        year = re.search(r"\b(1[89]\d\d|20\d\d)\b", _clean(cells[0]))
        species = re.sub(r"<A\b.*?</A>", "", cells[1], flags=re.I | re.S)   # drop badges
        lines = [x for x in (_clean(p) for p in re.split(r"<BR\s*/?>", species, flags=re.I)) if x]
        paper = re.search(r'<A HREF="([^"]+)"[^>]*>(.*?)</A>(.*?)(?:<A HREF|$)', cells[2],
                          flags=re.I | re.S)
        rows.append({
            "formula": lines[0] if lines else "",
            "name": " ".join(lines[1:]),
            "year": int(year.group(1)) if year else None,
            "tentative": bool(re.search(r"<I>", cells[1], re.I)),
            "disputed": bool(re.search(r"COLOR=ORANGE", cells[1], re.I)),
            "paper_url": urllib.parse.urljoin(SOURCE_URL, paper.group(1)) if paper else None,
            "paper_title": _clean(paper.group(2)) if paper else None,
            "paper_ref": _clean(paper.group(3)) if paper else None,
        })
    return rows


def main():
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as r:
        page = r.read().decode("utf-8", "replace")
    rows = parse(page)
    keys = [(r["formula"], r["name"]) for r in rows]
    if len(set(keys)) != len(keys):
        sys.exit("duplicate (formula, name) rows: the mapping key would be ambiguous")
    snapshot = {
        "source": SOURCE_URL,
        "fetched": datetime.date.today().isoformat(),
        "detections": rows,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{len(rows)} detections -> {os.path.relpath(OUT_PATH)}")


if __name__ == "__main__":
    main()
