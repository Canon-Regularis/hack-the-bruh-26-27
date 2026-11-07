"""Look up which edges a street name refers to (ALG-MVP-2, #19, the issue's "gazetteer").

Stand-in for orchid's held policy_compiler.Gazetteer until ALG-S3-2 swaps it in.
"""

import re
from collections import defaultdict

from engine.orchid import street_network as sn

STREET_TYPES = {"st": "street", "rd": "road", "ave": "avenue", "ter": "terrace", "pl": "place",
                "sq": "square", "cres": "crescent", "gdns": "gardens", "ct": "court", "dr": "drive",
                "ln": "lane"}
APOSTROPHES = re.compile(r"['’]")
PUNCTUATION = re.compile(r"[^\w\s]")


def normalise_name(name: str) -> str:
    """Return a street name in a standard form for matching.

    Ignores case, punctuation and abbreviations ("Johnston Ter" == "Johnston Terrace",
    "St Mary's St" == "Saint Marys Street"). Every word is kept, so qualifiers like
    "Upper" stay and Upper Bow never matches Bow.
    """
    words = PUNCTUATION.sub(" ", APOSTROPHES.sub("", name.lower())).split()
    if len(words) > 1 and words[0] == "st":
        words[0] = "saint"
    if words:
        words[-1] = STREET_TYPES.get(words[-1], words[-1])
    return " ".join(words)


def build_street_lookup(graph: sn.RoadGraph) -> dict[str, list[str]]:
    """Map each normalised street name to all its edge ids as str(eid), both directions."""
    lookup = defaultdict(list)
    for edge in graph.iter_edges():
        if edge.street_name:
            lookup[normalise_name(edge.street_name)].append(str(edge.eid))
    return dict(lookup)


def find_streets(lookup: dict[str, list[str]], names: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Find the edges for street names from clauses or complaints.

    Returns the edge ids for each name that matched, and the names that didn't.
    """
    found, unresolved = {}, []
    for name in names:
        eids = lookup.get(normalise_name(name))
        if eids:
            found[name] = eids
        else:
            unresolved.append(name)
    return found, unresolved
