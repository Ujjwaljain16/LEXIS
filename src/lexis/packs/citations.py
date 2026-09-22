"""
Generic citation extraction driven entirely by a pack's grammars. The
extractor knows no citation format itself: adding AIR, SCC, Bluebook, UK
neutral citations, ECLI, CELEX or any other scheme is a pack-data change.
"""
import re
from dataclasses import dataclass
from typing import List

from lexis.packs.schema import JurisdictionPack


@dataclass(frozen=True)
class CitationMatch:
    grammar: str
    text: str
    start: int
    end: int


def extract_citations(text: str, pack: JurisdictionPack) -> List[CitationMatch]:
    """All non-overlapping citations found in text. When two grammars match
    overlapping spans the earlier start wins, then the longer span, then the
    grammar listed first in the pack (deterministic)."""
    candidates = []
    for order, grammar in enumerate(pack.citation_grammars):
        for m in re.finditer(grammar.pattern, text):
            if m.end() > m.start():
                candidates.append((m.start(), -(m.end() - m.start()), order, CitationMatch(grammar.name, m.group(0), m.start(), m.end())))
    candidates.sort(key=lambda c: c[:3])
    chosen: List[CitationMatch] = []
    last_end = -1
    for _, _, _, match in candidates:
        if match.start >= last_end:
            chosen.append(match)
            last_end = match.end
    return chosen
