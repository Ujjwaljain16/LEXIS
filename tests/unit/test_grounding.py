"""generation/grounding.py: the model can never make a citation appear that was not in its context,
and unverifiable claims are never reported as supported."""
from lexis.generation.grounding import (
    ABSTENTION_MARKER, build_sources, is_abstention, resolve_citations, split_claims, verify_claims,
)


def chunk(i, text, doc="doc-a"):
    return {"payload": {"chunk_id": f"c{i}", "doc_id": doc, "content": text, "page_num": i, "chunk_index": i, "doc_type": "t"},
            "text": "IGNORED-because-payload-content-wins"}


def make_sources(n=3):
    return build_sources([chunk(i, f"Source text number {i}.") for i in range(1, n + 1)])


def test_build_sources_numbers_from_one_and_prefers_raw_payload_content():
    sources = make_sources()
    assert [s.index for s in sources] == [1, 2, 3]
    assert sources[0].text == "Source text number 1."          # not the retrieval "text" field
    assert sources[1].chunk_id == "c2" and sources[1].doc_id == "doc-a" and sources[1].page_num == 2


def test_build_sources_skips_empty_chunks_so_numbers_map_to_real_content():
    sources = build_sources([chunk(1, "real"), {"payload": {"chunk_id": "x", "content": "  "}, "text": ""}, chunk(3, "also real")])
    assert [s.text for s in sources] == ["real", "also real"]
    assert [s.index for s in sources] == [1, 2]


def test_resolve_keeps_valid_and_strips_hallucinated_indices():
    answer, cited, invalid = resolve_citations("A is true [1]. B is true [9]. C is true [2, 7].", make_sources())
    assert answer == "A is true [1]. B is true. C is true [2]."
    assert [s.index for s in cited] == [1, 2]
    assert sorted(invalid) == [7, 9]


def test_resolve_orders_cited_sources_by_first_use_and_dedupes():
    _, cited, _ = resolve_citations("X [3]. Y [1]. Z [3, 1].", make_sources())
    assert [s.index for s in cited] == [3, 1]


def test_resolve_with_no_citations():
    answer, cited, invalid = resolve_citations("No markers here.", make_sources())
    assert (answer, cited, invalid) == ("No markers here.", [], [])


def test_abstention_detected():
    assert is_abstention(f"  {ABSTENTION_MARKER}  ")
    assert not is_abstention("The term is five years [1].")


def test_split_claims_extracts_sentences_and_their_citations():
    claims = split_claims("The term is five years [1]. It renews yearly [2, 3]. Uncited sentence here.")
    assert [c.text for c in claims] == ["The term is five years.", "It renews yearly.", "Uncited sentence here."]
    assert [c.cited for c in claims] == [(1,), (2, 3), ()]


class FakeChecker:
    def __init__(self, entails):
        self.entails = entails   # set of (premise, hypothesis) pairs that count as entailed

    def check_entailment(self, premise, hypothesis, threshold=None):
        return (premise, hypothesis) in self.entails


def test_verify_claims_supported_unsupported_and_uncited():
    sources = make_sources()
    answer = "Claim one [1]. Claim two [2]. Claim three."
    checker = FakeChecker({("Source text number 1.", "Claim one.")})
    report = verify_claims(answer, sources, checker)
    assert [v.status for v in report.verdicts] == ["supported", "unsupported", "uncited"]
    assert report.support_rate == 1 / 3


def test_verify_claims_requires_a_cited_source_to_entail_not_any_source():
    sources = make_sources()
    # Source 2 entails the claim but the claim cites only source 1 -> must NOT count as supported.
    checker = FakeChecker({("Source text number 2.", "Claim one.")})
    report = verify_claims("Claim one [1].", sources, checker)
    assert report.verdicts[0].status == "unsupported"


def test_verify_claims_ignores_invalid_citation_numbers():
    report = verify_claims("Claim one [8].", make_sources(), FakeChecker(set()))
    assert report.verdicts[0].status == "uncited"


def test_support_rate_none_without_claims():
    assert verify_claims("", make_sources(), FakeChecker(set())).support_rate is None
