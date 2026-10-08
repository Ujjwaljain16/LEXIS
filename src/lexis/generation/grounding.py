"""
Citation grounding for generated answers.

The generator is shown numbered sources ("[1] ...") and told to cite them by number. Everything
here is deterministic post-processing of what it actually said, so a citation is only ever shown
to a user if it resolves to a chunk that was really in the context:

  build_sources          retrieved chunk dicts -> numbered SourceRef list (what the LLM sees)
  resolve_citations      answer text -> cleaned text + the SourceRefs actually cited; markers that
                         point at no source ("[9]" with 5 sources) are stripped and reported
  split_claims           answer text -> sentence-level claims with their cited indices
  verify_claims          per-claim support: does a CITED source entail the claim? (NLI, injected)
  is_abstention          the model's explicit "not in the context" marker

Nothing in this module calls an LLM or loads a model; the NLI checker is passed in, so it is unit
testable with a fake and the (unmeasured-on-legal-text) NLI stays an explicit opt-in.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple

ABSTENTION_MARKER = "INSUFFICIENT_CONTEXT"

_CITE_GROUP = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass(frozen=True)
class SourceRef:
    index: int            # 1-based number the LLM was shown
    chunk_id: str
    doc_id: str
    text: str             # exact chunk text shown to the model (also the citation span)
    page_num: Optional[int] = None
    chunk_index: Optional[int] = None
    doc_type: Optional[str] = None

    def to_public(self) -> Dict[str, Any]:
        return {"index": self.index, "chunk_id": self.chunk_id, "doc_id": self.doc_id, "text": self.text,
                "page_num": self.page_num, "chunk_index": self.chunk_index, "doc_type": self.doc_type}


def build_sources(chunks: Sequence[Dict[str, Any]]) -> List[SourceRef]:
    """Numbers retrieved chunk dicts (RetrievalEngine's shape: {"payload": {...}, "text": ...})
    starting at 1. Chunks with no usable text are skipped so every number maps to real content."""
    sources: List[SourceRef] = []
    for chunk in chunks:
        payload = chunk.get("payload") or {}
        text = payload.get("content") or chunk.get("text") or chunk.get("content") or ""
        if not str(text).strip():
            continue
        sources.append(SourceRef(
            index=len(sources) + 1,
            chunk_id=payload.get("chunk_id") or chunk.get("chunk_id") or chunk.get("id") or "",
            doc_id=payload.get("doc_id", ""),
            text=str(text),
            page_num=payload.get("page_num"),
            chunk_index=payload.get("chunk_index"),
            doc_type=payload.get("doc_type"),
        ))
    return sources


def format_context(sources: Sequence[SourceRef]) -> str:
    return "\n\n".join(f"[{s.index}] {s.text}" for s in sources)


def is_abstention(answer: str) -> bool:
    return ABSTENTION_MARKER in answer


def _indices_in(group: str) -> List[int]:
    return [int(part) for part in re.split(r"\s*,\s*", group.strip())]


def resolve_citations(answer: str, sources: Sequence[SourceRef]) -> Tuple[str, List[SourceRef], List[int]]:
    """Returns (cleaned_answer, cited_sources_in_first_use_order, invalid_indices).

    A marker such as "[2, 9]" keeps the valid 2 and drops the 9; a marker with no valid index is
    removed entirely. The model can therefore never make a citation appear that was not in its
    context."""
    by_index = {s.index: s for s in sources}
    cited: List[SourceRef] = []
    seen: Set[int] = set()
    invalid: List[int] = []

    def _rewrite(match: "re.Match[str]") -> str:
        keep: List[int] = []
        for idx in _indices_in(match.group(1)):
            if idx in by_index:
                keep.append(idx)
                if idx not in seen:
                    seen.add(idx)
                    cited.append(by_index[idx])
            elif idx not in invalid:
                invalid.append(idx)
        return "[" + ", ".join(str(i) for i in keep) + "]" if keep else ""

    cleaned = _CITE_GROUP.sub(_rewrite, answer)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, cited, invalid


@dataclass(frozen=True)
class Claim:
    text: str                  # sentence with citation markers removed
    cited: Tuple[int, ...]     # 1-based source numbers cited in that sentence


def split_claims(answer: str) -> List[Claim]:
    claims: List[Claim] = []
    for sentence in _SENTENCE_END.split(answer.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        cited: List[int] = []
        for group in _CITE_GROUP.findall(sentence):
            cited.extend(i for i in _indices_in(group) if i not in cited)
        text = re.sub(r"\s+", " ", _CITE_GROUP.sub("", sentence)).strip()
        text = re.sub(r"\s+([.,;:!?])", lambda m: m.group(1), text)
        if text:
            claims.append(Claim(text=text, cited=tuple(cited)))
    return claims


class EntailmentChecker(Protocol):
    def check_entailment(self, premise: str, hypothesis: str, threshold: Optional[float] = None) -> bool: ...


@dataclass(frozen=True)
class ClaimVerdict:
    claim: str
    cited: Tuple[int, ...]
    status: str                # "supported" | "unsupported" | "uncited"

    def to_public(self) -> Dict[str, Any]:
        return {"claim": self.claim, "cited": list(self.cited), "status": self.status}


@dataclass
class VerificationReport:
    verdicts: List[ClaimVerdict] = field(default_factory=list)

    @property
    def support_rate(self) -> Optional[float]:
        """Share of claims whose cited source entails them; None if there were no claims."""
        if not self.verdicts:
            return None
        return sum(v.status == "supported" for v in self.verdicts) / len(self.verdicts)

    def to_public(self) -> Dict[str, Any]:
        return {"support_rate": self.support_rate, "claims": [v.to_public() for v in self.verdicts]}


def verify_claims(answer: str, sources: Sequence[SourceRef], checker: EntailmentChecker,
                  threshold: Optional[float] = None) -> VerificationReport:
    """Per-claim support. A claim is `supported` only if at least one source it CITES entails it;
    a claim with no citation is `uncited` (never counted as supported); a cited claim whose cited
    sources do not entail it is `unsupported`. Conservative on purpose: an unverifiable claim
    must not be reported as faithful."""
    by_index = {s.index: s for s in sources}
    report = VerificationReport()
    for claim in split_claims(answer):
        valid = [by_index[i] for i in claim.cited if i in by_index]
        if not valid:
            report.verdicts.append(ClaimVerdict(claim.text, claim.cited, "uncited"))
            continue
        supported = any(checker.check_entailment(s.text, claim.text, threshold) for s in valid)
        report.verdicts.append(ClaimVerdict(claim.text, claim.cited, "supported" if supported else "unsupported"))
    return report
