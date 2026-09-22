"""
CUAD (Contract Understanding Atticus Dataset) benchmark adapter.

This is the ONLY module in the evaluation package allowed to know about
CUAD's field names, question-template format, or SQuAD-2.0-style
is_impossible/answers structure. It normalizes CUAD into the generic
lexis.evaluation.types.BenchmarkCase shape; nothing downstream (harness.py,
metrics.py) ever sees a CUAD-specific field.

Source: the official CUAD v1 release, distributed on the Hugging Face Hub
as theatticusproject/cuad (file CUAD_v1/CUAD_v1.json), in the original
SQuAD-2.0-style layout:
    {"version": "...", "data": [
        {"title": <contract name>, "paragraphs": [
            {"context": <full contract text>, "qas": [
                {"id": ..., "question": ..., "is_impossible": bool,
                 "answers": [{"text": ..., "answer_start": int}, ...]}
            ]}
        ]}
    ]}
Each contract has exactly one paragraph (the whole contract as one context
string) and 41 questions (one per CUAD clause category); questions whose
category does not apply to that contract have is_impossible=True and no
answers.
"""
from typing import Any, Dict, List, Tuple

from lexis.evaluation.dataset.mapping import deterministic_document_id, map_answer_texts_to_chunk_ids
from lexis.evaluation.types import BenchmarkCase
from lexis.indexing.schema import Chunk

BENCHMARK_NAME = "cuad"


class CUADLoader:
    """Loads and validates the raw CUAD v1 JSON. Performs no sampling, no
    chunk mapping, and no question filtering -- purely "is this file
    actually shaped like CUAD v1"."""

    def load(self, path: str) -> Dict[str, Any]:
        import json

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self._validate(raw, path)
        return raw

    def _validate(self, raw: Any, path: str) -> None:
        if not isinstance(raw, dict) or "data" not in raw:
            raise ValueError(
                f"'{path}' does not look like a CUAD v1 file: expected a top-level "
                f"object with a 'data' key, got {type(raw).__name__}."
            )
        if not isinstance(raw["data"], list) or not raw["data"]:
            raise ValueError(f"'{path}': 'data' must be a non-empty list of contracts.")

        for i, contract in enumerate(raw["data"]):
            if not isinstance(contract, dict) or "title" not in contract or "paragraphs" not in contract:
                raise ValueError(
                    f"'{path}': data[{i}] is missing required 'title'/'paragraphs' fields."
                )
            if not isinstance(contract["paragraphs"], list) or not contract["paragraphs"]:
                raise ValueError(f"'{path}': data[{i}] ('{contract.get('title')}') has no paragraphs.")
            for j, para in enumerate(contract["paragraphs"]):
                if "context" not in para or "qas" not in para:
                    raise ValueError(
                        f"'{path}': data[{i}].paragraphs[{j}] is missing 'context'/'qas'."
                    )
                for k, qa in enumerate(para["qas"]):
                    required = {"id", "question", "answers", "is_impossible"}
                    missing = required - qa.keys()
                    if missing:
                        raise ValueError(
                            f"'{path}': data[{i}].paragraphs[{j}].qas[{k}] is missing field(s) {missing}."
                        )


def select_contracts(raw: Dict[str, Any], num_contracts: int) -> List[Dict[str, Any]]:
    """Deterministically selects the first `num_contracts` contracts, ordered
    by title. Sorting (rather than relying on file order) makes the subset
    reproducible even if the source file's row order ever changes, without
    needing a seed or a stored list of chosen titles."""
    if num_contracts <= 0:
        raise ValueError("num_contracts must be positive.")
    contracts = sorted(raw["data"], key=lambda c: c["title"])
    return contracts[:num_contracts]


def _answerable_questions(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    """CUAD marks a (contract, category) pair with is_impossible=True and an
    empty answers list when that clause category does not apply to this
    contract. Such pairs have no ground truth to retrieve and are excluded
    from the benchmark case set entirely -- they are not converted into
    "expect nothing" cases, since there is nothing meaningful to score."""
    qas = contract["paragraphs"][0]["qas"]
    answerable = [qa for qa in qas if not qa.get("is_impossible") and qa.get("answers")]
    return sorted(answerable, key=lambda qa: qa["id"])


class CUADAdapter:
    """Converts raw, validated CUAD data into generic BenchmarkCase objects.

    Ground truth here is chunk-level: for each answerable question, CUAD
    gives one or more exact answer text spans within that contract's full
    text (SQuAD-2.0-style answer_start offsets). This adapter determines
    which of LEXIS's *actual, already-produced* chunks for that contract
    contain each answer text (see dataset/mapping.py), rather than assuming
    any particular chunk boundary. A question is only included in the
    output if at least one of its answer texts was found inside at least
    one real chunk; unmappable questions are reported separately rather
    than silently dropped or scored as "not relevant".
    """

    def build_cases(
        self,
        raw: Dict[str, Any],
        chunks_by_doc_id: Dict[str, List[Chunk]],
        num_contracts: int,
        max_total_questions: int,
    ) -> Tuple[List[BenchmarkCase], List[Dict[str, Any]]]:
        if max_total_questions <= 0:
            raise ValueError("max_total_questions must be positive.")

        contracts = select_contracts(raw, num_contracts)
        cases: List[BenchmarkCase] = []
        unmapped: List[Dict[str, Any]] = []

        for contract in contracts:
            title = contract["title"]
            doc_id = deterministic_document_id(title, prefix=BENCHMARK_NAME)

            if doc_id not in chunks_by_doc_id:
                raise ValueError(
                    f"No chunks were provided for contract '{title}' (doc_id={doc_id}). "
                    f"Every contract selected by select_contracts() must be chunked before "
                    f"calling build_cases() -- this adapter does not perform chunking itself."
                )
            chunks = chunks_by_doc_id[doc_id]

            for qa in _answerable_questions(contract):
                if len(cases) >= max_total_questions:
                    return cases, unmapped

                answer_texts = [a["text"] for a in qa["answers"]]
                matched_chunk_ids = map_answer_texts_to_chunk_ids(answer_texts, chunks)

                if not matched_chunk_ids:
                    unmapped.append({
                        "case_id": qa["id"],
                        "doc_id": doc_id,
                        "contract_title": title,
                        "reason": "no_chunk_contained_any_answer_text",
                        "answer_texts": answer_texts,
                    })
                    continue

                cases.append(BenchmarkCase(
                    case_id=qa["id"],
                    query=qa["question"],
                    relevant_chunk_ids=frozenset(matched_chunk_ids),
                    relevant_doc_ids=frozenset({doc_id}),
                    source_benchmark=BENCHMARK_NAME,
                    metadata={
                        "contract_title": title,
                        "answer_texts": answer_texts,
                    },
                ))

        return cases, unmapped
