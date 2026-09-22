"""
Tests for the CUAD benchmark adapter (evaluation/dataset/cuad_loader.py).

Fixture data below is structurally CUAD-shaped (title/paragraphs/context/qas
with is_impossible + answer_start, matching the real CUAD v1 SQuAD-2.0-style
release) but uses short, invented, non-legal sentences rather than real
contract text -- there is no need to embed real CUAD content to test the
adapter's structural/logical behaviour.
"""
import json

import pytest

from lexis.evaluation.dataset.cuad_loader import CUADAdapter, CUADLoader, select_contracts
from lexis.evaluation.dataset.mapping import deterministic_document_id
from lexis.indexing.schema import Chunk, ChunkMetadata

META = ChunkMetadata(source_file="x.txt", page_num=1, document_type="unknown")


def make_chunk(doc_id, idx, text):
    return Chunk.create(doc_id=doc_id, split_idx=idx, raw_content=text, metadata=META)


def make_qa(qa_id, question, answer_text=None, answer_start=0, impossible=False):
    return {
        "id": qa_id,
        "question": question,
        "is_impossible": impossible,
        "answers": [] if impossible or answer_text is None else [{"text": answer_text, "answer_start": answer_start}],
    }


def make_contract(title, context, qas):
    return {"title": title, "paragraphs": [{"context": context, "qas": qas}]}


def valid_raw_dataset():
    contract_b = make_contract(
        "Beta Report",
        "The renewal window is thirty days from notice.",
        [
            make_qa("Beta Report__Renewal", "What is the renewal window?", "renewal window is thirty days", 4),
            make_qa("Beta Report__Termination", "What are termination terms?", impossible=True),
        ],
    )
    contract_a = make_contract(
        "Alpha Policy",
        "Staffing must be reviewed quarterly by the safety office.",
        [
            make_qa("Alpha Policy__Staffing", "What are the staffing requirements?", "reviewed quarterly by the safety office", 19),
        ],
    )
    return {"version": "test-1.0", "data": [contract_b, contract_a]}  # deliberately unsorted


# --- CUADLoader validation ---

def test_loader_accepts_a_well_formed_file(tmp_path):
    path = tmp_path / "cuad.json"
    path.write_text(json.dumps(valid_raw_dataset()), encoding="utf-8")
    raw = CUADLoader().load(str(path))
    assert len(raw["data"]) == 2


def test_loader_rejects_missing_data_key(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
    with pytest.raises(ValueError, match="data"):
        CUADLoader().load(str(path))


def test_loader_rejects_empty_data_list(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": "1.0", "data": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        CUADLoader().load(str(path))


def test_loader_rejects_contract_missing_title(tmp_path):
    path = tmp_path / "bad.json"
    bad = {"version": "1.0", "data": [{"paragraphs": []}]}
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="title"):
        CUADLoader().load(str(path))


def test_loader_rejects_qa_missing_required_fields(tmp_path):
    path = tmp_path / "bad.json"
    bad = {
        "version": "1.0",
        "data": [{
            "title": "X",
            "paragraphs": [{"context": "text", "qas": [{"id": "x"}]}],  # missing question/answers/is_impossible
        }],
    }
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        CUADLoader().load(str(path))


# --- select_contracts: deterministic ordering ---

def test_select_contracts_orders_by_title_regardless_of_file_order():
    raw = valid_raw_dataset()  # Beta Report listed before Alpha Policy in the file
    selected = select_contracts(raw, num_contracts=2)
    assert [c["title"] for c in selected] == ["Alpha Policy", "Beta Report"]


def test_select_contracts_respects_the_limit():
    raw = valid_raw_dataset()
    selected = select_contracts(raw, num_contracts=1)
    assert len(selected) == 1
    assert selected[0]["title"] == "Alpha Policy"


def test_select_contracts_rejects_non_positive_count():
    with pytest.raises(ValueError):
        select_contracts(valid_raw_dataset(), num_contracts=0)


# --- CUADAdapter.build_cases ---

def _chunks_for(raw):
    alpha_id = deterministic_document_id("Alpha Policy", prefix="cuad")
    beta_id = deterministic_document_id("Beta Report", prefix="cuad")
    return {
        alpha_id: [make_chunk(alpha_id, 0, "Staffing must be reviewed quarterly by the safety office.")],
        beta_id: [make_chunk(beta_id, 0, "The renewal window is thirty days from notice.")],
    }


def test_build_cases_produces_generic_cases_with_correct_identity():
    raw = valid_raw_dataset()
    chunks_by_doc_id = _chunks_for(raw)

    cases, unmapped = CUADAdapter().build_cases(raw, chunks_by_doc_id, num_contracts=2, max_total_questions=50)

    assert unmapped == []
    assert len(cases) == 2  # the is_impossible Termination question is excluded
    case_ids = {c.case_id for c in cases}
    assert case_ids == {"Alpha Policy__Staffing", "Beta Report__Renewal"}

    staffing_case = next(c for c in cases if c.case_id == "Alpha Policy__Staffing")
    expected_doc_id = deterministic_document_id("Alpha Policy", prefix="cuad")
    assert staffing_case.relevant_doc_ids == frozenset({expected_doc_id})
    assert staffing_case.relevant_chunk_ids == frozenset({chunks_by_doc_id[expected_doc_id][0].chunk_id})
    assert staffing_case.source_benchmark == "cuad"


def test_cuad_specific_fields_do_not_leak_outside_metadata():
    raw = valid_raw_dataset()
    cases, _ = CUADAdapter().build_cases(raw, _chunks_for(raw), num_contracts=2, max_total_questions=50)
    case = cases[0]
    # Only generic BenchmarkCase fields exist at the top level.
    assert set(vars(case).keys()) == {
        "case_id", "query", "relevant_chunk_ids", "relevant_doc_ids", "source_benchmark", "metadata",
    }
    # CUAD-specific details are confined to metadata, not promoted to real fields.
    assert "contract_title" in case.metadata
    assert not hasattr(case, "is_impossible")
    assert not hasattr(case, "category")


def test_impossible_questions_are_excluded_not_converted_to_empty_cases():
    raw = valid_raw_dataset()
    cases, unmapped = CUADAdapter().build_cases(raw, _chunks_for(raw), num_contracts=2, max_total_questions=50)
    all_ids = {c.case_id for c in cases} | {u["case_id"] for u in unmapped}
    assert "Beta Report__Termination" not in all_ids


def test_unmappable_ground_truth_is_reported_not_silently_dropped_or_faked():
    raw = valid_raw_dataset()
    chunks_by_doc_id = _chunks_for(raw)
    # Corrupt the Alpha chunk so it no longer contains the real answer text.
    alpha_id = deterministic_document_id("Alpha Policy", prefix="cuad")
    chunks_by_doc_id[alpha_id] = [make_chunk(alpha_id, 0, "This chunk text has nothing to do with the answer.")]

    cases, unmapped = CUADAdapter().build_cases(raw, chunks_by_doc_id, num_contracts=2, max_total_questions=50)

    case_ids = {c.case_id for c in cases}
    assert "Alpha Policy__Staffing" not in case_ids
    unmapped_ids = {u["case_id"] for u in unmapped}
    assert "Alpha Policy__Staffing" in unmapped_ids
    entry = next(u for u in unmapped if u["case_id"] == "Alpha Policy__Staffing")
    assert entry["reason"] == "no_chunk_contained_any_answer_text"


def test_missing_chunks_for_a_selected_contract_raises_instead_of_skipping():
    raw = valid_raw_dataset()
    incomplete_chunks = {deterministic_document_id("Alpha Policy", prefix="cuad"): [make_chunk("x", 0, "text")]}
    with pytest.raises(ValueError, match="Beta Report"):
        CUADAdapter().build_cases(raw, incomplete_chunks, num_contracts=2, max_total_questions=50)


def test_max_total_questions_is_respected():
    raw = valid_raw_dataset()
    cases, _ = CUADAdapter().build_cases(raw, _chunks_for(raw), num_contracts=2, max_total_questions=1)
    assert len(cases) == 1


def test_max_total_questions_must_be_positive():
    with pytest.raises(ValueError):
        CUADAdapter().build_cases(valid_raw_dataset(), _chunks_for(valid_raw_dataset()), num_contracts=2, max_total_questions=0)
