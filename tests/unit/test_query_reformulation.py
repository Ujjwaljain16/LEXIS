"""
Tests for the generic, deterministic query-reformulation transformations
(evaluation/query_reformulation.py). All content is generic/invented
(memos, reports, made-up instructions), never CUAD/legal-specific, since
these transformations must generalize to any templated query set.
"""
import inspect

from lexis.evaluation.query_reformulation import (
    BoilerplateModel,
    NgramBoilerplateModel,
    fit_boilerplate_model,
    fit_ngram_boilerplate_model,
    normalize_query,
    reduce_boilerplate,
    reduce_ngram_boilerplate,
    split_sentences,
)


# --- normalize_query (Representation B) ---

def test_normalize_query_collapses_whitespace():
    assert normalize_query("What   is\n\nthe term?") == "What is the term?"


def test_normalize_query_empty_string():
    assert normalize_query("") == ""


def test_normalize_query_whitespace_only():
    assert normalize_query("   \n\t  ") == ""


def test_normalize_query_already_normalized_is_unchanged():
    q = "What is the notice period?"
    assert normalize_query(q) == q


def test_normalize_query_is_deterministic():
    q = "Some   query   text."
    assert normalize_query(q) == normalize_query(q)


# --- split_sentences ---

def test_split_sentences_empty_string():
    assert split_sentences("") == []


def test_split_sentences_whitespace_only():
    assert split_sentences("   ") == []


def test_split_sentences_single_sentence_no_split():
    assert split_sentences("What is the term length") == ["What is the term length"]


def test_split_sentences_punctuation_boundaries():
    result = split_sentences("Please review the attached memo. What is the deadline?")
    assert result == ["Please review the attached memo.", "What is the deadline?"]


def test_split_sentences_short_query_is_one_sentence():
    assert split_sentences("Deadline?") == ["Deadline?"]


# --- fit_boilerplate_model / reduce_boilerplate (Representation C) ---

PREAMBLE = "Please review the attached document carefully."


def make_templated_queries(n, detail_prefix="What is item"):
    """n queries sharing an identical preamble sentence but each with a
    distinct second sentence -- mirrors a templated query set generically,
    without referencing any real benchmark's wording."""
    return [f"{PREAMBLE} {detail_prefix} {i}?" for i in range(n)]


def test_fit_boilerplate_model_flags_sentence_present_in_all_queries():
    queries = make_templated_queries(10)
    model = fit_boilerplate_model(queries, min_frequency_ratio=0.5)
    assert model.is_boilerplate(PREAMBLE) is True


def test_fit_boilerplate_model_does_not_flag_sentence_unique_to_one_query():
    queries = make_templated_queries(10)
    model = fit_boilerplate_model(queries, min_frequency_ratio=0.5)
    assert model.is_boilerplate("What is item 3?") is False


def test_fit_boilerplate_model_threshold_is_configurable_not_hardcoded():
    queries = ["Alpha sentence. Beta sentence."] * 3 + ["Alpha sentence. Gamma sentence."] * 2
    # "Alpha sentence." appears in 5/5 = 1.0; "Beta sentence." in 3/5 = 0.6
    model_strict = fit_boilerplate_model(queries, min_frequency_ratio=0.9)
    assert model_strict.is_boilerplate("Alpha sentence.") is True
    assert model_strict.is_boilerplate("Beta sentence.") is False  # 0.6 < 0.9

    model_loose = fit_boilerplate_model(queries, min_frequency_ratio=0.5)
    assert model_loose.is_boilerplate("Beta sentence.") is True  # 0.6 >= 0.5


def test_fit_boilerplate_model_counts_repeated_sentence_within_one_query_once():
    queries = ["Repeat me. Repeat me. Unique content here."]
    model = fit_boilerplate_model(queries, min_frequency_ratio=0.5)
    assert model.sentence_frequency["Repeat me."] == 1  # one query, not two


def test_reduce_boilerplate_removes_common_sentence_keeps_specific_one():
    queries = make_templated_queries(10)
    model = fit_boilerplate_model(queries, min_frequency_ratio=0.5)
    result = reduce_boilerplate(queries[3], model)
    assert result == "What is item 3?"
    assert PREAMBLE not in result


def test_reduce_boilerplate_falls_back_to_original_if_everything_is_boilerplate():
    """No accidental deletion of meaningful tokens: if every sentence of a
    query is classified as boilerplate, the ORIGINAL query must be returned
    unchanged, never an empty string."""
    queries = [PREAMBLE, PREAMBLE, PREAMBLE]
    model = fit_boilerplate_model(queries, min_frequency_ratio=0.5)
    result = reduce_boilerplate(PREAMBLE, model)
    assert result == PREAMBLE
    assert result != ""


def test_reduce_boilerplate_empty_query_returns_empty_not_error():
    model = fit_boilerplate_model(["irrelevant"], min_frequency_ratio=0.5)
    assert reduce_boilerplate("", model) == ""


def test_reduce_boilerplate_single_sentence_query_with_empty_model():
    model = fit_boilerplate_model([], min_frequency_ratio=0.5)
    query = "What is the term length?"
    assert reduce_boilerplate(query, model) == query


def test_reduce_boilerplate_is_deterministic():
    queries = make_templated_queries(6)
    model = fit_boilerplate_model(queries, min_frequency_ratio=0.5)
    a = reduce_boilerplate(queries[2], model)
    b = reduce_boilerplate(queries[2], model)
    assert a == b


def test_reduce_boilerplate_preserves_order_of_kept_sentences():
    queries = [f"{PREAMBLE} First unique bit {i}. Second unique bit {i}." for i in range(6)]
    model = fit_boilerplate_model(queries, min_frequency_ratio=0.5)
    result = reduce_boilerplate(queries[0], model)
    assert result == "First unique bit 0. Second unique bit 0."


# --- fit_ngram_boilerplate_model / reduce_ngram_boilerplate (Representation C2) ---
# n-gram-span granularity, not whole-sentence: catches a fixed span that
# straddles a varying middle term, e.g. "... related to X that should ...",
# which whole-sentence matching can never detect.

def make_split_template_queries(n, varying_term_prefix="Category"):
    """Mirrors a generic template with a FIXED prefix, a VARYING middle
    term, and a FIXED suffix embedded in the SAME sentence -- generic
    content, not modeled on any real benchmark's exact wording."""
    return [
        f"Please identify the clause about {varying_term_prefix}{i} that requires legal review. "
        f"Question: what does it say?"
        for i in range(n)
    ]


def test_ngram_model_flags_fixed_prefix_span_around_varying_middle_term():
    queries = make_split_template_queries(10)
    model = fit_ngram_boilerplate_model(queries, ngram_length=4, min_frequency_ratio=0.5)
    # "Please identify the clause" recurs verbatim across all 10 queries
    assert model.is_boilerplate(("Please", "identify", "the", "clause")) is True


def test_ngram_model_does_not_flag_varying_middle_term():
    queries = make_split_template_queries(10)
    model = fit_ngram_boilerplate_model(queries, ngram_length=4, min_frequency_ratio=0.5)
    # a 4-gram containing the unique numbered category never recurs across queries
    assert model.is_boilerplate(("about", "Category3", "that", "requires")) is False


def test_reduce_ngram_boilerplate_removes_fixed_spans_keeps_varying_content():
    queries = make_split_template_queries(10)
    model = fit_ngram_boilerplate_model(queries, ngram_length=4, min_frequency_ratio=0.5)
    result = reduce_ngram_boilerplate(queries[3], model)
    assert "Category3" in result
    assert "Please identify the clause" not in result


def test_reduce_ngram_boilerplate_empty_query():
    model = fit_ngram_boilerplate_model(["irrelevant text here"], ngram_length=3, min_frequency_ratio=0.5)
    assert reduce_ngram_boilerplate("", model) == ""


def test_reduce_ngram_boilerplate_short_query_below_ngram_length_returned_unchanged():
    model = fit_ngram_boilerplate_model(["a longer sentence used only to fit the model"], ngram_length=6, min_frequency_ratio=0.5)
    short_query = "Two words"
    assert reduce_ngram_boilerplate(short_query, model) == short_query


def test_reduce_ngram_boilerplate_falls_back_to_original_if_everything_is_boilerplate():
    queries = ["Repeat this exact phrase everywhere"] * 5
    model = fit_ngram_boilerplate_model(queries, ngram_length=3, min_frequency_ratio=0.5)
    result = reduce_ngram_boilerplate("Repeat this exact phrase everywhere", model)
    assert result == "Repeat this exact phrase everywhere"
    assert result != ""


def test_reduce_ngram_boilerplate_is_deterministic():
    queries = make_split_template_queries(6)
    model = fit_ngram_boilerplate_model(queries, ngram_length=4, min_frequency_ratio=0.5)
    a = reduce_ngram_boilerplate(queries[2], model)
    b = reduce_ngram_boilerplate(queries[2], model)
    assert a == b


def test_ngram_model_threshold_is_configurable():
    queries = ["A B C D common tail"] * 3 + ["A B C D other tail"] * 2
    # "A B C D" appears in all 5; a very strict threshold still flags it, a
    # threshold above 1.0 (impossible) would not -- use two different real thresholds
    loose = fit_ngram_boilerplate_model(queries, ngram_length=4, min_frequency_ratio=0.5)
    strict = fit_ngram_boilerplate_model(queries, ngram_length=4, min_frequency_ratio=0.99)
    assert loose.is_boilerplate(("A", "B", "C", "D")) is True
    assert strict.is_boilerplate(("A", "B", "C", "D")) is True  # 5/5 = 1.0 clears even a strict bar
    assert strict.is_boilerplate(("B", "C", "D", "common")) is False  # only 3/5 = 0.6 < 0.99


# --- no leakage / no side effects (structural checks) ---

def test_module_has_no_ground_truth_or_retrieval_or_settings_dependency():
    import lexis.evaluation.query_reformulation as module
    source = inspect.getsource(module)
    import_lines = [line for line in source.splitlines() if line.strip().startswith(("import ", "from "))]
    forbidden_markers = ("hybrid_retriever", "RetrievalEngine", "map_answer_texts_to_chunk_ids", "config import settings")
    assert not any(any(marker in line for marker in forbidden_markers) for line in import_lines)


def test_boilerplate_functions_take_only_query_text_no_ground_truth_params():
    """Signature-level guard: fit_boilerplate_model/reduce_boilerplate must
    not accept anything resembling ground truth (answers, chunk ids,
    relevance labels)."""
    for func in (fit_boilerplate_model, reduce_boilerplate, fit_ngram_boilerplate_model, reduce_ngram_boilerplate, normalize_query, split_sentences):
        params = set(inspect.signature(func).parameters.keys())
        forbidden = {"answer_text", "answer_texts", "relevant_chunk_ids", "chunk", "chunks", "ground_truth", "labels"}
        assert not (params & forbidden), f"{func.__name__} accepts a forbidden parameter: {params & forbidden}"
