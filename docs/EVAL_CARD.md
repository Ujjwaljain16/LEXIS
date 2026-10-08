# LEXIS evaluation and system card

What this system is, what was measured, how, and what it must not be used for. Numbers here are copied from frozen,
re-verifiable artifacts (`python -m lexis.evaluation.verify_frozen` recomputes the headline from the per-case file).

## Intended use

Question answering over a **single, user-selected English-language commercial contract** at a time, returning an
answer with the exact source passages it cites, or declining when the passages do not contain the answer. It is a
research and engineering demonstration of a measured retrieval pipeline, **not legal advice** and not validated for
reliance in any legal matter.

## Out of scope / not validated

- Statutes, case law, non-English text, and non-US jurisdictions (packs for India and UK/EU exist as configuration only).
- Cross-document search over a whole corpus (measured and weak: see "Findings"; automatic document routing was 20%
  accurate and made results worse).
- Answer correctness. Retrieval is measured; **generated-answer quality (citation precision, abstention accuracy,
  faithfulness) has not been measured.**
- Multi-tenant data isolation, high availability, latency/throughput targets (none measured).

## System

Hybrid retrieval: BGE-M3 dense vectors (Qdrant) + BM25 (bm25s, local) over header-prefixed chunks, fused with
Reciprocal Rank Fusion (k=61). Optional cross-encoder rerank (`bge-reranker-v2-m3`, off by default). The generator
(Gemini via LiteLLM) sees numbered sources, cites them as `[n]`, and every `[n]` is validated against the sources it
was actually shown. Optional per-claim NLI check (off by default; accuracy on legal text unmeasured).
Model licences and dataset obligations: [`COMPLIANCE.md`](COMPLIANCE.md) (generated from the registry).

## Data

[CUAD v1](https://www.atticusprojectai.org/cuad) (510 SEC-filed commercial contracts, 41 clause categories).
Split by contract with a hash-based, group-disjoint manifest (`evaluation/splits/cuad_split_v1.json`):
dev 346 contracts / 4,669 answerable questions; test 164 contracts / 2,033 answerable questions (1,964 scored — the rest
could not be mapped to any produced chunk and are excluded and counted, not silently dropped). Ten contracts used during
early development are pinned to dev. The test split was run once, for the baseline, and frozen.

## Metrics

Recall@30 (share of gold answer chunks in the top 30), MRR (reciprocal rank of the first gold chunk), Hit@k (any gold
chunk in the top k). Ground truth: a chunk is relevant if it contains the annotated answer text. Uncertainty:
percentile bootstrap, 10,000 resamples over questions, seed 0. Comparisons: paired bootstrap CI on the difference plus
sign-flip permutation test, Holm-corrected across metrics; a gain is claimed only if both agree
(`lexis.evaluation.stats.claim_supported`).

## Results

Frozen held-out test, document-scoped protocol, hybrid retrieval, no rerank:

| Metric | Value | 95% CI |
|---|---|---|
| Recall@30 | 0.853 | [0.838, 0.867] |
| MRR | 0.523 | [0.505, 0.542] |
| Hit@1 / @5 / @10 / @30 | 0.404 / 0.665 / 0.763 / 0.910 | (descriptive; see README) |

Clause questions (n=1,415): Recall@30 0.923, MRR 0.617. Front-matter facts — title, date, parties (n=549):
Recall@30 0.671, MRR 0.281.

### Findings (dev, labelled as such)

- 89% of retrieval failures retrieved the right passage from the **wrong contract**.
- Restricting retrieval to the right contract: Recall@30 0.558 → 0.881, MRR 0.222 → 0.552 (p = 0.0008).
- Automatic document routing: 20% accurate, harmful versus scoping by the user.
- Cross-encoder rerank (60 dev contracts, 766 questions): MRR 0.501 → 0.548, +0.047, 95% CI [+0.021, +0.073], Holm
  p = 0.0022 (supported); Recall@30 +0.015, CI [-0.000, +0.031], Holm p = 0.059 (not supported).
- BM25 contextual prefix: MRR +0.026, CI [0.002, 0.058], Holm p = 0.16 (not established).
- Query-side boilerplate reduction: not significant (Holm p = 0.19); not claimed.

## Known weaknesses

- Header facts (document name, agreement date, parties) retrieve poorly: clause-oriented chunking is the wrong tool.
- ~3% of test questions cannot be mapped to a produced chunk (answer text split by chunking), so they are excluded.
- Only one benchmark, one domain, one language. CUAD contracts are SEC exhibits and may be over-represented in
  embedding-model training data.
- The document-scoped protocol assumes the user chooses the document.
- NLI threshold (0.5) is a conservative default, **not** a calibrated value.
- Provenance: an earlier version of the config redaction masked `*_tokens` settings, so config hashes from before that fix
  do not distinguish chunk sizes. The frozen artifact's chunk sizes are recorded in `run_config` and were 500/750.

## Re-verification

```bash
make repro                                   # tests, ratchet lint, recompute frozen headline, regenerate figure
python -m lexis.evaluation.verify_frozen     # just the artifact checks (14)
```

Re-running the benchmark itself needs Qdrant, a GPU and the pinned Colab notebooks in `notebooks/`.
