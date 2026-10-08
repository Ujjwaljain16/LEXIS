# What actually moved retrieval on legal text (and what didn't)

_Draft. Every number below is from a committed artifact in the [LEXIS repo](https://github.com/Ujjwaljain16/LEXIS);
`make repro` recomputes the headline result. Edit freely, but do not add a number that isn't in the repo._

I built a retrieval-augmented system for contract question answering and spent more time trying to disprove my own
improvements than making them. The most useful thing I learned is how little of "RAG quality" comes from the parts people
usually tune.

## The setup that made the numbers trustworthy

- **A frozen, contract-disjoint test split.** CUAD's 510 contracts are split by contract with a hash-based manifest:
  346 dev, 164 test. The test split was run exactly once, for the baseline, then frozen. Ten contracts I had looked at
  during development are pinned to dev — three of them would otherwise have leaked into test.
- **Intervals and paired tests on everything.** Bootstrap CIs over questions; for any "A beats B" claim, a paired
  bootstrap *and* a sign-flip permutation test with Holm correction. A gain counts only if both agree.
- **Provenance on every artifact:** git SHA, config hash, data hashes, package versions. A script re-derives the headline
  from the per-question file and fails if the README, the manifest, or the numbers drift.

Held-out result, hybrid dense + BM25 with reciprocal rank fusion: **Recall@30 0.853 [0.838, 0.867], MRR 0.523
[0.505, 0.542]** on 1,964 questions.

## What moved the needle: telling the system *which contract*

I classified every failure on a dev set. **89% of misses retrieved the right kind of passage from the wrong contract.**
Many CUAD contracts share boilerplate, so a "Governing Law" clause in one contract closely resembles the same clause
in others; searching the whole corpus becomes largely a contract-identification problem rather than a passage-ranking one.

Restricting retrieval to the user's selected contract moved Recall@30 from 0.558 to 0.881 and MRR from 0.222 to 0.552
(p = 0.0008). A case-level oracle reached MRR 0.907, which also corrected a hypothesis of mine that a ceiling above 0.60
wasn't reachable.

Then the obvious follow-up: can the system pick the contract itself? **No.** Automatic routing was 20% accurate and made
results worse than not routing. I kept it as a negative result rather than shipping it.

## What helped, a little, with the receipts

- **Cross-encoder reranking** (top-50 → reordered). On 766 dev questions: MRR 0.501 → 0.548, +0.047, 95% CI
  [+0.021, +0.073], Holm p = 0.002. It reordered the same candidates, so Recall@30 barely moved (+0.015, CI touching zero,
  p = 0.059 — not claimed). My first run of this on 50 questions showed the *same* +0.048 point estimate with a
  confidence interval of [-0.065, +0.156]. I'd have been equally "right" claiming it worked or didn't. The extra 700
  questions bought the answer, not a different effect.
- **Giving BM25 the same contextual header the dense model sees:** MRR +0.026, CI [0.002, 0.058], but Holm p = 0.16.
  Directionally positive, replicated on a second machine, not established. I say so in the README.

## What looked like a win and wasn't

Query-side boilerplate reduction raised Recall@30 by 0.057 on 50 questions — but only 5 questions changed, and the
permutation test gave p = 0.0625 (0.19 after correction). It's in the repo as "not significant", not as a feature.

## What's still weak

Facts that live in a contract's header — title, date, parties — are retrieved badly (Recall@30 0.67, MRR 0.28, versus
0.92 and 0.62 for clause questions). Clause-oriented chunking and semantic matching are the wrong tool for "what is this
agreement called?". It's the clearest next improvement and I haven't done it.

I also have **no measured answer-quality numbers** yet (citation precision, abstention accuracy, faithfulness). The
generation path validates every `[n]` citation against the passages the model was shown and abstains when they don't
contain the answer, but "the code does this" isn't "I measured it", and the README says so.

## Takeaways

1. Before tuning a retriever, classify the failures. Mine were 89% one kind, and it wasn't the kind I was tuning.
2. Freeze a test split early, touch it once, and let a script — not your memory — guard the numbers.
3. A confidence interval that spans zero is not a negative result; it's a request for more data. Get the data before
   you write the sentence.
4. Publish the negative results. The routing failure and the non-significant rungs are the most credible part of the repo.
