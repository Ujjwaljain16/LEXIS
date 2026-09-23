"""One-off generator for notebooks/cuad_doc_scoped_dev_ablation.ipynb.
Run locally, commit the generated .ipynb, do not re-run casually (it
overwrites the notebook from scratch)."""
import json

PINNED_COMMIT = "f9588cf08f7b513962b20a4340e8f88b6ab70c50"

def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

cells = []

cells.append(md_cell(f"""# LEXIS — dev-set ablation experiments (R1 / R2 / R4)

**Purpose**: measure the retrieval-improvement ladder's rungs against the 10-contract DEV set,
paired against the existing dev reference (`evaluation/reports/cuad_doc_scoped_dev.json`). This
notebook is explicitly **NOT** the frozen-test-baseline notebook
(`cuad_doc_scoped_test_baseline.ipynb`) — it never touches the 164-contract test split, never
freezes anything, and its whole point is to tune/compare configurations on dev data, which that
other notebook explicitly promises not to do.

**Why this runs on Colab and not locally**: the local dev machine (16GB RAM) segfaulted 4/4 times
chunking one of the 10 dev contracts under this session's memory pressure (exit code 139 —
confirmed real crash, not a hang). GPU + more headroom sidesteps it, matching how the frozen-test
notebook's Stage A/B already work around the same class of local constraint.

**Pinned commit**: `{PINNED_COMMIT}` on `foundation-remediation`. R1 (BM25 gets the same
contextual-chunk-prefix dense embeddings already had) is now unconditional in this codebase, so
every pass below already reflects it — there is no separate "R1 on/off" flag to toggle. R2 (HyPE)
and R4 (rerank) are each a single settings flag (`HYPE_ENABLED`, `RERANK_ENABLED`), measured
independently against the same freshly-ingested collection so each rung's effect is isolated, not
cumulative.

**A known caveat for R2 (HyPE)**: local testing earlier today hit Gemini's free-tier quota hard
enough (hundreds of calls across repeated smoke tests) that it may still be exhausted — this
notebook paces calls at 15/min with retry-and-backoff, but if the underlying daily quota is gone,
retries won't help until it resets. If the HyPE ingest cell fails or produces very few questions,
that's an expected, already-understood limitation, not a new bug — see the R2 section for what to
check.

**Sequence**: Setup → ingest the dev set once (with HyPE on, so R2's index exists for later) →
three isolated eval passes (baseline / +HyPE / +rerank) → paired stats against the dev reference →
download the results. Nothing here is frozen; results are for review, not for gating anything."""))

cells.append(md_cell("## 0. Setup"))

cells.append(code_cell("""from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_DIR = "/content/drive/MyDrive/lexis_dev_ablation_run"
os.makedirs(DRIVE_DIR, exist_ok=True)
print("Persistent run directory:", DRIVE_DIR)"""))

cells.append(code_cell(f"""PINNED_COMMIT = "{PINNED_COMMIT}"

import os
if not os.path.isdir("/content/LEXIS"):
    !git clone https://github.com/Ujjwaljain16/LEXIS.git /content/LEXIS
%cd /content/LEXIS
!git fetch origin
!git checkout {{PINNED_COMMIT}}
!pip install -q -e .

checked_out = !git rev-parse HEAD
checked_out = checked_out[0].strip()
assert checked_out == PINNED_COMMIT, f"checked out {{checked_out}}, expected {{PINNED_COMMIT}}"
print("Checked out and verified pinned commit:", checked_out)"""))

cells.append(code_cell("""import subprocess, sys

check = subprocess.run([sys.executable, "-c", "import lexis"], capture_output=True, text=True)
if check.returncode != 0:
    raise RuntimeError(
        "`import lexis` failed even in a FRESH subprocess -- the install itself is broken, not "
        "just kernel caching. Re-run `!pip install -e . -v` (drop -q) and inspect the real output.\\n"
        f"Subprocess stderr:\\n{check.stderr}"
    )
try:
    import lexis  # noqa: F401
    print("lexis is importable in THIS kernel -- proceed to the next cell.")
except ModuleNotFoundError:
    raise RuntimeError(
        "lexis installed correctly (a fresh subprocess CAN import it) but this notebook's own "
        "kernel cannot see it yet -- the standard Colab gotcha after pip-installing a new package. "
        "Fix: Runtime -> Restart session, then re-run every cell from the top."
    ) from None"""))

cells.append(code_cell("""# Credentials -- prefer Colab's Secrets manager (key icon in the left sidebar).
# Needs QDRANT_URL, QDRANT_API_KEY, POSTGRES_URL, and GEMINI_API_KEY (the last one only matters
# for the R2/HyPE section below -- the R1/baseline and R4/rerank passes never call it).
import os
from google.colab import userdata

os.environ["QDRANT_URL"] = userdata.get("QDRANT_URL")
os.environ["QDRANT_API_KEY"] = userdata.get("QDRANT_API_KEY")
os.environ["POSTGRES_URL"] = userdata.get("POSTGRES_URL")
os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
print("Credentials loaded from Colab Secrets (values not printed).")"""))

cells.append(code_cell("""import subprocess, sys

print("Python:", sys.version)
print("Git SHA:", subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
for pkg in ["sentence-transformers", "bm25s", "qdrant-client", "torch", "litellm"]:
    v = subprocess.run([sys.executable, "-m", "pip", "show", pkg], capture_output=True, text=True).stdout
    line = next((l for l in v.splitlines() if l.startswith("Version:")), "Version: (not found)")
    print(f"{pkg}: {line}")

import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))"""))

cells.append(code_cell("""from huggingface_hub import hf_hub_download

cuad_path = hf_hub_download(
    repo_id="theatticusproject/cuad",
    repo_type="dataset",
    filename="CUAD_v1/CUAD_v1.json",
)
print("CUAD dataset at:", cuad_path)"""))

cells.append(md_cell("""## 1. Ingest the 10-contract dev set once, with HyPE on

Same 10 contracts as the existing dev reference (`--num-contracts 10`, deterministic
title-sorted selection). `HYPE_ENABLED=true` here so the hype_questions collection gets
populated -- later eval passes can then toggle `HYPE_ENABLED` at query time independently of
whether ingestion wrote hype data, since retrieval only *searches* that collection when the flag
is on.

**This will take a while** -- HyPE paces itself to 15 requests/minute across roughly 500+ chunks,
so expect ~35+ minutes for this cell alone, on top of normal embedding/chunking time. That's
expected, not stuck; the point is respecting the API quota properly instead of failing most calls
the way local testing did earlier."""))

cells.append(code_cell("""import os
os.environ["QDRANT_COLLECTION_PRIMARY"] = "chunks_primary_colab_dev_ablation"
os.environ["BM25_INDEX_DIR"] = "/content/LEXIS/data/bm25_index_colab_dev_ablation"
os.environ["HYPE_ENABLED"] = "true"
INGEST_CHECKPOINT = f"{DRIVE_DIR}/ingest_checkpoint.json"
INGEST_THROWAWAY_OUTPUT = f"{DRIVE_DIR}/ingest_pass_throwaway.json"  # numbers from THIS pass are not used

!python scripts/setup_collections.py"""))

cells.append(code_cell("""# Resumable via the checkpoint: if this cell is interrupted (HyPE's rate limiting makes it long),
# just re-run it -- already-ingested contracts are skipped.
!python -m lexis.evaluation.run_eval \\
  --benchmark cuad \\
  --cuad-path {cuad_path} \\
  --num-contracts 10 \\
  --max-questions 50 \\
  --top-k 30 \\
  --protocol doc_scoped \\
  --ingest-checkpoint {INGEST_CHECKPOINT} \\
  --skip-diagnostics \\
  --output {INGEST_THROWAWAY_OUTPUT}

print("Ingest pass complete. Its own eval numbers are not used -- see the isolated passes below.")"""))

cells.append(md_cell("""## 2. Three isolated eval passes on the same ingested collection

`--skip-ingest` on all three (data is already written above) -- each just re-embeds the 50 dev
queries and searches, with a different flag combination. The baseline pass is run twice (settle
pass, same methodology as the frozen-test notebook: the very first post-ingest query pass can
differ from later ones) and the second run is what gets compared against."""))

cells.append(code_cell("""os.environ["HYPE_ENABLED"] = "false"
os.environ["RERANK_ENABLED"] = "false"
BASELINE_OUTPUT = f"{DRIVE_DIR}/dev_baseline.json"

# First pass (settle -- not used directly, see the cell below)
!python -m lexis.evaluation.run_eval \\
  --benchmark cuad \\
  --cuad-path {cuad_path} \\
  --num-contracts 10 \\
  --max-questions 50 \\
  --top-k 30 \\
  --protocol doc_scoped \\
  --skip-ingest \\
  --skip-diagnostics \\
  --output {BASELINE_OUTPUT}"""))

cells.append(code_cell("""# Settle pass -- THIS is the baseline result actually used below.
!python -m lexis.evaluation.run_eval \\
  --benchmark cuad \\
  --cuad-path {cuad_path} \\
  --num-contracts 10 \\
  --max-questions 50 \\
  --top-k 30 \\
  --protocol doc_scoped \\
  --skip-ingest \\
  --skip-diagnostics \\
  --output {BASELINE_OUTPUT}

print("Baseline (settled) written to", BASELINE_OUTPUT)"""))

cells.append(code_cell("""os.environ["HYPE_ENABLED"] = "true"
os.environ["RERANK_ENABLED"] = "false"
R2_OUTPUT = f"{DRIVE_DIR}/dev_r2_hype.json"

!python -m lexis.evaluation.run_eval \\
  --benchmark cuad \\
  --cuad-path {cuad_path} \\
  --num-contracts 10 \\
  --max-questions 50 \\
  --top-k 30 \\
  --protocol doc_scoped \\
  --skip-ingest \\
  --skip-diagnostics \\
  --output {R2_OUTPUT}

print("R2 (HyPE) pass written to", R2_OUTPUT)"""))

cells.append(code_cell("""os.environ["HYPE_ENABLED"] = "false"
os.environ["RERANK_ENABLED"] = "true"
R4_OUTPUT = f"{DRIVE_DIR}/dev_r4_rerank.json"

!python -m lexis.evaluation.run_eval \\
  --benchmark cuad \\
  --cuad-path {cuad_path} \\
  --num-contracts 10 \\
  --max-questions 50 \\
  --top-k 30 \\
  --protocol doc_scoped \\
  --skip-ingest \\
  --skip-diagnostics \\
  --output {R4_OUTPUT}

print("R4 (rerank) pass written to", R4_OUTPUT)"""))

cells.append(md_cell("""## 3. Paired comparison against the existing dev reference

`evaluation/reports/cuad_doc_scoped_dev.json` is the pre-R1-fix dev reference already in the
repo. Each pass above is compared against it via `evaluation/stats.py::paired_comparison`
(bootstrap CI + permutation test on per-case Recall@30/reciprocal-rank, matched by case_id) --
the same rigor used for every other ablation claim in this project. A gain only counts if
`claim_supported()` says so (CI excludes zero AND p < alpha after Holm correction across the two
comparisons here)."""))

cells.append(code_cell("""import json
from lexis.evaluation.stats import paired_comparison, claim_supported, holm_correction

reference = json.load(open("evaluation/reports/cuad_doc_scoped_dev.json"))["per_case"]
reference_by_id = {c["case_id"]: c for c in reference}

def compare(label, output_path):
    pass_cases = json.load(open(output_path))["per_case"]
    pass_by_id = {c["case_id"]: c for c in pass_cases}
    common = sorted(set(reference_by_id) & set(pass_by_id))
    print(f"--- {label} --- (n={len(common)}, reference n={len(reference)}, pass n={len(pass_cases)})")

    recall_a = [pass_by_id[i]["recall_at_k"] for i in common]
    recall_b = [reference_by_id[i]["recall_at_k"] for i in common]
    rr_a = [pass_by_id[i]["reciprocal_rank"] for i in common]
    rr_b = [reference_by_id[i]["reciprocal_rank"] for i in common]

    recall_result = paired_comparison(recall_a, recall_b, seed=0)
    rr_result = paired_comparison(rr_a, rr_b, seed=0)
    p_adj = holm_correction([recall_result.p_value, rr_result.p_value])

    print("Recall@30: %.4f vs %.4f  diff=%+.4f  CI[%.4f,%.4f]  p=%.4f  p_holm=%.4f  supported=%s" % (
        recall_result.mean_a, recall_result.mean_b, recall_result.mean_diff,
        recall_result.ci_lo, recall_result.ci_hi, recall_result.p_value, p_adj[0], claim_supported(recall_result)))
    print("MRR      : %.4f vs %.4f  diff=%+.4f  CI[%.4f,%.4f]  p=%.4f  p_holm=%.4f  supported=%s" % (
        rr_result.mean_a, rr_result.mean_b, rr_result.mean_diff,
        rr_result.ci_lo, rr_result.ci_hi, rr_result.p_value, p_adj[1], claim_supported(rr_result)))
    print()
    return recall_result, rr_result

baseline_results = compare("Baseline (R1, current code, no HyPE/rerank)", BASELINE_OUTPUT)
r2_results = compare("R2 (HyPE)", R2_OUTPUT)
r4_results = compare("R4 (rerank)", R4_OUTPUT)"""))

cells.append(md_cell("""## 4. Download results

Nothing here is frozen -- these are dev-set exploratory numbers to report and discuss, not
artifacts that gate anything. Copy whichever ones are useful back into the local repo's
`evaluation/reports/` directory yourself (that path is gitignored, same as every other report)."""))

cells.append(code_cell("""from google.colab import files

for path in [BASELINE_OUTPUT, R2_OUTPUT, R4_OUTPUT]:
    files.download(path)"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("notebooks/cuad_doc_scoped_dev_ablation.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Wrote notebooks/cuad_doc_scoped_dev_ablation.ipynb with", len(cells), "cells")
