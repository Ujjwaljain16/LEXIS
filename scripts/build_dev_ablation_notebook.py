"""Generator for notebooks/cuad_doc_scoped_dev_ablation.ipynb (baseline + R4 rerank, dev set only).
Run locally, commit the generated .ipynb. It overwrites the notebook from scratch."""
import json

PINNED_COMMIT = "ad8b885e9844d968009706e09fc6845f1fb22bdf"


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = []

cells.append(md_cell(f"""# LEXIS — dev-set ablation (baseline vs R4 cross-encoder rerank)

**Run order: top to bottom, one click per cell. Nothing to edit.** Total time on a Colab T4: roughly 30-60 min.

Measures the current code's baseline and the R4 cross-encoder rerank on the 10-contract DEV set, each paired against
the existing dev reference (`evaluation/reports/cuad_doc_scoped_dev.json`). It never touches the 164-contract test
split and never freezes anything.

**Why Colab:** the 16 GB local machine segfaults (exit 139) chunking one large contract.

**What this run does NOT include, on purpose:** R2 (HyPE). It needs ~500 LLM calls to index even this small set and the
Gemini free-tier key allows 20 requests/day, so it cannot be measured honestly right now; including it only burns time.
No Gemini secret is needed.

**Pinned commit:** `{PINNED_COMMIT}`. R1 (BM25 gets the same contextual chunk prefix dense embeddings use) is
unconditional in this code, so the *baseline* pass already includes it; R4 is the single flag `RERANK_ENABLED`.

**State is fresh by design:** new Qdrant collection, new BM25 directory, and a new Drive checkpoint (`v2`). Do not reuse an
older checkpoint with a new VM: the checkpoint lives on Drive but the BM25 index lives on the VM disk, so a mismatch would
silently drop contracts from keyword search."""))

cells.append(md_cell("## 0. Setup"))

cells.append(code_cell("""from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_DIR = "/content/drive/MyDrive/lexis_dev_ablation_v2"
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
        "`import lexis` failed even in a FRESH subprocess -- the install itself is broken. "
        "Re-run `!pip install -e . -v` and inspect the output.\\n" + check.stderr
    )
try:
    import lexis  # noqa: F401
    print("lexis is importable in THIS kernel -- proceed.")
except ModuleNotFoundError:
    raise RuntimeError(
        "lexis installed, but this kernel cannot see it yet (standard Colab gotcha). "
        "Fix: Runtime -> Restart session, then re-run every cell from the top."
    ) from None"""))

cells.append(code_cell("""# Needs Colab Secrets: QDRANT_URL, QDRANT_API_KEY, POSTGRES_URL  (no Gemini key needed in this notebook)
import os
from google.colab import userdata

os.environ["QDRANT_URL"] = userdata.get("QDRANT_URL")
os.environ["QDRANT_API_KEY"] = userdata.get("QDRANT_API_KEY")
os.environ["POSTGRES_URL"] = userdata.get("POSTGRES_URL")
print("Credentials loaded from Colab Secrets (values not printed).")"""))

cells.append(code_cell("""# Fail fast, with the REAL error, if Qdrant is unreachable (bad secret, typo, suspended cluster).
import asyncio, os
from qdrant_client import AsyncQdrantClient

async def _check():
    client = AsyncQdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"] or None, timeout=30)
    return await client.get_collections()

try:
    res = await _check()
    print(f"Qdrant reachable: {len(res.collections)} existing collections. Safe to continue.")
except Exception as e:
    raise RuntimeError(
        f"Qdrant NOT reachable: {type(e).__name__}: {e!r}. "
        "Fix before continuing: (1) Colab Secrets QDRANT_URL must include https:// and the port, e.g. "
        "https://<id>.<region>.cloud.qdrant.io:6333 ; (2) QDRANT_API_KEY must be the cluster's key; "
        "(3) open cloud.qdrant.io and make sure the cluster is Running (free clusters are suspended after "
        "inactivity and must be resumed)."
    ) from None
"""))

cells.append(code_cell("""import subprocess, sys
import torch

print("Python:", sys.version.split()[0])
print("Git SHA:", subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU. Runtime -> Change runtime type -> T4 GPU, then restart and re-run from the top.")"""))

cells.append(code_cell("""from huggingface_hub import hf_hub_download

cuad_path = hf_hub_download(
    repo_id="theatticusproject/cuad",
    repo_type="dataset",
    filename="CUAD_v1/CUAD_v1.json",
)
print("CUAD dataset at:", cuad_path)"""))

cells.append(md_cell("""## 1. Ingest the 10 dev contracts (once)

Safe to re-run from scratch at any time, including on a brand-new VM: the cell below deletes a stale Drive checkpoint
automatically when this VM has no BM25 index (the checkpoint lives on Drive, the index on the VM disk; trusting it would
silently drop contracts from keyword search). Re-ingesting is idempotent, so nothing is duplicated.
`EMBEDDING_BATCH_SIZE=8` and the GPU-memory settings avoid the out-of-memory crashes seen earlier on the T4."""))

cells.append(code_cell("""import os
os.environ["QDRANT_COLLECTION_PRIMARY"] = "chunks_primary_colab_dev_ablation_v2"
os.environ["BM25_INDEX_DIR"] = "/content/LEXIS/data/bm25_index_colab_dev_ablation_v2"
os.environ["EMBEDDING_BATCH_SIZE"] = "8"
os.environ["HYPE_ENABLED"] = "false"
os.environ["RERANK_ENABLED"] = "false"
# Keep other GPU libraries Colab preinstalls (TensorFlow/JAX) from grabbing the whole GPU.
os.environ.update({
    "TF_FORCE_GPU_ALLOW_GROWTH": "true", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "USE_TF": "0", "USE_FLAX": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
})
INGEST_CHECKPOINT = f"{DRIVE_DIR}/ingest_checkpoint_v2.json"
INGEST_THROWAWAY_OUTPUT = f"{DRIVE_DIR}/ingest_pass_throwaway.json"  # this pass's numbers are not used

bm25_corpus = os.path.join(os.environ["BM25_INDEX_DIR"], "corpus.jsonl")
if os.path.exists(INGEST_CHECKPOINT) and not os.path.exists(bm25_corpus):
    os.remove(INGEST_CHECKPOINT)
    print("Fresh VM detected (no BM25 index) -> deleted the stale Drive checkpoint; re-ingesting all 10 contracts.")

!python scripts/setup_collections.py"""))

cells.append(code_cell("""!python -m lexis.evaluation.run_eval \\
  --benchmark cuad \\
  --cuad-path {cuad_path} \\
  --num-contracts 10 \\
  --max-questions 50 \\
  --top-k 30 \\
  --protocol doc_scoped \\
  --ingest-checkpoint {INGEST_CHECKPOINT} \\
  --skip-diagnostics \\
  --output {INGEST_THROWAWAY_OUTPUT}

print("Ingest pass complete.")"""))

cells.append(md_cell("""## 2. Two isolated eval passes on the same ingested collection

`--skip-ingest` (data is already written). The baseline is run twice -- the first query pass after ingest can differ from
the settled state -- and the **second** run is used."""))

cells.append(code_cell("""os.environ["RERANK_ENABLED"] = "false"
BASELINE_OUTPUT = f"{DRIVE_DIR}/dev_baseline.json"

# First pass (settle -- not used directly)
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

cells.append(code_cell("""# Settle pass -- THIS is the baseline result used below.
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

cells.append(code_cell("""# Diagnostic (read-only, ~1 min): where does the GPU memory go? Paste this output back to me too.
import subprocess, sys
print("=== GPU before our code runs ===")
!nvidia-smi --query-gpu=memory.used,memory.total --format=csv
!nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

DIAG = '''
import sys, torch
def snap(tag):
    free, total = torch.cuda.mem_get_info()
    print(f"{tag:26s} device used={(total-free)/2**30:5.2f} GiB | torch allocated={torch.cuda.memory_allocated()/2**30:5.2f} GiB")
torch.zeros(1).cuda(); snap("after CUDA init")
import lexis.evaluation.run_eval; snap("after importing lexis")
print("tensorflow imported:", "tensorflow" in sys.modules, "| jax imported:", "jax" in sys.modules)
from lexis.ingestion.embedder import get_embedder
get_embedder(); snap("after loading bge-m3")
'''
print("=== step-by-step GPU use inside a fresh process ===")
subprocess.run([sys.executable, "-c", DIAG])"""))

cells.append(code_cell("""# R4: cross-encoder rerank (downloads BAAI/bge-reranker-v2-m3 on first use).
# Tries the GPU first; if the GPU is too full for a second model, retries with the reranker on CPU
# (slower -- tens of minutes -- but the result is the same ranking logic).
import os, subprocess, sys
os.environ["RERANK_ENABLED"] = "true"
R4_OUTPUT = f"{DRIVE_DIR}/dev_r4_rerank.json"

def run_r4(device):
    if os.path.exists(R4_OUTPUT):
        os.remove(R4_OUTPUT)
    env = dict(os.environ, RERANK_DEVICE=device) if device else dict(os.environ)
    subprocess.run([sys.executable, "-m", "lexis.evaluation.run_eval", "--benchmark", "cuad",
                    "--cuad-path", cuad_path, "--num-contracts", "10", "--max-questions", "50",
                    "--top-k", "30", "--protocol", "doc_scoped", "--skip-ingest", "--skip-diagnostics",
                    "--output", R4_OUTPUT], env=env, cwd="/content/LEXIS")
    return os.path.exists(R4_OUTPUT)

print("=== R4 rerank on GPU ===")
ok = run_r4(None)
if not ok:
    print("GPU attempt failed (expected if the GPU is full) -> retrying with the reranker on CPU. "
          "Slower; leave it running.")
    ok = run_r4("cpu")
assert ok, "R4 produced no result file on GPU or CPU -- see the error above"
print("R4 (rerank) result written to", R4_OUTPUT)"""))

cells.append(md_cell("""## 3. Paired comparison -- COPY THIS CELL'S OUTPUT BACK TO ME

Each pass is compared against the pre-R1 dev reference in the repo via `paired_comparison` (bootstrap CI + permutation
test on per-case values matched by case_id). A gain counts only if `supported=True` (CI excludes zero AND p < 0.05 after
Holm correction across Recall@30 and MRR)."""))

cells.append(code_cell("""import json
from lexis.evaluation.stats import paired_comparison, claim_supported, holm_correction

reference = json.load(open("evaluation/reports/cuad_doc_scoped_dev.json"))["per_case"]
reference_by_id = {c["case_id"]: c for c in reference}

def compare(label, output_path, against=None):
    cur = json.load(open(output_path))["per_case"]
    cur_by_id = {c["case_id"]: c for c in cur}
    base_by_id = against if against is not None else reference_by_id
    common = sorted(set(base_by_id) & set(cur_by_id))
    print(f"--- {label} --- (n={len(common)})")
    r_a = [cur_by_id[i]["recall_at_k"] for i in common]; r_b = [base_by_id[i]["recall_at_k"] for i in common]
    m_a = [cur_by_id[i]["reciprocal_rank"] for i in common]; m_b = [base_by_id[i]["reciprocal_rank"] for i in common]
    rr, mm = paired_comparison(r_a, r_b, seed=0), paired_comparison(m_a, m_b, seed=0)
    p = holm_correction([rr.p_value, mm.p_value])
    print("Recall@30: %.4f vs %.4f  diff=%+.4f  CI[%.4f,%.4f]  p_holm=%.4f  improved/regressed=%d/%d  supported=%s" % (
        rr.mean_a, rr.mean_b, rr.mean_diff, rr.ci_lo, rr.ci_hi, p[0], rr.n_improved, rr.n_regressed, claim_supported(rr)))
    print("MRR      : %.4f vs %.4f  diff=%+.4f  CI[%.4f,%.4f]  p_holm=%.4f  improved/regressed=%d/%d  supported=%s" % (
        mm.mean_a, mm.mean_b, mm.mean_diff, mm.ci_lo, mm.ci_hi, p[1], mm.n_improved, mm.n_regressed, claim_supported(mm)))
    print()
    return cur_by_id

baseline_by_id = compare("Baseline (R1 included) vs old dev reference", BASELINE_OUTPUT)
compare("R4 rerank vs old dev reference", R4_OUTPUT)
compare("R4 rerank vs current baseline  <-- the clean R4 effect", R4_OUTPUT, against=baseline_by_id)"""))

cells.append(md_cell("## 4. Download results (optional)"))

cells.append(code_cell("""from google.colab import files
for path in [BASELINE_OUTPUT, R4_OUTPUT]:
    files.download(path)"""))

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("notebooks/cuad_doc_scoped_dev_ablation.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("Wrote notebooks/cuad_doc_scoped_dev_ablation.ipynb with", len(cells), "cells")
