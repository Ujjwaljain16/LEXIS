"""Generator for notebooks/cuad_answer_eval_dev.ipynb: end-to-end ANSWER-quality evaluation (citation hit/precision,
abstention on CUAD's unanswerable questions, latency) on the first 60 DEV contracts through the shipped AnswerService.
Run locally, commit the generated .ipynb. DEV split only: develop prompts here; the held-out test split is reported
once, later, with the prompt frozen."""
import json

PINNED_COMMIT = "5b23378ed3dc2ab63849ae6c3b3f7c951f004640"
N_CONTRACTS = 60

def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = []

cells.append(md_cell(f"""# LEXIS - answer-quality evaluation (dev, {N_CONTRACTS} contracts)

**Run top to bottom. About 60-90 minutes on a Colab T4 (ingest ~30 min, then ~200 LLM calls at the rate limit).**

Measures the shipped answer path, document-scoped, with objective metrics and no LLM judge: citation hit rate and
precision against CUAD's gold chunks, abstention on questions whose clause category is absent from the contract,
how often the evidence was in the prompt, and latency. **Not measured:** free-text answer correctness.

**Needs these Colab Secrets** (values never printed): `QDRANT_URL`, `QDRANT_API_KEY`, `POSTGRES_URL`, and for the LLM:
`LLM_API_KEY`, `LLM_MODEL` (LiteLLM model id, e.g. `openai/<model>` for an OpenAI-compatible gateway), and
`LLM_API_BASE` (the gateway URL; omit for a provider LiteLLM routes natively).

DEV split only; pinned commit `{PINNED_COMMIT}`."""))

cells.append(md_cell("## 0. Setup"))

cells.append(code_cell("""from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_DIR = "/content/drive/MyDrive/lexis_answer_eval"
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
    raise RuntimeError("`import lexis` failed even in a FRESH subprocess -- install is broken.\\n" + check.stderr)
try:
    import lexis  # noqa: F401
    print("lexis is importable in THIS kernel -- proceed.")
except ModuleNotFoundError:
    raise RuntimeError(
        "lexis installed, but this kernel cannot see it yet (standard Colab gotcha). "
        "Fix: Runtime -> Restart session, then re-run every cell from the top."
    ) from None"""))

cells.append(code_cell("""# Needs Colab Secrets: QDRANT_URL, QDRANT_API_KEY, POSTGRES_URL 
import os
from google.colab import userdata

os.environ["QDRANT_URL"] = userdata.get("QDRANT_URL")
os.environ["QDRANT_API_KEY"] = userdata.get("QDRANT_API_KEY")
os.environ["POSTGRES_URL"] = userdata.get("POSTGRES_URL")
print("Credentials loaded from Colab Secrets (values not printed).")"""))

cells.append(code_cell("""# Fail fast, with the REAL error, if Qdrant is unreachable.
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
        "Check Colab Secrets QDRANT_URL (https:// + port) and QDRANT_API_KEY, and that the cluster is Running "
        "at cloud.qdrant.io (free clusters are suspended after inactivity)."
    ) from None"""))

cells.append(code_cell("""import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU. Runtime -> Change runtime type -> T4 GPU, restart, and re-run from the top.")"""))

cells.append(code_cell("""from huggingface_hub import hf_hub_download

cuad_path = hf_hub_download(
    repo_id="theatticusproject--cuad".replace("--", "/"),
    repo_type="dataset",
    filename="CUAD_v1/CUAD_v1.json",
)
print("CUAD dataset at:", cuad_path)"""))

cells.append(md_cell(f"""## 1. Shared runner (output is streamed so you always see progress)"""))

cells.append(code_cell(f"""import os, subprocess, sys

os.environ["QDRANT_COLLECTION_PRIMARY"] = "chunks_primary_colab_answer_eval"
os.environ["BM25_INDEX_DIR"] = "/content/LEXIS/data/bm25_index_colab_answer_eval"
os.environ["EMBEDDING_BATCH_SIZE"] = "8"
os.environ["HYPE_ENABLED"] = "false"
os.environ["RERANK_ENABLED"] = "false"
# Keep other GPU libraries Colab preinstalls (TensorFlow/JAX) from grabbing the whole GPU.
os.environ.update({{
    "TF_FORCE_GPU_ALLOW_GROWTH": "true", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "USE_TF": "0", "USE_FLAX": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}})

SCOPE_ARGS = ["--split-manifest", "evaluation/splits/cuad_split_v1.json", "--split-name", "dev",
              "--max-contracts", "{N_CONTRACTS}", "--max-questions", "10000"]

def stream(cmd, env=None):
    \"\"\"Run a command, echoing its output line by line (skipping progress-bar spam).\"\"\"
    p = subprocess.Popen(cmd, cwd="/content/LEXIS", env=env or os.environ.copy(), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        if "it/s]" in line or "s/it]" in line or "LiteLLM" in line:
            continue
        print(line, end="")
    return p.wait()

def run_eval(output, extra=(), env=None, skip_ingest=True):
    if os.path.exists(output):
        os.remove(output)
    cmd = [sys.executable, "-m", "lexis.evaluation.run_eval", "--benchmark", "cuad", "--cuad-path", cuad_path,
           *SCOPE_ARGS, "--top-k", "30", "--protocol", "doc_scoped", "--skip-diagnostics",
           "--output", output, *(["--skip-ingest"] if skip_ingest else []), *extra]
    code = stream(cmd, env)
    ok = code == 0 and os.path.exists(output)
    print(f"[run_eval exit={{code}}] result file written: {{ok}}")
    return ok

INGEST_CHECKPOINT = f"{{DRIVE_DIR}}/ingest_checkpoint_answer_eval.json"
bm25_corpus = os.path.join(os.environ["BM25_INDEX_DIR"], "corpus.jsonl")
if os.path.exists(INGEST_CHECKPOINT) and not os.path.exists(bm25_corpus):
    os.remove(INGEST_CHECKPOINT)
    print("Fresh VM detected (no BM25 index) -> deleted the stale Drive checkpoint; re-ingesting everything.")

assert stream([sys.executable, "scripts/setup_collections.py"]) == 0, "Qdrant collection setup failed (see message above)"
print("Ready.")"""))

cells.append(md_cell(f"""## 2. Ingest the {N_CONTRACTS} dev contracts (resumable)

Takes roughly 20-40 minutes. If Colab disconnects, reconnect, re-run Setup and the cell above, then re-run this one:
already-ingested contracts are skipped via the Drive checkpoint (and the self-heal above handles a recycled VM)."""))

cells.append(code_cell("""INGEST_THROWAWAY = f"{DRIVE_DIR}/ingest_pass_throwaway.json"   # this pass's numbers are not used
ok = run_eval(INGEST_THROWAWAY, extra=["--ingest-checkpoint", INGEST_CHECKPOINT], skip_ingest=False)
assert ok, "ingest pass failed -- see the error above"
print("Ingest complete.")"""))

cells.append(md_cell("""## 3. Connect the LLM (fails fast, before the long run)"""))

cells.append(code_cell("""from google.colab import userdata
import os

def _secret(name, required=True):
    try:
        return userdata.get(name)
    except Exception:
        if required:
            raise RuntimeError(f"Colab secret {name!r} is missing (Secrets panel -> add it -> toggle 'Notebook access').") from None
        return None

os.environ["GEMINI_API_KEY"] = _secret("LLM_API_KEY")            # the setting names predate multi-provider support
os.environ["GEMINI_MODEL_SYNTHESIS"] = _secret("LLM_MODEL")
_base = _secret("LLM_API_BASE", required=False)
if _base:
    os.environ["LLM_API_BASE"] = _base

import litellm
try:
    r = litellm.completion(model=os.environ["GEMINI_MODEL_SYNTHESIS"], api_key=os.environ["GEMINI_API_KEY"],
                           api_base=_base or None, max_tokens=16,
                           messages=[{"role": "user", "content": "Reply with the single word: ok"}])
    print("LLM reachable. Model replied:", (r.choices[0].message.content or "").strip()[:40])
except Exception as e:
    raise RuntimeError(f"LLM call failed: {type(e).__name__}. Check LLM_MODEL (LiteLLM id), LLM_API_BASE and the key's quota. "
                       f"First 200 chars of the provider message: {str(e)[:200]}") from None"""))

cells.append(md_cell("""## 4. Run the evaluation (one question at a time, rate-limited; progress streams below)"""))

cells.append(code_cell(f"""VERIFY = False   # True adds a per-claim NLI pass (loads one more model; NLI accuracy on legal text is unmeasured)
ANSWER_EVAL_OUTPUT = f"{{DRIVE_DIR}}/answer_eval_dev60.json"
if os.path.exists(ANSWER_EVAL_OUTPUT):
    os.remove(ANSWER_EVAL_OUTPUT)
os.environ["ANSWER_EVAL_REQUESTS_PER_MINUTE"] = "10"   # lower this if the provider returns rate-limit errors
cmd = [sys.executable, "-m", "lexis.evaluation.run_answer_eval", "--cuad-path", cuad_path,
       "--split-manifest", "evaluation/splits/cuad_split_v1.json", "--split-name", "dev",
       "--max-contracts", "{N_CONTRACTS}", "--n-answerable", "100", "--n-unanswerable", "100",
       "--output", ANSWER_EVAL_OUTPUT] + (["--verify"] if VERIFY else [])
code = stream(cmd)
assert code == 0 and os.path.exists(ANSWER_EVAL_OUTPUT), "answer evaluation failed -- see the error above"
print("Written to", ANSWER_EVAL_OUTPUT)"""))

cells.append(md_cell("""## 5. Results - COPY THIS CELL'S OUTPUT BACK TO ME"""))

cells.append(code_cell("""import json
rep = json.load(open(ANSWER_EVAL_OUTPUT)); s = rep["summary"]
def fmt(m):
    return "n/a" if m["value"] is None else "%.3f [%.3f, %.3f] n=%d" % (m["value"], m["lo"], m["hi"], m["n"])
print("model:", rep["run_config"]["generator_model"], "| gateway:", rep["run_config"]["llm_gateway_host"], "| commit", rep["provenance"]["git_sha"][:7])
print("counts:", s["counts"], "| failure rate:", fmt(s["failure_rate"]))
print()
print("ANSWERABLE questions")
for k, v in s["answerable"].items():
    print("  %-42s %s" % (k, fmt(v)))
print()
print("UNANSWERABLE questions")
for k, v in s["unanswerable"].items():
    print("  %-42s %s" % (k, fmt(v)))
print()
print("abstention balanced accuracy:", s["abstention_balanced_accuracy"])
print("NLI-judged support:", s["nli_judged_support"])
print("latency (s):", {k: (round(v, 2) if v is not None else None) for k, v in s["latency_s"].items()})
errs = [c["error"] for c in rep["per_case"] if c["outcome"] == "failed"]
print()
print("failures by error type:", {e: errs.count(e) for e in set(errs)})"""))

cells.append(md_cell("## 6. Download results (optional)"))

cells.append(code_cell("""from google.colab import files
files.download(ANSWER_EVAL_OUTPUT)"""))

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("notebooks/cuad_answer_eval_dev.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("Wrote notebooks/cuad_answer_eval_dev.ipynb with", len(cells), "cells")
