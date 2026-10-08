"""Generator for notebooks/cuad_doc_scoped_dev_ablation_r7.ipynb: baseline vs R7 (document-preamble prior) on the
first 60 contracts of the DEV split. Run locally, commit the generated .ipynb. Never touches the held-out test split."""
import json

PINNED_COMMIT = "060bd3c18c2d23317127026083f6e61348ca2a71"
N_CONTRACTS = 60

def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = []

cells.append(md_cell(f"""# LEXIS - dev ablation: baseline vs R7 document-preamble prior ({N_CONTRACTS} dev contracts)

**Run top to bottom. Nothing to edit. About 45-75 minutes on a Colab T4.**

**Why:** questions about a contract's identity (title, parties, agreement/effective date) retrieve poorly (frozen test:
Document Name MRR 0.08, Agreement Date 0.14). On the DEV split 82-91% of those answers start in the first 2,000
characters, versus 5-15% for ordinary clause categories, so R7 adds each scoped document's first `N=2` chunks as one more
RRF list. N=2 was fixed from that dev analysis *before* any run; there is a single treatment arm (no sweep).

**Primary claim (pre-specified):** paired change in MRR and Recall@30 over all scored questions, Holm-corrected, claimed only
if `claim_supported`. **Descriptive:** header-fact vs clause-question subgroups. **Guardrail:** clause questions must not
regress (the prior could push an irrelevant opening chunk above a correct clause).

**Guardrails:** DEV split only (`--max-contracts` is refused for test). Nothing is frozen. Pinned commit `{PINNED_COMMIT}`.
Fresh Qdrant collection / BM25 dir / Drive folder; the ingest cell self-heals a stale checkpoint on a recycled VM."""))

cells.append(md_cell("## 0. Setup"))

cells.append(code_cell("""from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_DIR = "/content/drive/MyDrive/lexis_dev60_r7"
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

cells.append(code_cell("""# Needs Colab Secrets: QDRANT_URL, QDRANT_API_KEY, POSTGRES_URL  (no Gemini key needed)
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

os.environ["QDRANT_COLLECTION_PRIMARY"] = "chunks_primary_colab_dev60_r7"
os.environ["BM25_INDEX_DIR"] = "/content/LEXIS/data/bm25_index_colab_dev60_r7"
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

INGEST_CHECKPOINT = f"{{DRIVE_DIR}}/ingest_checkpoint_dev60_r7.json"
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

cells.append(md_cell("""## 3. Baseline, then R7 (separate passes over the same ingested data; query-time only, so these are quick)"""))

cells.append(code_cell("""BASELINE_OUTPUT = f"{DRIVE_DIR}/dev60_r7_baseline.json"
os.environ["PREAMBLE_PRIOR_CHUNKS"] = "0"
assert run_eval(BASELINE_OUTPUT), "baseline pass failed -- see the error above"
print("Baseline written to", BASELINE_OUTPUT)"""))

cells.append(code_cell("""R7_OUTPUT = f"{DRIVE_DIR}/dev60_r7_preamble2.json"
os.environ["PREAMBLE_PRIOR_CHUNKS"] = "2"
assert run_eval(R7_OUTPUT), "R7 pass failed -- see the error above"
print("R7 written to", R7_OUTPUT)"""))

cells.append(md_cell("""## 4. Paired comparison - COPY THIS CELL'S OUTPUT BACK TO ME

R7 vs baseline on identical questions. Headline = all questions (Holm over MRR and Recall@30). Subgroups are descriptive."""))

cells.append(code_cell("""import json
from collections import defaultdict
from lexis.evaluation.stats import paired_comparison, claim_supported, holm_correction

base = json.load(open(BASELINE_OUTPUT)); r7 = json.load(open(R7_OUTPUT))
bm = {c["case_id"]: c for c in base["per_case"]}; rm = {c["case_id"]: c for c in r7["per_case"]}
common = sorted(set(bm) & set(rm))
assert base["provenance"]["git_sha"] == r7["provenance"]["git_sha"], "runs used different code"
assert base["provenance"]["data_sha256"] == r7["provenance"]["data_sha256"], "runs used different data"
assert base["run_config"]["preamble_prior_chunks"] == 0 and r7["run_config"]["preamble_prior_chunks"] == 2, "flags are not as expected"
assert not base["run_config"]["rerank_enabled"] and not r7["run_config"]["rerank_enabled"]
print(f"questions compared: {len(common)} | contracts: {base['run_config']['num_contracts']} | commit {base['provenance']['git_sha'][:7]}")
print(f"unmapped (excluded): baseline {base['num_cases_unmapped']}, R7 {r7['num_cases_unmapped']}")
print(f"mean Recall@30: {base['mean_recall_at_k']:.4f} -> {r7['mean_recall_at_k']:.4f}")
print(f"mean MRR      : {base['mean_reciprocal_rank']:.4f} -> {r7['mean_reciprocal_rank']:.4f}")

def compare(ids):
    rec = paired_comparison([rm[i]["recall_at_k"] for i in ids], [bm[i]["recall_at_k"] for i in ids], seed=0)
    mrr = paired_comparison([rm[i]["reciprocal_rank"] for i in ids], [bm[i]["reciprocal_rank"] for i in ids], seed=0)
    return rec, mrr

rec, mrr = compare(common)
p = holm_correction([rec.p_value, mrr.p_value])
print()
print("HEADLINE (all questions)")
print("  Recall@30 diff=%+.4f CI[%+.4f,%+.4f] p_holm=%.4f improved/regressed/tied=%d/%d/%d supported=%s" % (
    rec.mean_diff, rec.ci_lo, rec.ci_hi, p[0], rec.n_improved, rec.n_regressed, rec.n_tied, claim_supported(rec)))
print("  MRR       diff=%+.4f CI[%+.4f,%+.4f] p_holm=%.4f improved/regressed/tied=%d/%d/%d supported=%s" % (
    mrr.mean_diff, mrr.ci_lo, mrr.ci_hi, p[1], mrr.n_improved, mrr.n_regressed, mrr.n_tied, claim_supported(mrr)))

# Subgroups (descriptive). Header categories were identified from the DEV answer-position analysis, not from these results.
HEADER = {"Document Name", "Parties", "Agreement Date", "Effective Date"}
def category(case_id): return case_id.split("__")[-1]
groups = defaultdict(list)
for i in common:
    groups["header-fact" if category(i) in HEADER else "clause"].append(i)
print()
print("SUBGROUPS (descriptive, uncorrected)")
for name, ids in groups.items():
    r, m = compare(ids)
    print("  %-11s n=%4d  MRR %.4f -> %.4f (diff %+.4f CI[%+.4f,%+.4f])  Recall@30 diff %+.4f CI[%+.4f,%+.4f]" % (
        name, len(ids), sum(bm[i]["reciprocal_rank"] for i in ids)/len(ids), sum(rm[i]["reciprocal_rank"] for i in ids)/len(ids),
        m.mean_diff, m.ci_lo, m.ci_hi, r.mean_diff, r.ci_lo, r.ci_hi))
print()
print("PER HEADER CATEGORY (MRR baseline -> R7)")
for cat in sorted(HEADER):
    ids = [i for i in common if category(i) == cat]
    if ids:
        print("  %-16s n=%3d  %.3f -> %.3f" % (cat, len(ids), sum(bm[i]["reciprocal_rank"] for i in ids)/len(ids), sum(rm[i]["reciprocal_rank"] for i in ids)/len(ids)))
r1b = sum(1 for i in common if bm[i]["reciprocal_rank"] == 1.0); r1r = sum(1 for i in common if rm[i]["reciprocal_rank"] == 1.0)
print(f"\\nrank-1 hits: baseline {r1b} -> R7 {r1r} of {len(common)}")"""))

cells.append(md_cell("## 5. Download results (optional)"))

cells.append(code_cell("""from google.colab import files
for path in [BASELINE_OUTPUT, R7_OUTPUT]:
    files.download(path)"""))

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("notebooks/cuad_doc_scoped_dev_ablation_r7.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("Wrote notebooks/cuad_doc_scoped_dev_ablation_r7.ipynb with", len(cells), "cells")
