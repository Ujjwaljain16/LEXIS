"""Generator for notebooks/cuad_dev_ablation_r7b_weighted.ipynb: pre-specified sweep of the preamble prior's RRF weight on
dev contracts 1-60, a coded selection rule, and confirmation of the chosen weight on unseen dev contracts 61-120.
Run locally, commit the generated .ipynb. DEV split only; never touches the held-out test split."""
import json

PINNED_COMMIT = "af2712bfb0f1de2024e98b3dbb190c33520c78d7"
N_CONTRACTS = 120

def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = []

cells.append(md_cell(f"""# LEXIS - R7b: can the preamble prior help header questions WITHOUT hurting clause questions?

**Run top to bottom. Nothing to edit. About 2-2.5 hours on a Colab T4 (ingest ~60-75 min, then 6 short query passes).**

**Why:** the first R7 run (preamble prior at full weight) gave the first CI-supported Recall@30 gain of any rung
(+0.053) and lifted title/parties/date questions' MRR 0.26 -> 0.50, but it **failed its own guardrail**: clause-question
MRR fell 0.59 -> 0.51 (CI [-0.107, -0.065]) because an opening chunk that no retrieval path ranked was voted as high as a
rank-1 hit. The fix being tested: a smaller RRF weight for the prior, so it only breaks ties / lifts chunks that already
have retrieval support.

**Pre-specified protocol (fixed before any run, no tuning on the confirmation slice):**
1. **Slice A** = dev contracts 1-60 (already looked at). Candidate weights **0.25, 0.5, 1.0** (N=2 chunks fixed).
2. **Selection rule:** a weight is eligible only if clause-question MRR does not regress by more than **0.02**
   (95% CI lower bound >= -0.02) and overall Recall@30 improves. Choose the eligible weight with the largest overall MRR
   gain (ties: smaller weight). If none is eligible, R7 is rejected as a general default and the run stops there.
3. **Slice B** = dev contracts 61-120 (never seen). Baseline vs the chosen weight: claim only if `claim_supported`
   (Holm over Recall@30 and MRR), and report whether the clause non-inferiority holds.

**Guardrails:** DEV split only (`--max-contracts`/`--contract-offset` are refused for test). Nothing is frozen.
Pinned commit `{PINNED_COMMIT}`. Fresh Qdrant collection / BM25 dir / Drive folder; self-healing checkpoint."""))

cells.append(md_cell("## 0. Setup"))

cells.append(code_cell("""from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_DIR = "/content/drive/MyDrive/lexis_dev120_r7b"
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

os.environ["QDRANT_COLLECTION_PRIMARY"] = "chunks_primary_colab_dev120_r7b"
os.environ["BM25_INDEX_DIR"] = "/content/LEXIS/data/bm25_index_colab_dev120_r7b"
os.environ["EMBEDDING_BATCH_SIZE"] = "8"
os.environ["HYPE_ENABLED"] = "false"
os.environ["RERANK_ENABLED"] = "false"
# Keep other GPU libraries Colab preinstalls (TensorFlow/JAX) from grabbing the whole GPU.
os.environ.update({{
    "TF_FORCE_GPU_ALLOW_GROWTH": "true", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "USE_TF": "0", "USE_FLAX": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}})

def scope_args(offset, n):
    return ["--split-manifest", "evaluation/splits/cuad_split_v1.json", "--split-name", "dev",
            "--max-contracts", str(n), "--contract-offset", str(offset), "--max-questions", "10000"]

def stream(cmd, env=None):
    \"\"\"Run a command, echoing its output line by line (skipping progress-bar spam).\"\"\"
    p = subprocess.Popen(cmd, cwd="/content/LEXIS", env=env or os.environ.copy(), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        if "it/s]" in line or "s/it]" in line or "LiteLLM" in line:
            continue
        print(line, end="")
    return p.wait()

def run_eval(output, extra=(), env=None, skip_ingest=True, offset=0, n=60):
    if os.path.exists(output):
        os.remove(output)
    cmd = [sys.executable, "-m", "lexis.evaluation.run_eval", "--benchmark", "cuad", "--cuad-path", cuad_path,
           *scope_args(offset, n), "--top-k", "30", "--protocol", "doc_scoped", "--skip-diagnostics",
           "--output", output, *(["--skip-ingest"] if skip_ingest else []), *extra]
    code = stream(cmd, env)
    ok = code == 0 and os.path.exists(output)
    print(f"[run_eval exit={{code}}] result file written: {{ok}}")
    return ok

INGEST_CHECKPOINT = f"{{DRIVE_DIR}}/ingest_checkpoint_dev120_r7b.json"
bm25_corpus = os.path.join(os.environ["BM25_INDEX_DIR"], "corpus.jsonl")
if os.path.exists(INGEST_CHECKPOINT) and not os.path.exists(bm25_corpus):
    os.remove(INGEST_CHECKPOINT)
    print("Fresh VM detected (no BM25 index) -> deleted the stale Drive checkpoint; re-ingesting everything.")

assert stream([sys.executable, "scripts/setup_collections.py"]) == 0, "Qdrant collection setup failed (see message above)"
print("Ready.")"""))

cells.append(md_cell(f"""## 2. Ingest the {N_CONTRACTS} dev contracts (resumable)

Takes roughly 60-75 minutes. If Colab disconnects, reconnect, re-run Setup and the cell above, then re-run this one:
already-ingested contracts are skipped via the Drive checkpoint (and the self-heal above handles a recycled VM)."""))

cells.append(code_cell("""INGEST_THROWAWAY = f"{DRIVE_DIR}/ingest_pass_throwaway.json"   # this pass's numbers are not used
ok = run_eval(INGEST_THROWAWAY, extra=["--ingest-checkpoint", INGEST_CHECKPOINT], skip_ingest=False, offset=0, n=120)
assert ok, "ingest pass failed -- see the error above"
print("Ingest complete.")"""))

cells.append(md_cell("""## 3. Stage A: sweep the prior's weight on dev contracts 1-60 (query-time only)"""))

cells.append(code_cell("""import json
from collections import defaultdict
from lexis.evaluation.stats import paired_comparison, claim_supported, holm_correction

WEIGHTS = [0.25, 0.5, 1.0]      # pre-specified candidates
MARGIN = 0.02                   # pre-specified clause-question non-inferiority margin (MRR)
PRIOR_CHUNKS = 2
HEADER = {"Document Name", "Parties", "Agreement Date", "Effective Date"}   # from the dev answer-position analysis

def category(case_id): return case_id.split("__")[-1]

def load(path): return json.load(open(path))

def run_arm(tag, offset, weight):
    \"\"\"weight=None -> baseline (prior off). Re-uses a finished output from the same commit (restart-safe).\"\"\"
    out = f"{DRIVE_DIR}/r7b_{tag}.json"
    if os.path.exists(out):
        try:
            if load(out)["provenance"]["git_sha"] == PINNED_COMMIT:
                print(f"[{tag}] reusing finished result"); return out
        except Exception:
            pass
    env = dict(os.environ, PREAMBLE_PRIOR_CHUNKS=str(0 if weight is None else PRIOR_CHUNKS),
               PREAMBLE_PRIOR_WEIGHT=str(1.0 if weight is None else weight))
    print(f"[{tag}] running (offset={offset}, prior_weight={weight})")
    assert run_eval(out, env=env, offset=offset, n=60), f"{tag} pass failed -- see the error above"
    return out

def paired(base, arm, ids):
    bm = {c["case_id"]: c for c in base["per_case"]}; am = {c["case_id"]: c for c in arm["per_case"]}
    rec = paired_comparison([am[i]["recall_at_k"] for i in ids], [bm[i]["recall_at_k"] for i in ids], seed=0)
    mrr = paired_comparison([am[i]["reciprocal_rank"] for i in ids], [bm[i]["reciprocal_rank"] for i in ids], seed=0)
    return rec, mrr

def analyse(base, arm):
    bm = {c["case_id"] for c in base["per_case"]}; am = {c["case_id"] for c in arm["per_case"]}
    assert bm == am, "arms scored different question sets"
    assert base["provenance"]["git_sha"] == arm["provenance"]["git_sha"], "runs used different code"
    assert base["provenance"]["data_sha256"] == arm["provenance"]["data_sha256"], "runs used different data"
    ids = sorted(bm)
    hdr = [i for i in ids if category(i) in HEADER]; cls = [i for i in ids if category(i) not in HEADER]
    rec, mrr = paired(base, arm, ids)
    p = holm_correction([rec.p_value, mrr.p_value])
    _, hdr_mrr = paired(base, arm, hdr); _, cls_mrr = paired(base, arm, cls)
    return dict(n=len(ids), n_hdr=len(hdr), n_cls=len(cls), rec=rec, mrr=mrr, p=p, hdr_mrr=hdr_mrr, cls_mrr=cls_mrr)

def hit_at(res, k):
    return sum(1 for c in res["per_case"] if c["reciprocal_rank"] > 0 and 1.0 / c["reciprocal_rank"] <= k + 1e-9) / len(res["per_case"])

def row(label, a):
    print("  %-12s Recall@30 %+.4f [%+.4f,%+.4f] | MRR %+.4f [%+.4f,%+.4f] | header MRR %+.4f | clause MRR %+.4f [%+.4f,%+.4f]" % (
        label, a["rec"].mean_diff, a["rec"].ci_lo, a["rec"].ci_hi, a["mrr"].mean_diff, a["mrr"].ci_lo, a["mrr"].ci_hi,
        a["hdr_mrr"].mean_diff, a["cls_mrr"].mean_diff, a["cls_mrr"].ci_lo, a["cls_mrr"].ci_hi))

A_base = load(run_arm("A_base", 0, None))
arms_A = {w: load(run_arm(f"A_w{w}", 0, w)) for w in WEIGHTS}

print(f"STAGE A (dev contracts 1-60): n={len(A_base['per_case'])} questions; paired differences vs baseline")
results_A = {}
for w in WEIGHTS:
    assert arms_A[w]["run_config"]["preamble_prior_weight"] == w and arms_A[w]["run_config"]["preamble_prior_chunks"] == PRIOR_CHUNKS
    results_A[w] = analyse(A_base, arms_A[w]); row(f"weight {w}", results_A[w])
print("  Hit@1/5/10  baseline: %.3f / %.3f / %.3f" % tuple(hit_at(A_base, k) for k in (1, 5, 10)))
for w in WEIGHTS:
    print("              weight %-4s: %.3f / %.3f / %.3f" % ((w,) + tuple(hit_at(arms_A[w], k) for k in (1, 5, 10))))

eligible = [(results_A[w]["mrr"].mean_diff, -w, w) for w in WEIGHTS
            if results_A[w]["cls_mrr"].ci_lo >= -MARGIN and results_A[w]["rec"].mean_diff > 0]
CHOSEN = max(eligible)[2] if eligible else None
print()
print("SELECTION RULE (clause MRR CI lower bound >= -%.2f and Recall@30 gain > 0; then largest MRR gain):" % MARGIN)
print("  eligible weights:", [e[2] for e in eligible] or "NONE")
print("  CHOSEN:", CHOSEN if CHOSEN is not None else "none - R7 is rejected as a general default")"""))

cells.append(md_cell("""## 4. Stage B: confirm the chosen weight on unseen dev contracts 61-120 - COPY THIS CELL'S OUTPUT (and Stage A's) BACK TO ME"""))

cells.append(code_cell("""if CHOSEN is None:
    print("No weight passed the pre-specified selection rule on Stage A; nothing to confirm. Report this as a negative result.")
else:
    B_base = load(run_arm("B_base", 60, None))
    B_arm = load(run_arm(f"B_w{CHOSEN}", 60, CHOSEN))
    assert B_arm["run_config"]["contract_offset"] == 60 and B_arm["run_config"]["preamble_prior_weight"] == CHOSEN
    b = analyse(B_base, B_arm)
    print(f"STAGE B (dev contracts 61-120, never used for selection): n={b['n']} questions, weight={CHOSEN}, {PRIOR_CHUNKS} chunks")
    print("  mean Recall@30: %.4f -> %.4f" % (B_base["mean_recall_at_k"], B_arm["mean_recall_at_k"]))
    print("  mean MRR      : %.4f -> %.4f" % (B_base["mean_reciprocal_rank"], B_arm["mean_reciprocal_rank"]))
    print()
    print("HEADLINE (Holm over Recall@30 and MRR)")
    print("  Recall@30 diff=%+.4f CI[%+.4f,%+.4f] p_holm=%.4f improved/regressed/tied=%d/%d/%d supported=%s" % (
        b["rec"].mean_diff, b["rec"].ci_lo, b["rec"].ci_hi, b["p"][0], b["rec"].n_improved, b["rec"].n_regressed, b["rec"].n_tied, claim_supported(b["rec"])))
    print("  MRR       diff=%+.4f CI[%+.4f,%+.4f] p_holm=%.4f improved/regressed/tied=%d/%d/%d supported=%s" % (
        b["mrr"].mean_diff, b["mrr"].ci_lo, b["mrr"].ci_hi, b["p"][1], b["mrr"].n_improved, b["mrr"].n_regressed, b["mrr"].n_tied, claim_supported(b["mrr"])))
    print()
    print("SUBGROUPS (descriptive)")
    print("  header-fact n=%d  MRR diff %+.4f CI[%+.4f,%+.4f]" % (b["n_hdr"], b["hdr_mrr"].mean_diff, b["hdr_mrr"].ci_lo, b["hdr_mrr"].ci_hi))
    print("  clause      n=%d  MRR diff %+.4f CI[%+.4f,%+.4f]" % (b["n_cls"], b["cls_mrr"].mean_diff, b["cls_mrr"].ci_lo, b["cls_mrr"].ci_hi))
    noninf = b["cls_mrr"].ci_lo >= -MARGIN
    print()
    print("CLAUSE NON-INFERIORITY (CI lower bound >= -%.2f): %s" % (MARGIN, "PASS" if noninf else "FAIL"))
    print("Hit@1/5/10  baseline: %.3f / %.3f / %.3f   prior: %.3f / %.3f / %.3f   (exploratory; Hit@5 is what a 5-chunk prompt sees)" % (
        tuple(hit_at(B_base, k) for k in (1, 5, 10)) + tuple(hit_at(B_arm, k) for k in (1, 5, 10))))
    for cat in sorted(HEADER):
        ids = [c["case_id"] for c in B_base["per_case"] if category(c["case_id"]) == cat]
        bm = {c["case_id"]: c["reciprocal_rank"] for c in B_base["per_case"]}; am = {c["case_id"]: c["reciprocal_rank"] for c in B_arm["per_case"]}
        if ids: print("  %-16s n=%3d  MRR %.3f -> %.3f" % (cat, len(ids), sum(bm[i] for i in ids) / len(ids), sum(am[i] for i in ids) / len(ids)))
    verdict = "CONFIRMED (supported gain, clause non-inferiority holds)" if (claim_supported(b["rec"]) or claim_supported(b["mrr"])) and noninf else \\
              "NOT CONFIRMED - report as inconclusive/negative"
    print()
    print("VERDICT:", verdict)"""))

cells.append(md_cell("## 5. Download results (optional)"))

cells.append(code_cell("""from google.colab import files
import glob
for path in sorted(glob.glob(f"{DRIVE_DIR}/r7b_*.json")):
    files.download(path)"""))

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("notebooks/cuad_dev_ablation_r7b_weighted.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("Wrote notebooks/cuad_dev_ablation_r7b_weighted.ipynb with", len(cells), "cells")
