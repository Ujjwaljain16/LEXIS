"""
Importing the ingestion/evaluation entry points must not drag in umap or TensorFlow.

Bug fixed: indexing/raptor.py imported umap at module level, and umap.parametric_umap imports
TensorFlow. ingestion/pipeline.py imports raptor on every ingest/eval run (none of which use RAPTOR
clustering), so on Colab -- where TensorFlow is preinstalled but pinned to a newer protobuf than this
project's dependencies allow -- every run died at import time with a protobuf VersionError. Locally it
merely cost seconds and gigabytes of RAM. Checked in a fresh subprocess so modules already imported by
other tests in this process can't mask a regression.
"""
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", [
    "lexis.ingestion.pipeline",
    "lexis.indexing.raptor",
    "lexis.evaluation.run_eval",
])
def test_importing_does_not_pull_in_umap_or_tensorflow(module):
    code = (
        f"import sys; import {module}; "
        "bad = sorted(m for m in ('umap', 'tensorflow') if m in sys.modules); "
        "assert not bad, f'eagerly imported: {bad}'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
