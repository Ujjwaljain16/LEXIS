# One-command entry points. Heavy steps (ingest, evaluation) are meant for Colab/GPU; see README.
PY ?= python

.PHONY: install test lint check serve repro-report

install:            ## editable install with dev extras
	$(PY) -m pip install -e ".[dev]"

test:               ## unit tests (no network, no models)
	$(PY) -m pytest tests/unit -q

lint:               ## zero-hardcoding ratchet (fails if violations increase)
	$(PY) -m lexis.quality.hardcode_lint

check: lint test    ## what CI runs

serve:              ## API + UI on http://localhost:8000  (set LEXIS_API_KEYS, or AUTH_DISABLED=true for local use)
	$(PY) -m uvicorn lexis.serving.app:app --host 0.0.0.0 --port 8000

repro-report:       ## regenerate README figure from the frozen test report (evaluation/reports is gitignored; see README "Reproduction")
	$(PY) scripts/make_results_report.py
