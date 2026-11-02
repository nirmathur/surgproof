PY ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
PORT ?= 8600
SRC ?= gt_corrupted

.PHONY: all setup setup-dev lock serve setup-full demo run gt corrupted eval recheck render test clean-tracker

all: gt corrupted eval recheck render

# End-to-end app: a private .venv with exactly the locked, hash-checked runtime dependencies.
setup:
	@command -v ffmpeg >/dev/null || { echo "ffmpeg is required (macOS: brew install ffmpeg)"; exit 1; }
	@command -v ffprobe >/dev/null || { echo "ffprobe is required (macOS: brew install ffmpeg)"; exit 1; }
	@command -v claude >/dev/null || echo "warning: the claude CLI was not found; the pipeline needs it to call the model"
	@[ -x .venv/bin/python ] || python3 -m venv .venv
	.venv/bin/python -m pip install --quiet --require-hashes -r requirements-e2e.lock
	@echo "Ready. Now: make serve"

# Adds pytest and the test client. Not needed to run the app.
setup-dev: setup
	.venv/bin/python -m pip install --quiet --require-hashes -r requirements-e2e-dev.lock

# Regenerate the lock files after changing requirements-e2e.txt or requirements-e2e-dev.txt.
lock:
	uv pip compile --universal --generate-hashes --quiet requirements-e2e.txt -o requirements-e2e.lock
	uv pip compile --universal --generate-hashes --quiet requirements-e2e.txt requirements-e2e-dev.txt -o requirements-e2e-dev.lock

# Start the app at http://localhost:$(PORT). It binds to this machine only.
serve:
	PORT=$(PORT) $(PY) -m surgproof.server

# The original research pipeline (Gemini, torch, streamlit).
setup-full:
	$(PY) -m pip install -r requirements.txt -r web/requirements.txt || $(PY) -m pip install --break-system-packages -r requirements.txt -r web/requirements.txt
	@if [ ! -f .env ] && [ -f .env.example ]; then cp .env.example .env; fi

demo:
	PORT=$(PORT) ./run.sh

run:
	PORT=$(PORT) ./run.sh

demo: run

gt:
	$(PY) -m tracker.index
	$(PY) -m tracker.detect
	$(PY) -m tracker.field
	$(PY) -m tracker.camera --source gt
	$(PY) -m tracker.track --source gt
	$(PY) -m tracker.signals --source gt

corrupted: gt
	$(PY) -m tracker.corrupt
	$(PY) -m tracker.camera --source gt_corrupted
	$(PY) -m tracker.track --source gt_corrupted
	$(PY) -m tracker.signals --source gt_corrupted

eval: corrupted
	$(PY) -m tracker.evaluate

recheck: eval
	$(PY) -m tracker.recheck --source $(SRC)

render:
	$(PY) -m tracker.render --clip video12_15750 --source gt_corrupted --errors

# Gemini detections: write out/tracker/detections_gemini.parquet (same schema), then: make gemini
gemini:
	$(PY) -m tracker.camera --source gemini
	$(PY) -m tracker.track --source gemini
	$(PY) -m tracker.signals --source gemini
	$(PY) -m tracker.recheck --source gemini

test:
	$(PY) -m pytest -q tests

clean-tracker:
	rm -rf out/tracker out/eval out/render
