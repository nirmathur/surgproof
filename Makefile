PY ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
PORT ?= 8600
SRC ?= gt_corrupted

.PHONY: all setup demo run gt corrupted eval recheck render test clean-tracker

all: gt corrupted eval recheck render

setup:
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
