# Run from narrator/:  make all   (every stage cached in narrator/cache: reruns are free)
PY := cd .. && .venv/bin/python
.PHONY: all data kb events claims check render eval review preview showcase test
all: data kb events claims check render eval review preview
data:     ; $(PY) -m narrator.data
kb:       ; $(PY) -m narrator.kb build
events:   ; $(PY) -m narrator.events
claims:   ; $(PY) -m narrator.claims --mode both --split test
check:    ; $(PY) -m narrator.check
render:   ; $(PY) -m narrator.render
eval:     ; $(PY) -m narrator.eval
review:   ; $(PY) -m narrator.review --claims narrator/out/showcase/claims.jsonl --out narrator/out/showcase/review.json || true
preview:  ; $(PY) -m narrator.demo_preview --clip video12_15750 --out narrator/out/demo_preview.html
showcase: ; $(PY) -m narrator.showcase --url "$(URL)"
test:     ; $(PY) -m pytest -q narrator/tests
