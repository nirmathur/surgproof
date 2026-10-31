# Surgical Atlas

An autonomous Gemini agent that teaches itself a surgery from public sources, labels surgical video,
checks its own labels so no human has to, trains a specialist only on what it verified, and narrates
the operation using only what it can stand behind.

## Setup
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...
# check model IDs in config.py against the Argon builder guide (or set FLASH_MODEL / ARGON_MODEL)
```

## Run order (each step's output is a demo on its own)
| # | Command | Output | Target time |
|---|---|---|---|
| 0 | download zip (see below) | `raw/` | start immediately |
| 1 | `python prepare_data.py --inspect` then `python prepare_data.py` | `data/index.csv` | by 12:15 |
| 2 | `python features.py &` (background) | `data/features.npy` | runs alongside |
| 3 | `python med_school.py --video <narrated lap chole YouTube URL>` | `out/manual.json` | by 13:00 |
| 4a | `python label.py --limit 40` (smoke test), then `python label.py` | `out/labels_manual.jsonl` | by 14:00 |
| 4b | `python label.py --mode blank --splits test` | blank baseline | parallel |
| 4c | `python label.py --votes 2 --splits train,test` | vote baseline | optional |
| 5 | `python accept.py` | `out/decisions.csv`, `out/results.json` | by 14:15 |
| 6 | `python student.py` | `out/student.json` | by 15:15 |
| 7 | `python atlas.py --procedure "laparoscopic appendectomy" --video <URL>` | `out/atlas/...` | any time |
| 8 | `python narrate.py --video <test video>` | `out/narration_*.json` | stretch, 16:00 |
| 9 | `streamlit run app.py` | demo | 15:15 onward |

Download:
```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('minwoosun/CholecSeg8k','data/CholecSeg8k.zip',repo_type='dataset',local_dir='raw')"
```

## Honest claims
- "No human labels used in training." Ground truth is used only to evaluate on held-out test videos.
- The `calib` split and `oracle_upper_bound` student use human labels and are reported as upper bounds only.
- Act 1 (any surgery) has no ground truth: showcase, not proof.

## Prior work to name on stage
SurgVLP (learning from surgical lectures, CAMMA Strasbourg), SurgSLOT (self-evolving data engine with
human verification), Med-Gemini (zero-shot CVS on surgical video). Our step: an agent replaces the
human verifier, and verified labels train a better specialist.

## If you are short on Gemini quota
Lower `FRAME_STRIDE` sampling (raise it to 16 for ~500 frames), skip 4c, and cap `--argon-budget`.
Everything is cached in `cache.sqlite`, so reruns are free.
