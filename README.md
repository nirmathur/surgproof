<div align="center">

# SurgProof

### A local research tool that drafts labels and masks for laparoscopic cholecystectomy video, for a person to review.

![Code licence](https://img.shields.io/badge/code-MIT-lightgrey?style=flat-square)

</div>

---

## Scope and limits

**Read this first.**

- **One procedure.** Laparoscopic cholecystectomy only. Other procedures are out of scope, and footage that does not look like this procedure should fail rather than produce labels.
- **Public or simulator footage only. No real patient video.** Frames are sent to a third-party model (Anthropic, through Claude Code).
- **One user, on one machine.** It runs on 127.0.0.1. It is not hardened for a network, for several users or for production.
- **Research only.** It is not a medical device and must not inform patient care. Nothing here is a clinical assessment.
- **Everything the AI makes is a draft.** Masks are rough. Narration is an AI draft restricted to what the labels say. Propagated frames are estimates, and most have not been seen by anyone.
- **A person's sign-off covers only what that person saw.** The final pass reviews amber labels plus an audit sample, not the whole video.
- **Small evidence.** See the results below: few videos, wide intervals, and some labels that could not be measured.

## How it works

1. **Ingest.** Your video is normalised to frames, and keyframes are taken once a second.
2. **Detect.** The model reports presence and an outline for gallbladder, liver, grasper, L-hook, cystic duct and blood on each keyframe.
3. **Second pass.** A different model looks again without seeing the first answer. It also names the surgical phase.
4. **Amber.** A label stays amber when the passes disagree or are unsure. A tighter crop can accept a label as present, but it can never settle one as absent: that, and anything left unsure, goes to a person.
5. **Masks and propagation.** Accepted outlines become draft masks, carried to the frames in between by optical flow.
6. **Narration.** Subtitles are checked sentence by sentence against the labels. A sentence that fails is dropped.
7. **Final pass.** A person clears the ambers and an audit sample drawn from accepted labels, AI rejects and propagated frames. Too many audit failures block the release until everything has been reviewed.

## Results

Real model runs, scored by `surgproof/eval.py` against CholecSeg8k ground truth that the pipeline never reads. The clips had not been scored before, and one keyframe per second was used. The numbers are in [docs/e2e_status/qa_results.json](docs/e2e_status/qa_results.json).

| Held-out set | videos / clips / keyframes | Final accuracy, mean over videos | 95% CI over videos | labels decided | coverage | left amber | Pooled on the same labels: final | detect alone | second pass alone |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| Videos never scored before (primary) | 3 / 10 / 36 | 0.932 | 0.785 to 1.000 | 183 | 0.847 | 33 | 0.929 | 0.934 | 0.940 |
| All unused clips, all 6 held-out videos | 6 / 21 / 74 | 0.957 | 0.906 to 1.000 | 351 | 0.790 | 93 | 0.952 | 0.952 | 0.943 |

**What that does and does not show:**

- **Accuracy counts only the labels the system decided.** About a fifth went to a person instead (coverage above).
- **The double-check did not beat one pass.** On the same labels, final accuracy is no better than detect alone. What it adds is routing doubtful labels to a person.
- **Cystic duct does not work.** The held-out set has no keyframe with a cystic duct. On the one video that has them (video01, train split), both passes missed all of them: 0 of 11 found by detect, 0 of 11 by the second pass.
- **Blood is barely measured.** The held-out set has no blood, so its sensitivity is unknown there. There were no false positives on 74 held-out keyframes. On video01, a single video, detect found 26 of 27 and the second pass 27 of 27. One video gives no confidence interval.
- **Phase accuracy is not measured.** There are no phase labels for these clips on this machine. The two passes agreed on the phase on only 23 of 74 held-out keyframes, so treat every phase shown as a guess.
- **Accepted by the amber loop is much weaker than accepted by both passes.** On all 6 videos, labels both passes called present were right 141 of 141 times; labels accepted only after the amber loop were right 33 of 42. Every false accept came from that route.
- **Masks are rough.** Mean IoU counting errors as 0 is 0.555 over 6 videos (95% CI 0.325 to 0.785), and 0.198 on the worst video.
- **Examined clips are left out, and one of them is the hardest known.** On video12_19500 and video12_19580, the system decided only 41.7% of labels (0.933 accurate on those, n=15), with mask IoU 0.221. Expect footage like that, not the table above, on a bad day.
- **Six videos is very few.** Keyframes one second apart inside one operation are strongly correlated, and nothing here speaks to other surgeons, hospitals or cameras. Rows overlap: the 6-video row contains the primary row, plus unused clips from three videos that were examined earlier.

## Run it

You need Python 3.10+, `ffmpeg`, and [Claude Code](https://claude.com/claude-code) logged in once (run `claude`). There is no API key: the pipeline calls headless Claude Code.

```bash
git clone https://github.com/nirmathur/surgproof && cd surgproof
make setup                       # private .venv with the app's dependencies
make serve                       # http://127.0.0.1:8600, this machine only
```

Drop a laparoscopic cholecystectomy video (mp4 or mov, up to 5 minutes, public or simulator footage only) on the Upload page. Frames are sent to a third-party model, so never upload real patient video. The job page shows each stage as it runs, then offers **Watch** (the video with the AI's draft boxes, masks, phase and narration), **Final pass** (a person clears what is left amber plus an audit sample) and **Download** (annotated MP4, COCO JSON, provenance). The COCO dataset stays locked until the final pass is finished and released.

On first run, `make serve` prints a one-time setup code to create your local account. The Upload page then asks you to accept the notice and state where the footage came from.

Model usage is real money or plan credit: about $0.06 per keyframe across the runs above, and a job stops at a cost cap of $25 unless you set `SURGPROOF_MAX_COST_USD`. A 5-minute video at one keyframe per second is an estimated $17, and has not been run for real. Leave disk space too: about 1 to 3 MB per second of video.

The same pipeline from the command line:

```bash
.venv/bin/python -m surgproof.run path/to/video.mp4 --out jobs/my_video
```

<details>
<summary><b>Score it against ground truth</b></summary>

Only videos assembled from CholecSeg8k clips have ground truth. The scorer is separate from the pipeline, and the pipeline never reads ground truth. This example uses the calibration split, because scoring a held-out clip spends it.

```bash
.venv/bin/python -m pip install huggingface_hub
.venv/bin/python -c "from huggingface_hub import hf_hub_download; hf_hub_download('minwoosun/CholecSeg8k','data/CholecSeg8k.zip',repo_type='dataset',local_dir='raw')"
(cd raw && unzip -q data/CholecSeg8k.zip)
.venv/bin/python tools/make_test_video.py --split calib --video video27 --seconds 6 --out test_videos/calib27.mp4
.venv/bin/python -m surgproof.run test_videos/calib27.mp4 --out jobs/calib27
.venv/bin/python -m surgproof.eval jobs/calib27
```

It prints presence accuracy and mask IoU with n, and writes `jobs/calib27/eval_report.json`. `python -m surgproof.eval aggregate` pools scored jobs by video, with intervals over videos. It refuses to score a video with no registered ground truth, keyframes that do not match the frames they claim to be, or a mock run.

</details>

<details>
<summary><b>Tests, without calling the model</b></summary>

```bash
SURGPROOF_VLM=mock .venv/bin/python -m surgproof.run path/to/video.mp4 --out jobs/mock_run   # plumbing only: every row is tagged mock
.venv/bin/python -m pytest -q tests/test_e2e.py tests/test_qa_eval.py tests/test_qa_make_test_video.py tests/test_qa_redteam.py tests/test_qa_http_redteam.py tests/test_qa_hardening.py tests/test_qa_inputs.py
SURGPROOF_E2E_REAL=1 .venv/bin/python -m pytest -q -s tests/test_e2e.py -k real              # the real thing: spends model credit
```

</details>

<details>
<summary><b>The earlier research pipeline</b></summary>

This is a separate, earlier Gemini 4 Argon pipeline. Nothing in the results above came from it, and no number from it applies to the app above.

```bash
make setup-full                  # adds torch, streamlit and more; copies .env.example to .env (add your GEMINI_API_KEY)
python3 web/export.py            # builds the toolbox data from out/
PORT=8601 python3 web/serve.py   # toolbox at http://localhost:8601, Watch mode at /watch.html
```

`prepare_data.py` → `med_school.py` → `label.py` → `accept.py` → `tools/vote_worker.py` → `tools/refine_worker.py` → `tools/dense_worker.py` → `tools/smooth_dense.py`, then score with `tools/score_dense.py` and `eval/`.

</details>

## Where it goes next

More held-out videos with blood and cystic duct, phase ground truth so phase can be scored at all, and better masks.

## Prior work

SurgVLP, SurgSLOT, Med-Gemini and the SurgAtlas public video dataset.

---

<sub>Data: CholecSeg8k, Endoscapes2023 and CholecT50 (CC BY-NC-SA 4.0), not redistributed here. Code: MIT.</sub>
