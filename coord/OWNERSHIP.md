# Who owns what (updated 31 Oct: Harald owns the verifier)

Split rule: anything that needs the CholecSeg8k dataset or the shared Gemini cache stays on Nir's machine. Anything that only needs small JSON contracts goes to Harald, so his machine never needs the dataset.

| Lane | Machine | Files | Needs dataset? |
|---|---|---|---|
| architect (integrator, DESIGN.md, deep research) | nir | config.py, gem.py, tools/, docs/DESIGN.md, merges to main | no |
| data | nir | prepare_data.py, features.py | yes |
| manual (lap chole teaching) | nir | med_school.py | no, but feeds label |
| label (labeller: runs Flash over frames, caching, blank/vote baselines) | nir | label.py | yes |
| verifier (zero-label acceptance loop: checks, critic, abstention -> decisions.csv) | harald | accept.py, verify/ | yes: download CholecSeg8k from HF himself (README command) |
| student (specialist training) | nir | student.py, mock_test.py | yes |
| demo (Streamlit app) | harald | app.py | no: builds on coord/artifacts + DESIGN.md |
| narrate (grounded narration) | harald | narrate.py | no: any public video + manual.json |
| atlas (Act 1: any surgery) | harald | atlas.py | no: public video only |
| eval (stats, leakage audit, video-level CIs) | nir (student lane) | eval/ | yes. Kept off Harald's machine so the verifier is not graded by its author |

Branches: Nir's lanes push lane/<lane>; Harald's push harald/<lane>. Only nir-architect merges into main.
