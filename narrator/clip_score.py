"""Non-Gemini zero-shot scorer (OpenCLIP). Runs ONLY in narrator/.venv-clip (own torch; the shared venv is untouched).

narrator/.venv-clip/bin/python narrator/clip_score.py <in.json> <out.json>
in:  [{"key": str, "path": str, "cls": str}]   out: {key: {"rank": int, "probs": {cls: p}}}
Zero-shot only: no thresholds fitted on any labels.
"""
import json
import sys

import open_clip
import torch
from PIL import Image

PROMPTS = {
    "grasper": "a laparoscopic surgery image with a metal grasper forceps instrument",
    "l_hook": "a laparoscopic surgery image with an L-shaped electrocautery hook instrument",
    "liver": "a laparoscopic surgery image showing the dark red liver",
    "gallbladder": "a laparoscopic surgery image showing the gallbladder",
    "cystic_duct": "a laparoscopic surgery image showing the thin tubular cystic duct",
    "blood": "a laparoscopic surgery image with bleeding and blood",
    "background": "a laparoscopic surgery image of fat and connective tissue",
}


def main(inp, outp):
    items = json.load(open(inp))
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    names = list(PROMPTS)
    with torch.no_grad():
        t = model.encode_text(tok([PROMPTS[n] for n in names]))
        t = t / t.norm(dim=-1, keepdim=True)
        cache, out = {}, {}
        for it in items:
            if it["path"] not in cache:
                im = pre(Image.open(it["path"]).convert("RGB")).unsqueeze(0)
                f = model.encode_image(im)
                f = f / f.norm(dim=-1, keepdim=True)
                cache[it["path"]] = (100.0 * f @ t.T).softmax(dim=-1)[0].tolist()
            p = dict(zip(names, cache[it["path"]]))
            order = sorted(names, key=lambda n: -p[n])
            out[it["key"]] = {"rank": order.index(it["cls"]) + 1, "probs": p}
    json.dump(out, open(outp, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
