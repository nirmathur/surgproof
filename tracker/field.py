"""Step 2: visible field per video. Per-pixel percentile of sampled frames (50 = median), threshold the dark
border, keep the largest blob, fill it (convex hull: the endoscope field is convex)."""
import cv2
import numpy as np

from tracker.common import ROOT, Timer, cli, load_cfg, load_frames, p, skip, work


def estimate(paths, cfg):
    c = cfg["field"]
    imgs = [cv2.imread(str(ROOT / q), cv2.IMREAD_GRAYSCALE) for q in paths]
    med = np.percentile(np.stack(imgs), c["percentile"], axis=0).astype(np.uint8)
    m = (med > c["dark_thresh"]).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c["morph_kernel"],) * 2)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        m = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
    # fill holes (dark instruments / shadows inside the field)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(np.concatenate(cs)) if c["convex_hull"] else None
    m = cv2.drawContours(np.zeros_like(m), [hull] if hull is not None else cs, -1, 1, thickness=-1)
    return m, med


def run(cfg, force=False):
    fr = load_frames(cfg)
    rng = np.random.default_rng(cfg["seed"])
    sheet = []
    for v, g in fr.groupby("video_id"):
        out = work(cfg, f"field/{v}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not force:
            m = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE) > 0
        else:
            idx = np.sort(rng.choice(len(g), min(cfg["field"]["n_sample_frames"], len(g)), replace=False))
            m, med = estimate(g.image_path.values[idx], cfg)
            cv2.imwrite(str(out), m * 255)
            m = m > 0
        img = cv2.imread(str(ROOT / g.image_path.values[len(g) // 2]))
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, (0, 255, 255), 3)
        cv2.putText(img, f"{v} field={m.mean():.0%}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        sheet.append(cv2.resize(img, (427, 240)))
        print(f"{v}: visible field {m.mean():.1%} of frame")
    while len(sheet) % 6:
        sheet.append(np.zeros_like(sheet[0]))
    rows = [np.concatenate(sheet[i:i + 6], 1) for i in range(0, len(sheet), 6)]
    cv2.imwrite(str(p(cfg, "checks", "visible_field.png")), np.concatenate(rows, 0))


if __name__ == "__main__":
    a = cli("Visible field per video", source=False).parse_args()
    with Timer("field"):
        run(load_cfg(), a.force)
