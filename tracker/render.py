"""Step 9: overlay renderer. Boxes, track ids, active flags, visible field and camera motion for one clip.

  python -m tracker.render --clip video12_15750 [--source gt_corrupted] [--fmt mp4|gif] [--errors]
"""
import json

import cv2
import numpy as np
import pandas as pd

from tracker.common import ROOT, cli, field_mask, load_cfg, load_frames, p, work

COL = {"grasper": (0, 220, 0), "l_hook": (255, 0, 255)}  # BGR


def active_flags(row, cfg):
    s = cfg["signals"]
    fl = [k for k in s["zscore"] if k not in s["quality"] and row.get(f"z_{k}", 0) > s["z_thresh"]]
    fl += [k for k in s["raw"] if row.get(k, 0) > s["binary_thresh"]]
    return fl


def render(cfg, clip, source, fmt, show_errors, scale=0.75):
    frames = load_frames(cfg)
    fr = frames[frames.clip_id == clip].sort_values("frame_idx")
    if fr.empty:
        raise SystemExit(f"unknown clip {clip}")
    video = fr.video_id.iloc[0]
    tracks = pd.read_parquet(work(cfg, f"tracks_{source}.parquet"))
    tracks = tracks[tracks.clip_id == clip]
    sig = pd.read_parquet(work(cfg, f"signals_{source}.parquet"))
    sig = sig[sig.clip_id == clip].set_index(["frame_idx", "label"])
    sus = None
    fp = work(cfg, f"flags_{source}.parquet")
    if fp.exists():
        sus = pd.read_parquet(fp).query("clip_id == @clip").set_index(["frame_idx", "label"])
    cam = pd.read_parquet(work(cfg, f"camera_{source}.parquet")).query("clip_id == @clip").set_index("frame_idx")
    errs = None
    if show_errors and work(cfg, "errors.parquet").exists():
        errs = pd.read_parquet(work(cfg, "errors.parquet")).query("clip_id == @clip")
    cs, _ = cv2.findContours(field_mask(cfg, video).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    out_frames = []
    for _, r in fr.iterrows():
        img = cv2.imread(str(ROOT / r.image_path))
        f = r.frame_idx
        cv2.drawContours(img, cs, -1, (0, 200, 255), 1)
        for _, d in tracks[tracks.frame_idx == f].iterrows():
            c = COL.get(d.label, (255, 255, 255))
            row = sig.loc[(f, d.label)].to_dict() if (f, d.label) in sig.index else {}
            fl = active_flags(row, cfg)
            cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), c, 3 if fl else 2)
            txt = f"{d.label} T{d.track_id}"
            if sus is not None and (f, d.label) in sus.index:
                txt += f" s={sus.loc[(f, d.label), 'suspicion']:.2f}"
            y = int(max(d.y1 - 8, 18))
            cv2.putText(img, txt, (int(d.x1), y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, txt, (int(d.x1), y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1, cv2.LINE_AA)
            for k, name in enumerate(fl):
                yy = int(d.y1) + 20 + 18 * k
                cv2.putText(img, name, (int(d.x1) + 4, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, name, (int(d.x1) + 4, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        # flags on rows without a detection (missing tool)
        lines = []
        for lab in cfg["tool_labels"]:
            if (f, lab) in sig.index and sig.loc[(f, lab), "n_det"] == 0:
                fl = active_flags(sig.loc[(f, lab)].to_dict(), cfg)
                if fl:
                    lines.append(f"MISSING {lab}: {','.join(fl)}")
        if errs is not None:
            for _, e in errs[errs.frame_idx == f].iterrows():
                lines.append(f"INJECTED {e.error_type} ({e.label})")
                if pd.notna(e.get("x1", np.nan)):
                    cv2.rectangle(img, (int(e.x1), int(e.y1)), (int(e.x2), int(e.y2)), (0, 255, 255), 1)
        # header: frame, camera motion arrow
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, f"{clip} f={f} src={source}", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if f in cam.index:
            dx, dy = cam.loc[f, "cam_dx"], cam.loc[f, "cam_dy"]
            o = (img.shape[1] - 40, 60)
            cv2.circle(img, o, 30, (80, 80, 80), 1)
            cv2.arrowedLine(img, o, (int(o[0] + 3 * dx), int(o[1] + 3 * dy)), (0, 255, 255), 2, tipLength=0.3)
        for k, t in enumerate(lines):
            cv2.putText(img, t, (8, img.shape[0] - 12 - 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, t, (8, img.shape[0] - 12 - 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        out_frames.append(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))

    out = p(cfg, "render", f"{clip}_{source}.{fmt}")
    if fmt == "mp4":
        h, w = out_frames[0].shape[:2]
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), 8, (w, h))
        if not vw.isOpened():
            vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 8, (w, h))
        for im in out_frames:
            vw.write(im)
        vw.release()
    else:
        from PIL import Image
        ims = [Image.fromarray(cv2.cvtColor(cv2.resize(im, None, fx=0.6, fy=0.6), cv2.COLOR_BGR2RGB)) for im in out_frames]
        ims[0].save(out, save_all=True, append_images=ims[1:], duration=125, loop=0)
    # contact sheet of 8 frames for quick viewing
    pick = np.linspace(0, len(out_frames) - 1, 8).astype(int)
    sheet = [cv2.resize(out_frames[i], None, fx=0.5, fy=0.5) for i in pick]
    sheet = np.concatenate([np.concatenate(sheet[:4], 1), np.concatenate(sheet[4:], 1)], 0)
    cv2.imwrite(str(p(cfg, "render", f"{clip}_{source}_sheet.png")), sheet)
    print("wrote", out)
    return out


if __name__ == "__main__":
    ap = cli("Render overlay for a clip")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--fmt", default="mp4", choices=["mp4", "gif"])
    ap.add_argument("--errors", action="store_true", help="draw injected errors (gt_corrupted)")
    a = ap.parse_args()
    render(load_cfg(), a.clip, a.source, a.fmt, a.errors)
