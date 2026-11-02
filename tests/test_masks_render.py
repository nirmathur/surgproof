"""Stage 8: the output must match the source frame for frame, and only draw what the job's files say."""
import json, subprocess

import cv2
import numpy as np
import pytest

from surgproof import masks as M, propagate as P, render as R

W, H = 320, 240
FPS = 25


def ffprobe(path, entries):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          f"stream={entries}", "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


def make_source(path, n=30, audio=False, fps=FPS):
    args = ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate={fps}:duration={n / fps}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={n / fps}", "-c:a", "aac"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(args, check=True)


def make_job(tmp_path, n=30, audio=False, narration=True, phases=True, status="accept", fps=FPS):
    job = tmp_path / "job"
    (job / "frames").mkdir(parents=True)
    make_source(job / "source.mp4", n, audio, fps)
    cap = cv2.VideoCapture(str(job / "source.mp4"))
    ok, first = cap.read(); cap.release(); assert ok
    cv2.imwrite(str(job / "frames" / "kf_00000.jpg"), first)
    m = np.zeros((H, W), bool); m[60:180, 80:240] = True
    M.write_mask(M.mask_path(job, "kf_00000", "grasper"), m)
    (job / "meta.json").write_text(json.dumps({"fps": fps, "width": W, "height": H, "n_frames": n, "kf_fps": 1,
                                               "keyframes": [{"kf": "kf_00000", "frame_index": 0, "t": 0.0}]}))
    (job / "masks.jsonl").write_text(json.dumps({"kf": "kf_00000", "label": "grasper", "status": status, "path": "masks/kf_00000_grasper.png", "box": M.mask_box(m)}) + "\n")
    if phases:
        (job / "phase.jsonl").write_text(json.dumps({"kf": "kf_00000", "phase": "CalotTriangleDissection", "conf": 0.91}) + "\n")
    if narration:
        (job / "narration.json").write_text(json.dumps({"segments": [
            {"t0": 0.0, "t1": 1.0, "phase": "CalotTriangleDissection", "text": "A grasper is visible. It holds tissue.", "evidence": ["kf_00000"]}]}))
    P.run(job)
    return job


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    job = make_job(tmp_path_factory.mktemp("r"))
    return job, R.run(job)


def test_output_matches_source_frame_count_and_fps(rendered):
    job, res = rendered
    s = ffprobe(job / "annotated.mp4", "nb_read_frames,r_frame_rate,width,height,pix_fmt,codec_name")
    assert int(s["nb_read_frames"]) == 30 == res["n_frames"]
    assert s["r_frame_rate"] == "25/1"
    assert s["codec_name"] == "h264" and s["pix_fmt"] == "yuv420p"  # plays in a browser
    assert int(s["width"]) % 2 == 0 and int(s["height"]) % 2 == 0


def test_fractional_fps_is_preserved(tmp_path):
    job = make_job(tmp_path, n=12, fps=30000 / 1001, narration=False)
    R.run(job)
    assert ffprobe(job / "annotated.mp4", "r_frame_rate")["r_frame_rate"] == "30000/1001"


def test_annotation_changes_pixels_only_where_a_mask_exists(tmp_path):
    job = make_job(tmp_path, narration=False, phases=False)
    R.run(job)
    lay = R.Layout(W, H)
    cap = cv2.VideoCapture(str(job / "annotated.mp4")); ok, out = cap.read(); cap.release(); assert ok
    src = cv2.resize(cv2.imread(str(job / "frames" / "kf_00000.jpg")), (lay.fw, lay.fh))
    view = out[lay.oy:lay.oy + lay.fh, lay.ox:lay.ox + lay.fw].astype(int)
    diff = np.abs(view - src).mean(axis=2)
    inside = diff[int(70 * lay.sy):int(170 * lay.sy), int(90 * lay.sx):int(230 * lay.sx)]
    outside = diff[int(200 * lay.sy):, :]
    assert inside.mean() > outside.mean() + 5


def test_masks_can_be_turned_off(tmp_path):
    job = make_job(tmp_path, narration=False, phases=False)
    R.run(job, show_masks=False)
    lay = R.Layout(W, H)
    cap = cv2.VideoCapture(str(job / "annotated.mp4")); _, out = cap.read(); cap.release()
    src = cv2.resize(cv2.imread(str(job / "frames" / "kf_00000.jpg")), (lay.fw, lay.fh))
    view = out[lay.oy:lay.oy + lay.fh, lay.ox:lay.ox + lay.fw].astype(int)
    assert np.abs(view - src).mean(axis=2)[int(100 * lay.sy):int(160 * lay.sy), int(120 * lay.sx):int(200 * lay.sx)].mean() < 15


def test_source_audio_is_kept(tmp_path):
    job = make_job(tmp_path, audio=True, narration=False)
    R.run(job)
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name", "-of", "csv=p=0",
                          str(job / "annotated.mp4")], capture_output=True, text=True).stdout
    assert out.strip().startswith("aac")


def test_no_audio_source_still_renders(rendered):
    job, _ = rendered
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name", "-of", "csv=p=0",
                          str(job / "annotated.mp4")], capture_output=True, text=True).stdout
    assert out.strip() == ""


def test_amber_label_is_visibly_different_from_accepted(tmp_path):
    a = make_job(tmp_path / "a", narration=False, phases=False, status="accept"); R.run(a, show_masks=False)
    b = make_job(tmp_path / "b", narration=False, phases=False, status="AMBER"); R.run(b, show_masks=False)
    read = lambda j: cv2.VideoCapture(str(j / "annotated.mp4")).read()[1].astype(int)
    assert np.abs(read(a) - read(b)).mean() > 0.01
    rows = M.read_jsonl(b / "tracks.jsonl")
    assert {r["status"] for r in rows} == {"AMBER"}


def test_chip_says_unsure_for_amber_via_sprite_cache():
    lay = R.Layout(W, H); sprites = R.Sprites()
    frame = np.zeros((H, W, 3), np.uint8)
    row = {"frame_index": 1, "label": "liver", "status": "AMBER", "propagated": True, "box": [10, 10, 100, 100], "keyframe": None}

    class NoFile:
        def mask_file(self, r):
            return "/nonexistent"
    R.draw_frame(frame, [row], None, None, 1, 0.04, NoFile(), lay, sprites, True, "note")
    assert any("unsure" in k[0] for k in sprites.cache)


def test_missing_narration_and_phase_still_render(tmp_path):
    job = make_job(tmp_path, narration=False, phases=False)
    res = R.run(job)
    assert res["n_cues"] == 0
    assert (job / "narration.vtt").read_text().startswith("WEBVTT")


def test_vtt_cues_come_from_narration_only(rendered):
    job, res = rendered
    vtt = (job / "narration.vtt").read_text()
    assert res["n_cues"] == 2
    assert "A grasper is visible." in vtt and "It holds tissue." in vtt
    assert "00:00:01.000" in vtt


def test_build_cues_shares_time_and_splits_long_text():
    text = "Short. " + "word " * 80
    cues = R.build_cues({"segments": [{"t0": 2.0, "t1": 12.0, "text": text}]})
    assert cues[0][0] == 2.0 and abs(cues[-1][1] - 12.0) < 0.01
    assert all(len(R.wrap_line(c[2])) <= 2 for c in cues)
    # Nothing is lost in splitting.
    assert " ".join(c[2] for c in cues).split() == text.split()
    assert all(a[1] == pytest.approx(b[0], abs=0.01) for a, b in zip(cues, cues[1:]))


def test_empty_or_inverted_segments_give_no_cues():
    assert R.build_cues(None) == []
    assert R.build_cues({"segments": [{"t0": 5, "t1": 5, "text": "x"}, {"t0": 1, "t1": 2, "text": ""}]}) == []


def test_phase_is_nearest_keyframe_and_prettified():
    assert R.prettify_phase("CalotTriangleDissection") == "Calot triangle dissection"
    assert R.prettify_phase("Preparation") == "Preparation"

    class A(R.Annotations):
        def __init__(self):
            self.phases = [(0, "A", 0.5), (25, "B", 0.6), (50, "C", None)]
            self.phase_idx = [0, 25, 50]
    a = A()
    assert a.phase_at(10)[1] == "A" and a.phase_at(20)[1] == "B" and a.phase_at(49)[1] == "C" and a.phase_at(500)[1] == "C"


def test_layout_is_even_and_keeps_aspect_for_portrait_and_hd():
    for w, h in [(854, 480), (1920, 1080), (720, 1280), (3840, 2160), (640, 480)]:
        lay = R.Layout(w, h)
        assert lay.fw % 2 == 0 and lay.fh % 2 == 0
        assert lay.ox >= 0 and lay.oy >= lay.band and lay.oy + lay.fh <= lay.H - lay.band
        assert abs(lay.fw / lay.fh - w / h) < 0.02


def test_failure_leaves_no_partial_annotated_file(tmp_path, monkeypatch):
    job = make_job(tmp_path, narration=False)
    monkeypatch.setattr(R, "count_video_frames", lambda p: 1)
    with pytest.raises(RuntimeError, match="frames"):
        R.run(job)
    assert not (job / "annotated.mp4").exists() and not (job / ".annotated.tmp.mp4").exists()


def test_footer_reflects_release_state(tmp_path):
    job = make_job(tmp_path, narration=False, phases=False)
    R.run(job)
    before = cv2.VideoCapture(str(job / "annotated.mp4")).read()[1].astype(int)
    (job / "released_coco.json").write_text("{}")
    R.run(job)
    after = cv2.VideoCapture(str(job / "annotated.mp4")).read()[1].astype(int)
    lay = R.Layout(W, H)
    assert np.abs(before[lay.H - lay.band:] - after[lay.H - lay.band:]).mean() > 0.01
