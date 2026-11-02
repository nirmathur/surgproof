"""The five pages in a real browser, on a synthetic finished job plus a real upload through the Upload page.

Uses installed Google Chrome, because Playwright's bundled Chromium cannot decode H.264. Skips where the
browser, or binding a local port, is not available (some sandboxes refuse both).
"""
from __future__ import annotations

import shutil
import socket
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("cv2")
playwright_api = pytest.importorskip("playwright.sync_api")
if not shutil.which("ffmpeg"):
    pytest.skip("ffmpeg is required", allow_module_level=True)

import uvicorn  # noqa: E402

from surgproof import server  # noqa: E402
from tests import web_fixture as fx  # noqa: E402


def free_port() -> int:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", 0))
        except OSError as exc:
            pytest.skip(f"cannot bind a local port here: {exc}")
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    pytest.importorskip("surgproof.export")
    jobs = tmp_path_factory.mktemp("browser") / "jobs"
    fx.finished_job(jobs)
    port = free_port()
    config = uvicorn.Config(server.create_app(jobs, pipeline_target="tests.web_fixture:pipeline_boom"),
                            host="127.0.0.1", port=port, log_level="warning")
    instance = uvicorn.Server(config)
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    end = time.time() + 60
    while not instance.started and time.time() < end:
        time.sleep(0.1)
    assert instance.started
    yield f"http://127.0.0.1:{port}", jobs
    instance.should_exit = True
    thread.join(timeout=30)


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as pw:
        try:
            instance = pw.chromium.launch(channel="chrome", headless=True)
        except Exception as exc:
            pytest.skip(f"could not start Chrome: {exc}")
        yield instance
        instance.close()


@pytest.fixture
def page(browser):
    context = browser.new_context(viewport={"width": 1280, "height": 900}, accept_downloads=True)
    page = context.new_page()
    page.errors = []
    page.on("pageerror", lambda err: page.errors.append(str(err)))
    yield page
    context.close()
    assert not page.errors, f"JavaScript errors: {page.errors}"


def canvas_has_pixels(page, selector: str) -> bool:
    return page.evaluate(
        """(sel) => { const c = document.querySelector(sel);
        const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        for (let i = 3; i < d.length; i += 4) if (d[i]) return true; return false; }""", selector)


def test_upload_page_rejects_with_a_message_then_opens_the_job(site, page, tmp_path):
    base, _ = site
    page.goto(base)
    page.wait_for_selector("#drop")
    bad = tmp_path / "notes.txt"
    bad.write_text("x")
    page.set_input_files("#file-input", str(bad))
    assert "not an mp4" in page.locator("[role=alert]").inner_text()

    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"not a video" * 50)
    page.set_input_files("#file-input", str(fake))
    page.wait_for_function("document.querySelector('[role=alert]')?.textContent.includes('could not be read')")

    video = fx.make_video(tmp_path / "my clip.mp4")
    page.set_input_files("#file-input", str(video))
    page.wait_for_url("**/#/jobs/*")
    page.wait_for_selector("#stages li")
    assert page.locator("#stages li").count() == 9
    assert page.locator("h1").inner_text() == "my clip.mp4"
    # This site's pipeline stand-in always fails, so the failure path must be visible and offer a retry.
    page.wait_for_selector("[role=alert] button", timeout=180000)
    assert "synthetic failure" in page.locator("[role=alert]").inner_text()


def test_job_page_links_every_output(site, page):
    base, _ = site
    page.goto(f"{base}/#/jobs/job-done")
    page.wait_for_selector("#stages li")
    assert page.locator("#stages .dot.done").count() == 9
    assert page.locator(".tab.off").count() == 0
    assert "2 objects outlined" in page.locator("section.stack p.small").last.inner_text()


def test_watch_plays_with_overlays_phase_and_subtitles(site, page):
    base, _ = site
    page.goto(f"{base}/#/jobs/job-done/watch")
    page.wait_for_selector("#watch-video")
    page.wait_for_function("document.querySelector('#watch-video').readyState >= 2", timeout=60000)
    assert page.evaluate("document.querySelector('#watch-video').videoWidth") == fx.W, "the video did not decode"
    page.wait_for_function("document.querySelector('#watch-frame').textContent.includes('frame 0')")
    assert canvas_has_pixels(page, "#watch-canvas")
    assert page.locator("#watch-phase").inner_text() == "Preparation · 90%"
    assert page.locator("#watch-subtitle").inner_text() == "The grasper is visible."

    # Seek to a propagated frame in the last segment: every overlay must follow.
    page.evaluate("document.querySelector('#watch-video').currentTime = 2.5")
    page.wait_for_function("document.querySelector('#watch-frame').textContent.includes('frame 25')")
    assert "propagated" in page.locator("#watch-frame").inner_text()
    assert page.locator("#watch-phase").inner_text().startswith("Clipping cutting")
    assert page.locator("#watch-subtitle").inner_text() == "Unsure about the duct here."

    # The masks and boxes toggles really clear the canvas.
    for label in ("Masks", "Boxes"):
        page.get_by_label(label).uncheck()
    page.wait_for_function(
        """() => { const c = document.querySelector('#watch-canvas');
        return !c.getContext('2d').getImageData(0, 0, c.width, c.height).data.some((v, i) => i % 4 === 3 && v); }""")

    # Plays forward without error, and the overlay frame counter advances with it.
    page.get_by_label("Boxes").check()
    page.evaluate("const v = document.querySelector('#watch-video'); v.muted = true; v.currentTime = 0; v.play()")
    page.wait_for_function("document.querySelector('#watch-video').currentTime > 0.5", timeout=30000)
    assert page.locator("#watch-frame").inner_text() != ""


def test_final_pass_then_release_then_download(site, page):
    base, jobs = site
    page.goto(f"{base}/#/jobs/job-done/download")
    page.wait_for_selector(".dl")
    assert page.locator(".dl").nth(1).locator("button").inner_text() == "Locked"

    page.goto(f"{base}/#/jobs/job-done/final-pass")
    page.wait_for_selector("#fp-queue .fpitem")
    assert page.locator("#fp-release-btn").is_disabled()
    assert page.locator("#fp-queue .fpitem .pill.AMBER").count() == 1
    page.wait_for_function(f"document.querySelector('#fp-canvas').width === {fx.W}")

    # First up is the amber. Correct it by drawing, which goes through real canvas clicks.
    assert "kf_00001 · Gallbladder" in page.locator(".fpitem.on").inner_text()
    page.click("#fp-draw")
    box = page.locator("#fp-canvas").bounding_box()
    for fx_, fy in ((0.1, 0.1), (0.5, 0.1), (0.5, 0.6)):
        page.mouse.click(box["x"] + box["width"] * fx_, box["y"] + box["height"] * fy)
    page.click("#fp-save-draw")
    page.wait_for_function("document.querySelector('#fp-progress').textContent.startsWith('1 of')")

    # Keyboard approve for all but the last, which is rejected: audit disagreements must show.
    total = page.locator("#fp-queue .fpitem").count()
    ready = "!document.querySelector('#fp-approve').disabled"
    for done in range(1, total - 1):
        page.wait_for_function(ready)
        page.keyboard.press("a")
        page.wait_for_function("(n) => document.querySelector('#fp-progress').textContent.startsWith(`${n} of`)", arg=done + 1)
    assert page.locator("#fp-release-btn").is_disabled(), "release must stay locked with one item left"
    page.wait_for_function(ready)
    page.keyboard.press("x")
    page.wait_for_function("(n) => document.querySelector('#fp-progress').textContent.startsWith(`${n} of`)", arg=total)
    assert "1 corrected by you" in page.locator("#fp-release").inner_text()

    assert page.locator("#fp-release-btn").is_disabled(), "release needs a named reviewer"
    page.fill("#fp-reviewer", "Test Reviewer")
    assert page.locator("#fp-release-btn").is_enabled()
    page.click("#fp-release-btn")
    page.wait_for_function("document.querySelector('#fp-release-btn').textContent === 'Released'")
    assert (jobs / "job-done" / "released_coco.json").exists()
    assert page.locator("#fp-approve").is_disabled()

    page.goto(f"{base}/#/jobs/job-done/download")
    page.wait_for_selector(".dl a.btn.primary")
    assert page.locator(".dl a.btn.primary").count() == 3
    with page.expect_download() as download_info:
        page.locator(".dl").nth(1).locator("a").click()
    assert download_info.value.suggested_filename == "job-done_released_coco.json"


def test_no_verdict_before_the_frame_is_on_screen(site, browser):
    base, jobs = site
    job = jobs / "job-slow"
    fx.finished_job(jobs, "job-slow")
    context = browser.new_context(viewport={"width": 1280, "height": 900})
    page = context.new_page()
    held = []  # keep the frame request open: no blocking here, Playwright's sync API is single-threaded
    page.route("**/files/frames/**", held.append)
    page.goto(f"{base}/#/jobs/job-slow/final-pass")
    page.wait_for_selector("#fp-approve")
    for selector in ("#fp-approve", "#fp-reject", "#fp-draw"):
        assert page.locator(selector).is_disabled(), selector
    for key in ("a", "x", "d"):
        page.keyboard.press(key)
    assert not (job / "final_pass.jsonl").exists()
    page.wait_for_timeout(200)
    assert held, "the frame request should be in flight"
    page.unroute("**/files/frames/**")
    for route in held:
        route.continue_()
    page.wait_for_function("!document.querySelector('#fp-approve').disabled", timeout=30000)
    assert page.evaluate("document.querySelector('#fp-canvas').width") == fx.W

    # Points land in video pixels, wherever the canvas sits on screen.
    page.click("#fp-draw")
    box = page.locator("#fp-canvas").bounding_box()
    for x, y in ((32, 18), (160, 18), (160, 90)):
        page.mouse.click(box["x"] + box["width"] * x / fx.W, box["y"] + box["height"] * y / fx.H)
    page.click("#fp-save-draw")
    page.wait_for_function("document.querySelector('#fp-progress').textContent.startsWith('1 of')")
    row = server.json.loads((job / "final_pass.jsonl").read_text().splitlines()[0])
    assert row["action"] == "correct"
    for (got_x, got_y), (x, y) in zip(row["polygon"], ((32, 18), (160, 18), (160, 90))):
        assert abs(got_x - x) <= 1.5 and abs(got_y - y) <= 1.5
    context.close()


def test_filename_with_markup_is_not_injected(site, page):
    base, jobs = site
    job = jobs / "job-html"
    job.mkdir()
    payload = '<img src=x onerror="window.pwned=1">.mp4'
    (job / "job.json").write_text(server.json.dumps({"id": "job-html", "filename": payload}), encoding="utf-8")
    (job / "status.json").write_text(server.json.dumps({"stage": "detect", "error": None}), encoding="utf-8")
    page.goto(f"{base}/#/jobs/job-html")
    page.wait_for_selector("h1")
    assert page.locator("h1").inner_text() == payload
    assert page.evaluate("window.pwned") is None


def test_phone_width_has_no_sideways_scroll(site, browser):
    base, _ = site
    context = browser.new_context(viewport={"width": 390, "height": 800})
    page = context.new_page()
    for route in ("", "#/jobs/job-done", "#/jobs/job-done/watch", "#/jobs/job-done/final-pass", "#/jobs/job-done/download"):
        page.goto(f"{base}/{route}")
        page.wait_for_timeout(1000)
        overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        assert overflow <= 0, f"{route or 'upload'} overflows by {overflow}px"
    context.close()
