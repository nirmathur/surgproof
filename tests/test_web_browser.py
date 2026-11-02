"""The five pages in a real browser, on a synthetic finished job plus a real upload through the Upload page.

Uses installed Google Chrome, because Playwright's bundled Chromium cannot decode H.264. Skips where the
browser, or binding a local port, is not available (some sandboxes refuse both).
"""
from __future__ import annotations

import os
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
    os.environ["SURGPROOF_MIN_FREE_MB"] = "1"  # the disk preflight has its own tests
    jobs = tmp_path_factory.mktemp("browser") / "jobs"
    fx.finished_job(jobs)
    port = free_port()
    app = server.create_app(jobs, pipeline_target="tests.web_fixture:pipeline_boom")
    config = uvicorn.Config(app,
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
    page.signed_in_context = context
    page.on("pageerror", lambda err: page.errors.append(str(err)))
    yield page
    context.close()
    assert not page.errors, f"JavaScript errors: {page.errors}"


class HeldRequests:
    """Keep matching requests in flight until release(). Never blocks: Playwright's sync API is single-threaded."""

    def __init__(self, page, pattern):
        self.routes = []
        self.holding = True
        page.route(pattern, self._handle)

    def _handle(self, route):
        if self.holding:
            self.routes.append(route)
        else:
            route.continue_()

    def release(self):
        self.holding = False
        for route in self.routes:
            route.continue_()


def sign_in_browser(context, site):
    """Puts a real session cookie into this browser context."""
    base, jobs = site
    state = context.request.get(f"{base}/api/auth/state").json()
    if state["setup_required"]:
        code = (jobs / ".setup_code").read_text(encoding="utf-8")
        res = context.request.post(f"{base}/api/auth/setup", data={"name": fx.TEST_USER, "password": fx.TEST_PASSWORD, "setup_code": code})
    else:
        res = context.request.post(f"{base}/api/auth/login", data={"password": fx.TEST_PASSWORD})
    assert res.ok, res.text()


def wait_for_annotated_frame(page, frame: int):
    page.wait_for_function(
        """(n) => { const t = document.querySelector('#watch-frame').textContent;
        return t.includes(`frame ${n} of`) && !t.includes('loading'); }""", arg=frame, timeout=60000)


def click_on_canvas(page, fractions):
    """Mouse clicks use viewport coordinates, and clicking a button can scroll the page: bring the canvas back first."""
    canvas = page.locator("#fp-canvas")
    canvas.scroll_into_view_if_needed()
    box = canvas.bounding_box()
    for fx_, fy in fractions:
        page.mouse.click(box["x"] + box["width"] * fx_, box["y"] + box["height"] * fy)


def canvas_has_pixels(page, selector: str) -> bool:
    return page.evaluate(
        """(sel) => { const c = document.querySelector(sel);
        const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        for (let i = 3; i < d.length; i += 4) if (d[i]) return true; return false; }""", selector)


def test_upload_page_rejects_with_a_message_then_opens_the_job(site, page, tmp_path):
    base, _ = site
    sign_in_browser(page.signed_in_context, site)
    page.goto(base)
    page.wait_for_selector("#drop")
    # Blocking: until the notice is answered, nothing can be chosen or dropped.
    assert "third-party" in page.locator("#notice-text").inner_text()
    assert page.locator("#file-input").is_disabled()
    assert "blocked" in page.locator("#drop").get_attribute("class")
    page.check("#attest-rights")
    assert page.locator("#file-input").is_disabled(), "needs a footage source as well"
    page.check("#source-simulator")
    assert page.locator("#file-input").is_enabled()
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
    assert page.locator("h1").inner_text().startswith("Job ")
    assert "my clip" not in page.content()
    # This site's pipeline stand-in always fails, so the failure path must be visible and offer a retry.
    page.wait_for_selector("[role=alert] button", timeout=180000)
    assert "synthetic failure" in page.locator("[role=alert]").inner_text()


def test_job_page_links_every_output(site, page):
    base, _ = site
    sign_in_browser(page.signed_in_context, site)
    page.goto(f"{base}/#/jobs/job-done")
    page.wait_for_selector("#stages li")
    assert page.locator("#stages .dot.done").count() == 9
    assert page.locator(".tab.off").count() == 0
    assert "2 objects outlined" in page.locator("section.stack p.small").last.inner_text()


def test_watch_plays_with_overlays_phase_and_subtitles(site, page):
    base, _ = site
    sign_in_browser(page.signed_in_context, site)
    page.goto(f"{base}/#/jobs/job-done/watch")
    page.wait_for_selector("#watch-video")
    page.wait_for_function("document.querySelector('#watch-video').readyState >= 2", timeout=60000)
    assert page.evaluate("document.querySelector('#watch-video').videoWidth") == fx.W, "the video did not decode"
    wait_for_annotated_frame(page, 0)
    assert canvas_has_pixels(page, "#watch-canvas")
    assert page.locator("#watch-phase").inner_text() == "Preparation · 90%"
    assert page.locator("#watch-subtitle").inner_text() == "The grasper is visible."

    # Seek to a propagated frame in the last segment: every overlay must follow.
    page.evaluate("document.querySelector('#watch-video').currentTime = 2.5")
    wait_for_annotated_frame(page, 25)
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
    sign_in_browser(page.signed_in_context, site)
    page.goto(f"{base}/#/jobs/job-done/download")
    page.wait_for_selector(".dl")
    assert page.locator(".dl button").all_inner_texts() == ["Locked"] * page.locator(".dl").count()

    page.goto(f"{base}/#/jobs/job-done/final-pass")
    page.wait_for_selector("#fp-queue .fpitem")
    assert fx.TEST_USER in page.locator("#fp-reviewer").inner_text()
    assert page.locator("#fp-release-btn").is_disabled()
    assert page.locator("#fp-queue .fpitem .pill.AMBER").count() == 1
    page.wait_for_function(f"document.querySelector('#fp-canvas').width === {fx.W}")

    # First up is the amber. Correct it by drawing, which goes through real canvas clicks.
    assert "kf_00001 · Gallbladder" in page.locator(".fpitem.on").inner_text()
    page.click("#fp-draw")
    click_on_canvas(page, ((0.1, 0.1), (0.5, 0.1), (0.5, 0.6)))
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

    assert page.locator("#fp-release-btn").is_enabled()
    page.click("#fp-release-btn")
    page.wait_for_function("document.querySelector('#fp-release-btn').textContent === 'Released'")
    assert (jobs / "job-done" / "released_coco.json").exists()
    assert page.locator("#fp-approve").is_disabled()

    page.goto(f"{base}/#/jobs/job-done/download")
    page.wait_for_selector(".dl a.btn.primary")
    assert page.locator(".dl a.btn.primary").count() >= 4
    with page.expect_download() as download_info:
        page.locator(".dl[data-file='released_coco.json'] a").click()
    assert download_info.value.suggested_filename == "job-done_released_coco.json"

    # The wording must match the file: how many annotations a person actually reviewed.
    released = server.json.loads((jobs / "job-done" / "released_coco.json").read_text())
    n_all = len(released["annotations"])
    n_seen = sum(1 for a in released["annotations"] if a["attributes"].get("human_reviewed") and not a["attributes"].get("propagated"))
    text = page.locator("#dl-reviewed").inner_text()
    assert f"Of {n_all} annotations" in text and f"{n_seen} were individually reviewed" in text
    assert f"other {n_all - n_seen}" in text and "nobody looked at" in text
    assert "erified" not in page.locator("main").inner_text()

    # Every player style is reachable, and a person's decision shows as theirs.
    page.goto(f"{base}/#/jobs/job-done/watch")
    page.wait_for_selector("#watch-video")
    wait_for_annotated_frame(page, 0)
    origins = page.evaluate("[...document.querySelectorAll('#watch-objects li')].map(l => [l.dataset.origin, l.textContent])")
    assert any(o == "human" and "reviewed" in text for o, text in origins), origins
    page.evaluate("document.querySelector('#watch-video').currentTime = 0.5")
    wait_for_annotated_frame(page, 5)
    origins = page.evaluate("[...document.querySelectorAll('#watch-objects li')].map(l => l.dataset.origin)")
    assert set(origins) == {"propagated"}, origins
    assert "Tracked between keyframes" in page.locator("#watch-key").inner_text()


def test_no_verdict_before_the_frame_is_on_screen(site, browser):
    base, jobs = site
    job = jobs / "job-slow"
    fx.finished_job(jobs, "job-slow")
    context = browser.new_context(viewport={"width": 1280, "height": 900})
    sign_in_browser(context, site)
    page = context.new_page()
    held = HeldRequests(page, "**/files/frames/**")
    page.goto(f"{base}/#/jobs/job-slow/final-pass")
    page.wait_for_selector("#fp-approve")
    for selector in ("#fp-approve", "#fp-reject", "#fp-draw"):
        assert page.locator(selector).is_disabled(), selector
    for key in ("a", "x", "d"):
        page.keyboard.press(key)
    assert not (job / "final_pass.jsonl").exists()
    page.wait_for_timeout(200)
    assert held.routes, "the frame request should be in flight"
    held.release()
    page.wait_for_function("!document.querySelector('#fp-approve').disabled", timeout=30000)
    assert page.evaluate("document.querySelector('#fp-canvas').width") == fx.W

    # Points land in video pixels, wherever the canvas sits on screen.
    page.click("#fp-draw")
    click_on_canvas(page, [(x / fx.W, y / fx.H) for x, y in ((32, 18), (160, 18), (160, 90))])
    page.click("#fp-save-draw")
    page.wait_for_function("document.querySelector('#fp-progress').textContent.startsWith('1 of')")
    row = server.json.loads((job / "final_pass.jsonl").read_text().splitlines()[0])
    assert row["action"] == "correct"
    for (got_x, got_y), (x, y) in zip(row["polygon"], ((32, 18), (160, 18), (160, 90))):
        assert abs(got_x - x) <= 1.5 and abs(got_y - y) <= 1.5
    context.close()


def test_mock_model_jobs_are_labelled_as_placeholders(site, page):
    base, jobs = site
    sign_in_browser(page.signed_in_context, site)
    fx.finished_job(jobs, "job-mock")
    meta = jobs / "job-mock" / "meta.json"
    meta.write_text(server.json.dumps({**server.json.loads(meta.read_text()), "vlm_provider": "mock"}), encoding="utf-8")
    for route in ("", "/watch", "/final-pass", "/download"):
        page.goto(f"{base}/#/jobs/job-mock{route}")
        page.wait_for_selector("#mock-warning")
    page.goto(f"{base}/#/jobs/job-done")
    page.wait_for_selector("#stages li")
    assert page.locator("#mock-warning").count() == 0


def test_overlay_failure_is_visible_not_silent(site, page):
    base, _ = site
    sign_in_browser(page.signed_in_context, site)
    page.route("**/overlay?*", lambda route: route.fulfill(status=500, body="{}"))
    page.goto(f"{base}/#/jobs/job-done/watch")
    page.wait_for_selector("#watch-error", timeout=60000)
    assert "plain video" in page.locator("#watch-error").inner_text()
    page.wait_for_function("document.querySelector('#watch-frame').textContent.includes('failed to load')", timeout=60000)
    assert not canvas_has_pixels(page, "#watch-canvas")
    # It recovers by itself when the annotations come back, and the warning goes.
    page.unroute("**/overlay?*")
    page.wait_for_function("!document.querySelector('#watch-error')", timeout=30000)
    wait_for_annotated_frame(page, 0)
    assert canvas_has_pixels(page, "#watch-canvas")


def test_sign_in_is_required_and_session_can_end(site, page):
    base, _ = site
    sign_in_browser(page.signed_in_context, site)  # the account exists from here on
    page.signed_in_context.clear_cookies()
    page.goto(f"{base}/#/jobs/job-done")
    page.wait_for_selector("#auth-form")
    assert page.locator("#stages").count() == 0
    page.fill("#auth-password", "not the password")
    page.click("#auth-submit")
    page.wait_for_selector("[role=alert]")
    assert "wrong password" in page.locator("[role=alert]").inner_text()
    page.fill("#auth-password", fx.TEST_PASSWORD)
    page.click("#auth-submit")
    page.wait_for_selector("#stages li")
    assert page.locator("#session-user").inner_text() == fx.TEST_USER
    page.click("#sign-out")
    page.wait_for_selector("#auth-form")


def test_delete_needs_two_clicks_then_the_job_is_gone(site, page):
    base, jobs = site
    sign_in_browser(page.signed_in_context, site)
    fx.finished_job(jobs, "job-delete-me")
    page.goto(f"{base}/#/jobs/job-delete-me")
    page.wait_for_selector("#job-delete")
    assert "Deleted automatically" in page.locator("#job-expires").inner_text()
    page.click("#job-delete")
    assert (jobs / "job-delete-me").exists(), "one click must not delete"
    page.click("#job-delete")
    page.wait_for_selector("#drop")
    assert not (jobs / "job-delete-me").exists()


def test_no_stale_boxes_while_a_frames_annotations_are_loading(site, page):
    base, _ = site
    sign_in_browser(page.signed_in_context, site)
    held = HeldRequests(page, "**/overlay?start=20*")  # the chunk holding frames 20-39 stays in flight
    page.goto(f"{base}/#/jobs/job-done/watch")
    page.wait_for_selector("#watch-frame")
    wait_for_annotated_frame(page, 0)
    assert canvas_has_pixels(page, "#watch-canvas")

    page.evaluate("document.querySelector('#watch-video').currentTime = 2.5")
    page.wait_for_function("document.querySelector('#watch-frame').textContent.includes('loading annotations')")
    assert not canvas_has_pixels(page, "#watch-canvas"), "the old frame's boxes are still on screen"

    assert held.routes, "the overlay request should be in flight"
    held.release()
    page.wait_for_function("document.querySelector('#watch-frame').textContent.includes('propagated')")
    assert canvas_has_pixels(page, "#watch-canvas")


def test_model_or_pipeline_text_with_markup_is_not_injected(site, page):
    base, jobs = site
    sign_in_browser(page.signed_in_context, site)
    job = jobs / "job-html"
    job.mkdir()
    payload = '<img src=x onerror="window.pwned=1">'
    (job / "status.json").write_text(server.json.dumps({"stage": "detect", "message": payload, "error": None}), encoding="utf-8")
    page.goto(f"{base}/#/jobs/job-html")
    page.wait_for_selector("#message")
    page.wait_for_function("document.querySelector('#message').textContent.length > 0")
    assert page.locator("#message").inner_text() == payload
    assert page.evaluate("window.pwned") is None


def test_phone_width_has_no_sideways_scroll(site, browser):
    base, _ = site
    context = browser.new_context(viewport={"width": 390, "height": 800})
    sign_in_browser(context, site)
    page = context.new_page()
    for route in ("", "#/jobs/job-done", "#/jobs/job-done/watch", "#/jobs/job-done/final-pass", "#/jobs/job-done/download"):
        page.goto(f"{base}/{route}")
        page.wait_for_timeout(1000)
        overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        assert overflow <= 0, f"{route or 'upload'} overflows by {overflow}px"
    context.close()
