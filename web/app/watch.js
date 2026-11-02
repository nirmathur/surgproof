import { h, api, jobHeader, fmtTime, labelColour, labelName, titleCase, tracePolygon, drawChip, displayScale, AMBER } from '/lib.js';

const CHUNK_SECONDS = 2;
const MAX_CHUNKS = 90;
const PHASE_SHADES = ['#27406b', '#2f5f8f', '#3f7fb0', '#4f9a9f', '#5f8f7f', '#7a7f6a', '#8a6f5f'];

// Split a segment's text into sentences and share its time by length, so long commentary stays readable.
function subtitleCues(segments) {
  const cues = [];
  for (const segment of segments) {
    const sentences = String(segment.text || '').match(/[^.!?]+[.!?]*\s*/g) || [];
    const total = sentences.reduce((sum, s) => sum + s.length, 0) || 1;
    let t = segment.t0;
    for (const sentence of sentences) {
      const end = t + (segment.t1 - segment.t0) * (sentence.length / total);
      cues.push({ t0: t, t1: end, text: sentence.trim() });
      t = end;
    }
  }
  return cues;
}

function phaseBands(phases, duration) {
  const bands = [];
  for (const phase of phases) {
    const last = bands[bands.length - 1];
    if (last && last.phase === phase.phase) continue;
    if (last) last.t1 = phase.t;
    bands.push({ phase: phase.phase, t0: bands.length ? phase.t : 0, t1: duration });
  }
  return bands;
}

export async function renderWatch(root, { id }) {
  const url = `/api/jobs/${encodeURIComponent(id)}`;
  const job = await api(url);
  root.append(jobHeader(job, 'watch'));
  if (job.state !== 'done') {
    root.append(h('div', { class: 'notice' }, `This job is ${job.state}. Watch opens when it finishes.`));
    return null;
  }
  const data = await api(`${url}/watch`);
  const fps = data.fps;
  const chunkFrames = Math.max(1, Math.ceil(fps * CHUNK_SECONDS));

  const video = h('video', { src: `${url}/video`, controls: true, playsInline: true, preload: 'auto', id: 'watch-video' });
  const canvas = h('canvas', { class: 'over', id: 'watch-canvas', width: data.width, height: data.height });
  const phaseChip = h('div', { class: 'phase-chip', id: 'watch-phase', hidden: true });
  const subText = h('span', { id: 'watch-subtitle' });
  const subs = h('div', { class: 'subs', hidden: true }, subText);
  const playhead = h('div', { class: 'playhead' });
  const timeline = h('div', { class: 'timeline', id: 'watch-timeline', role: 'slider', 'aria-label': 'Surgical phases over time' });
  const status = h('p', { class: 'small muted mono', id: 'watch-frame' });
  const legend = h('div', { class: 'row small muted' });
  const seglist = h('div', { class: 'seglist' });

  const toggles = { masks: true, boxes: true, subtitles: true };
  const toggleRow = h('div', { class: 'row' }, ...Object.keys(toggles).map((key) => h('label', { class: 'toggle' },
    h('input', { type: 'checkbox', checked: true, onchange: (e) => { toggles[key] = e.target.checked; redraw(); } }),
    titleCase(key))));

  root.append(h('div', { class: 'watchgrid' },
    h('section', { class: 'stack' },
      h('div', { class: 'frame' }, video, canvas, phaseChip, subs),
      timeline,
      h('div', { class: 'row spread' }, status, toggleRow),
      legend),
    h('section', { class: 'card' }, h('h2', {}, 'Narration'), seglist)));

  // ---- phase, narration
  const duration = data.n_frames && fps ? data.n_frames / fps : 0;
  const names = [...new Set(data.phases.map((p) => p.phase))];
  const shade = (phase) => PHASE_SHADES[names.indexOf(phase) % PHASE_SHADES.length];
  for (const band of phaseBands(data.phases, duration)) {
    timeline.append(h('div', {
      style: `width:${((band.t1 - band.t0) / (duration || 1)) * 100}%;background:${shade(band.phase)}`,
      title: titleCase(band.phase),
    }));
  }
  timeline.append(playhead);
  timeline.addEventListener('click', (e) => {
    const box = timeline.getBoundingClientRect();
    video.currentTime = ((e.clientX - box.left) / box.width) * duration;
  });
  legend.append(...names.map((name) => h('span', {}, h('span', { class: 'swatch', style: `background:${shade(name)}` }), titleCase(name))));

  const cues = subtitleCues(data.narration);
  const segButtons = data.narration.map((segment) => h('button', {
    class: 'seg',
    onclick: () => { video.currentTime = segment.t0; },
  },
  h('span', { class: 'muted small mono' }, `${fmtTime(segment.t0)} · ${titleCase(segment.phase)}`),
  h('br'), segment.text,
  (segment.evidence || []).length
    ? h('span', { class: 'muted small', style: 'display:block;margin-top:4px' }, `Evidence: ${segment.evidence.join(', ')}`)
    : null));
  seglist.append(...(segButtons.length ? segButtons : [h('p', { class: 'muted small' }, 'No narration for this job.')]));

  // ---- overlay chunks
  const chunks = new Map();
  let stopped = false;
  let currentFrame = -1;

  function chunk(index) {
    if (index < 0 || (data.n_frames && index * chunkFrames >= data.n_frames)) return null;
    if (!chunks.has(index)) {
      const entry = { frames: null };
      chunks.set(index, entry);
      api(`${url}/overlay?start=${index * chunkFrames}&end=${(index + 1) * chunkFrames}`).then((res) => {
        entry.frames = res.frames;
        if (!stopped && Math.floor(currentFrame / chunkFrames) === index) redraw();
      }).catch(() => { chunks.delete(index); });
      if (chunks.size > MAX_CHUNKS) chunks.delete(chunks.keys().next().value);
    }
    return chunks.get(index);
  }

  const ctx = canvas.getContext('2d');

  function drawObjects(objects) {
    const scale = displayScale(canvas);
    for (const object of objects) {
      const held = object.status === 'AMBER' && !object.human;
      const colour = held ? AMBER : labelColour(object.label);
      if (toggles.masks && object.polygons.length) {
        ctx.fillStyle = `${colour}3d`;
        for (const polygon of object.polygons) { tracePolygon(ctx, polygon); ctx.fill(); }
      }
      if (toggles.boxes && object.box) {
        const [x0, y0, x1, y1] = object.box;
        ctx.setLineDash(held ? [8 * scale, 6 * scale] : []);
        ctx.strokeStyle = colour;
        ctx.lineWidth = 2 * scale;
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        ctx.setLineDash([]);
        drawChip(ctx, held ? `${labelName(object.label)} · unsure` : labelName(object.label), x0, y0, colour, scale);
      }
    }
  }

  let lastTime = 0;
  function redraw() {
    const entry = chunks.get(Math.floor(currentFrame / chunkFrames));
    if (entry && entry.frames) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const objects = entry.frames[String(currentFrame)] || [];
      drawObjects(objects);
      const propagated = objects.some((o) => o.propagated);
      const source = objects.length ? (propagated ? 'propagated by optical flow' : 'AI keyframe') : 'nothing annotated';
      status.textContent = `${fmtTime(lastTime)} · frame ${currentFrame} of ${data.n_frames} · ${source}`;
    }
    subs.hidden = !toggles.subtitles || !subText.textContent;
  }

  function onFrame(mediaTime) {
    lastTime = mediaTime;
    const frame = Math.round(mediaTime * fps);
    const chunkIndex = Math.floor(frame / chunkFrames);
    chunk(chunkIndex);
    chunk(chunkIndex + 1);
    if (frame !== currentFrame) {
      currentFrame = frame;
      redraw();
    }
    const phase = data.phases.filter((p) => p.t <= mediaTime).pop();
    phaseChip.hidden = !phase;
    if (phase) {
      const conf = typeof phase.conf === 'number' ? ` · ${Math.round(phase.conf * 100)}%` : '';
      phaseChip.textContent = `${titleCase(phase.phase)}${conf}`;
    }
    const cue = cues.find((c) => c.t0 <= mediaTime && mediaTime < c.t1);
    const text = cue ? cue.text : '';
    if (subText.textContent !== text) subText.textContent = text;
    subs.hidden = !toggles.subtitles || !text;
    data.narration.forEach((segment, i) => {
      segButtons[i].classList.toggle('on', segment.t0 <= mediaTime && mediaTime < segment.t1);
    });
    if (duration) playhead.style.left = `${Math.min(100, (mediaTime / duration) * 100)}%`;
  }

  // requestVideoFrameCallback reports the exact timestamp of the frame on screen, which keeps boxes locked to it.
  const frameLoop = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
  if (frameLoop) {
    const step = (_now, metadata) => {
      if (stopped) return;
      onFrame(metadata.mediaTime);
      video.requestVideoFrameCallback(step);
    };
    video.requestVideoFrameCallback(step);
  } else {
    const step = () => {
      if (stopped) return;
      onFrame(video.currentTime);
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  video.addEventListener('seeked', () => onFrame(video.currentTime));
  video.addEventListener('loadeddata', () => onFrame(video.currentTime));

  const onKey = (e) => {
    if (e.target && ['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
    if (e.key === ',' || e.key === '.') {
      video.pause();
      video.currentTime = Math.max(0, video.currentTime + (e.key === ',' ? -1 : 1) / fps);
    }
  };
  window.addEventListener('keydown', onKey);
  return () => {
    stopped = true;
    window.removeEventListener('keydown', onKey);
    video.pause();
    video.removeAttribute('src');
    video.load();
  };
}
