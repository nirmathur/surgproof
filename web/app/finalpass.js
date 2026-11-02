import { h, api, postJson, jobHeader, labelColour, labelName, tracePolygon, drawChip, displayScale, AMBER } from '/lib.js';

const REVIEWER_KEY = 'surgproof.reviewer';

function readReviewer() {
  try { return localStorage.getItem(REVIEWER_KEY) || ''; } catch (_) { return ''; }
}
function saveReviewer(name) {
  try { localStorage.setItem(REVIEWER_KEY, name); } catch (_) { /* a per-browser convenience only */ }
}

function boundsOf(item, width, height) {
  const points = item.outline.flat();
  if (item.verify_box) points.push([item.verify_box[0], item.verify_box[1]], [item.verify_box[2], item.verify_box[3]]);
  if (!points.length) return null;
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const pad = 0.25 * Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 40);
  return {
    x: Math.max(0, Math.min(...xs) - pad),
    y: Math.max(0, Math.min(...ys) - pad),
    x1: Math.min(width, Math.max(...xs) + pad),
    y1: Math.min(height, Math.max(...ys) + pad),
  };
}

export async function renderFinalPass(root, { id }) {
  const url = `/api/jobs/${encodeURIComponent(id)}`;
  const job = await api(url);
  root.append(jobHeader(job, 'final'));
  if (job.state !== 'done') {
    root.append(h('div', { class: 'notice' }, `This job is ${job.state}. The final pass opens when it finishes.`));
    return null;
  }

  let state = await api(`${url}/final-pass`);
  let currentId = null;
  let drawing = null;
  let stopped = false;
  // A verdict needs the frame on screen, and a drawn point needs the canvas in video pixels.
  let frameReady = false;

  const image = new Image();
  const canvas = h('canvas', { id: 'fp-canvas' });
  const ctx = canvas.getContext('2d');
  const frameBox = h('div', { class: 'frame fpframe' }, canvas);
  const heading = h('div', { class: 'row spread' });
  const details = h('dl', { class: 'kv', id: 'fp-details' });
  const crop = h('canvas', { class: 'crop' });
  const rounds = h('pre', { class: 'rounds' });
  const actions = h('div', { class: 'row', id: 'fp-actions' });
  const queueList = h('div', { class: 'fpqueue', id: 'fp-queue' });
  const progress = h('p', { class: 'small muted mono', id: 'fp-progress' });
  const barFill = h('div', { style: 'width:0' });
  const errors = h('div');
  const releaseBox = h('section', { class: 'card stack', id: 'fp-release' });
  const reviewer = h('input', { class: 'input', id: 'fp-reviewer', placeholder: 'Your name (needed to release)', 'aria-label': 'Reviewer name', value: readReviewer(), oninput: () => { saveReviewer(reviewer.value); renderRelease(); } });

  root.append(h('div', { class: 'stack' }, errors, h('div', { class: 'fpgrid' },
    h('section', { class: 'stack' }, heading, frameBox, actions, h('p', { class: 'small muted', id: 'fp-hint' })),
    h('aside', { class: 'stack' },
      h('section', { class: 'card stack' }, progress, h('div', { class: 'bar' }, barFill), reviewer, queueList),
      releaseBox,
      h('section', { class: 'card stack' }, h('h2', { style: 'margin:0' }, 'Evidence'), details, crop, rounds)))));

  const current = () => state.items.find((item) => item.id === currentId);
  const nextPending = () => state.items.find((item) => !item.verdict);

  function draw() {
    const item = current();
    if (!item || !frameReady) return;
    canvas.width = state.width || image.naturalWidth;
    canvas.height = state.height || image.naturalHeight;
    const scale = displayScale(canvas);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    const verdict = item.verdict;
    const humanDrawn = verdict && verdict.polygon;
    const outlines = drawing ? [] : (humanDrawn ? [verdict.polygon] : item.outline);
    const colour = item.kind === 'amber' ? AMBER : labelColour(item.label);
    if (item.final_present !== false) {
      for (const polygon of outlines) {
        tracePolygon(ctx, polygon);
        ctx.fillStyle = `${colour}33`;
        ctx.fill();
        ctx.strokeStyle = colour;
        ctx.lineWidth = 2 * scale;
        ctx.stroke();
      }
      if (outlines.length) {
        const [x, y] = outlines[0].reduce((best, p) => (p[1] < best[1] ? p : best));
        drawChip(ctx, labelName(item.label), x, y, colour, scale);
      }
    }
    if (item.verify_box && !drawing) {
      const [x0, y0, x1, y1] = item.verify_box;
      ctx.setLineDash([6 * scale, 5 * scale]);
      ctx.strokeStyle = '#8a909e';
      ctx.lineWidth = 1.5 * scale;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.setLineDash([]);
    }
    if (drawing) {
      ctx.strokeStyle = '#3b82f6';
      ctx.fillStyle = '#3b82f6';
      ctx.lineWidth = 2 * scale;
      ctx.beginPath();
      drawing.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.stroke();
      drawing.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x, y, 4 * scale, 0, 2 * Math.PI); ctx.fill(); });
    }

    const bounds = boundsOf(item, canvas.width, canvas.height);
    crop.hidden = !bounds;
    if (bounds) {
      const w = bounds.x1 - bounds.x;
      const ch = bounds.y1 - bounds.y;
      crop.width = 600;
      crop.height = Math.max(1, Math.round((600 * ch) / w));
      const cropCtx = crop.getContext('2d');
      cropCtx.drawImage(image,
        (bounds.x / canvas.width) * image.naturalWidth, (bounds.y / canvas.height) * image.naturalHeight,
        (w / canvas.width) * image.naturalWidth, (ch / canvas.height) * image.naturalHeight,
        0, 0, crop.width, crop.height);
      const zoom = crop.width / w;
      cropCtx.setTransform(zoom, 0, 0, zoom, -bounds.x * zoom, -bounds.y * zoom);
      cropCtx.strokeStyle = colour;
      cropCtx.lineWidth = 2 / zoom;
      for (const polygon of item.outline) { tracePolygon(cropCtx, polygon); cropCtx.stroke(); }
      cropCtx.setTransform(1, 0, 0, 1, 0, 0);
    }
  }
  image.addEventListener('load', () => { frameReady = true; if (!stopped) render(); });
  image.addEventListener('error', () => {
    errors.replaceChildren(h('div', { class: 'notice error', role: 'alert' }, 'This frame could not be loaded, so it cannot be reviewed.'));
  });

  function renderDetails(item) {
    const answer = (value) => (value === null || value === undefined ? 'no answer' : (value ? 'present' : 'absent'));
    const rows = [
      ['Why here', item.kind === 'amber' ? (item.reason || 'The amber loop did not settle this label') : 'Random audit of a label the AI accepted on its own'],
      ['AI decision', item.ai_status],
      ['AI proposes', answer(item.ai_present)],
      ['Second pass', answer(item.verify_present)],
      ['Confidence', typeof item.detect_conf === 'number' ? item.detect_conf.toFixed(2) : 'not given'],
      ['Outline from', item.outline_source],
      ['Frame', item.frame_index],
    ];
    details.replaceChildren(...rows.flatMap(([key, value]) => [h('dt', {}, key), h('dd', {}, String(value))]));
    rounds.hidden = !item.rounds.length;
    rounds.textContent = item.rounds.length ? `Amber loop rounds:\n${JSON.stringify(item.rounds, null, 1)}` : '';
  }

  function renderActions(item) {
    const locked = state.released || !frameReady;
    actions.replaceChildren();
    if (drawing) {
      actions.append(
        h('button', { class: 'btn primary', id: 'fp-save-draw', disabled: drawing.length < 3, onclick: () => saveDrawing() }, 'Save outline', h('kbd', {}, 'Enter')),
        h('button', { class: 'btn', onclick: () => { drawing = null; render(); } }, 'Cancel', h('kbd', {}, 'Esc')));
      return;
    }
    actions.append(
      h('button', { class: 'btn good', id: 'fp-approve', disabled: locked || !item.can_approve, title: item.can_approve ? null : 'The AI drew no mask here', onclick: () => review('approve') }, 'Approve', h('kbd', {}, 'A')),
      h('button', { class: 'btn bad', id: 'fp-reject', disabled: locked, onclick: () => review('reject') }, 'Not present', h('kbd', {}, 'X')),
      h('button', { class: 'btn', id: 'fp-draw', disabled: locked, onclick: () => { drawing = []; render(); } }, 'Draw correction', h('kbd', {}, 'D')));
    if (item.verdict) {
      actions.append(h('span', { class: 'small muted' }, `Recorded: ${item.verdict.action}${state.released ? '' : '. You can change it.'}`));
    }
  }

  function renderRelease() {
    const pending = state.pending;
    const button = h('button', {
      class: `btn ${pending === 0 && !state.released ? 'good' : ''}`,
      id: 'fp-release-btn',
      disabled: pending > 0 || state.released || !reviewer.value.trim(),
      onclick: () => release(),
    }, state.released ? 'Released' : 'Release dataset');
    const auditLine = `Audit: ${state.audit_reviewed} of ${state.n_audit} checked, ${state.audit_disagreements} corrected by you (random sample of ${state.n_audit_pool} auto-accepted labels).`;
    releaseBox.replaceChildren(
      h('h2', { style: 'margin:0' }, 'Release'),
      h('p', { class: 'small muted' }, auditLine),
      state.released
        ? h('p', { class: 'small' }, 'This dataset is released. ', h('a', { href: `#/jobs/${encodeURIComponent(id)}/download` }, 'Download it'))
        : h('p', { class: 'small muted' }, pending > 0 ? `Locked: ${pending} of ${state.total} items still need your decision.` : (reviewer.value.trim() ? 'Every item is reviewed. Releasing writes the final dataset and locks this page.' : 'Every item is reviewed. Enter your name above to sign off.')),
      button);
  }

  function render() {
    progress.textContent = `${state.reviewed} of ${state.total} reviewed · ${state.n_amber} amber, ${state.n_audit} audit`;
    barFill.style.width = `${state.total ? (state.reviewed / state.total) * 100 : 100}%`;
    frameBox.classList.toggle('drawing', Boolean(drawing));
    queueList.replaceChildren(...state.items.map((item) => h('button', {
      class: `fpitem${item.id === currentId ? ' on' : ''}`,
      'data-id': item.id,
      onclick: () => select(item.id),
    },
    h('span', { class: `dot ${item.verdict ? 'done' : (item.kind === 'amber' ? 'stalled' : '')}` }),
    h('span', { class: 'grow' }, `${item.kf} · ${labelName(item.label)}`),
    h('span', { class: `pill ${item.kind === 'amber' ? 'AMBER' : ''}` }, item.kind))));
    renderRelease();
    document.getElementById('fp-hint').textContent = drawing
      ? 'Click around the structure to place points, then save.'
      : '';
    const item = current();
    if (!item) {
      heading.replaceChildren(h('h2', { style: 'margin:0' }, state.total ? 'Pick an item' : 'Nothing to review'));
      actions.replaceChildren();
      frameBox.hidden = true;
      return;
    }
    frameBox.hidden = false;
    heading.replaceChildren(
      h('h2', { style: 'margin:0' }, `${labelName(item.label)}: is it there, and is the outline right?`),
      h('span', { class: `pill ${item.kind === 'amber' ? 'AMBER' : ''}` }, item.kind));
    renderDetails(item);
    renderActions(item);
    draw();
  }

  function select(itemId) {
    currentId = itemId;
    drawing = null;
    const item = current();
    if (item && !image.src.endsWith(item.frame_url)) {
      frameReady = false;
      image.src = item.frame_url;
    }
    render();
  }

  async function reload(payload) {
    errors.replaceChildren();
    try {
      await postJson(`${url}/final-pass`, payload);
    } catch (err) {
      errors.replaceChildren(h('div', { class: 'notice error', role: 'alert' }, err.message));
      return false;
    }
    state = await api(`${url}/final-pass`);
    return !stopped;
  }

  async function review(action, polygon) {
    const item = current();
    if (!item || state.released) return;
    const ok = await reload({ action, kf: item.kf, label: item.label, polygon, reviewer: reviewer.value });
    if (!ok) return;
    drawing = null;
    const next = nextPending();
    select(next ? next.id : currentId);
  }

  function saveDrawing() {
    if (drawing && drawing.length >= 3) review('correct', drawing.map(([x, y]) => [Math.round(x * 10) / 10, Math.round(y * 10) / 10]));
  }

  async function release() {
    if (await reload({ action: 'release', reviewer: reviewer.value })) render();
  }

  canvas.addEventListener('click', (e) => {
    if (!drawing) return;
    const box = canvas.getBoundingClientRect();
    drawing.push([(e.clientX - box.left) * (canvas.width / box.width), (e.clientY - box.top) * (canvas.height / box.height)]);
    render();
  });

  const onKey = (e) => {
    if (e.target && ['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const item = current();
    if (!item || state.released || !frameReady) return;
    const key = e.key.toLowerCase();
    if (drawing) {
      if (key === 'enter') saveDrawing();
      if (key === 'escape') { drawing = null; render(); }
      return;
    }
    if (key === 'a') document.getElementById('fp-approve')?.click();
    if (key === 'x') document.getElementById('fp-reject')?.click();
    if (key === 'd') document.getElementById('fp-draw')?.click();
  };
  window.addEventListener('keydown', onKey);

  const first = nextPending() || state.items[0];
  if (first) select(first.id); else render();
  return () => { stopped = true; window.removeEventListener('keydown', onKey); };
}
