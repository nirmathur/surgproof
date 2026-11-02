import { h, api, postJson, jobHeader, deleteButton, labelColour, labelName, tracePolygon, drawChip, displayScale, AMBER } from '/lib.js';

function stageRow(stage, job) {
  const active = stage.state === 'active';
  const stageProgress = job.stage_progress;
  return h('li', { class: stage.state, 'data-stage': stage.key },
    h('span', { class: `dot ${stage.state}` }),
    h('span', {}, stage.label),
    h('span', { class: 'muted small mono' },
      stage.evidence || (active && typeof stageProgress === 'number' ? `${Math.round(stageProgress * 100)}%` : '')));
}

function drawPreview(canvas, preview) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const scale = displayScale(canvas);
  for (const object of preview.objects || []) {
    const colour = object.status === 'AMBER' ? AMBER : labelColour(object.label);
    tracePolygon(ctx, object.polygon);
    ctx.fillStyle = `${colour}33`;
    ctx.fill();
    ctx.strokeStyle = colour;
    ctx.lineWidth = 2 * scale;
    ctx.stroke();
    const [x, y] = object.polygon.reduce((best, p) => (p[1] < best[1] ? p : best));
    drawChip(ctx, labelName(object.label), x, y, colour, scale);
  }
}

export async function renderJob(root, { id }) {
  const url = `/api/jobs/${encodeURIComponent(id)}`;
  const header = h('div');
  const stageList = h('ol', { class: 'stages', id: 'stages' });
  const fill = h('div', { style: 'width:0' });
  const message = h('p', { class: 'small muted', id: 'message' });
  const alerts = h('div');
  const stats = h('p', { class: 'small muted mono' });
  const manage = h('div', { class: 'stack' });
  let deleteShown = false;
  const img = h('img', { alt: 'Latest keyframe processed by the pipeline' });
  const canvas = h('canvas', { class: 'over' });
  const caption = h('p', { class: 'small muted' });
  const frame = h('div', { class: 'frame', hidden: true }, img, canvas);

  root.append(header, h('div', { class: 'stack' }, alerts, h('div', { class: 'jobgrid' },
    h('section', { class: 'card stack' },
      h('div', { class: 'bar' }, fill), message, stageList, stats, manage),
    h('section', { class: 'stack' }, frame, caption))));

  let lastKf = null;
  let stopped = false;
  let timer = null;

  async function retry() {
    alerts.replaceChildren();
    await postJson(`${url}/retry`, {});
    tick();
  }

  async function tick() {
    let job;
    try {
      job = await api(url);
    } catch (err) {
      if (stopped) return;
      alerts.replaceChildren(h('div', { class: 'notice error' }, err.message));
      timer = setTimeout(tick, 3000);
      return;
    }
    if (stopped) return;
    header.replaceChildren(jobHeader(job, 'job'));
    stageList.replaceChildren(...job.stages.map((stage) => stageRow(stage, job)));
    const fraction = job.state === 'done' ? 1 : Math.min(1, job.progress || 0);
    fill.style.width = `${fraction * 100}%`;
    message.textContent = job.state === 'done'
      ? 'Finished: this is an AI draft. Nothing can be exported until a person completes the final pass.'
      : (job.message || '');
    const costed = typeof job.cost_usd === 'number' ? ` · model cost $${job.cost_usd.toFixed(2)}` : '';
    stats.textContent = `Last output ${job.last_activity ? new Date(job.last_activity).toLocaleTimeString() : 'none yet'}${costed}`;

    if (!deleteShown) {
      deleteShown = true;
      manage.append(deleteButton(job, () => { location.hash = '#/'; }));
    }
    if (job.expires) {
      const line = document.getElementById('job-expires') || h('p', { class: 'small muted', id: 'job-expires' });
      line.textContent = `Deleted automatically on ${new Date(job.expires).toLocaleDateString()}.`;
      if (!line.isConnected) manage.prepend(line);
    }

    alerts.replaceChildren();
    if (job.state === 'failed') {
      alerts.append(h('div', { class: 'notice error stack', role: 'alert' },
        h('p', {}, job.error),
        job.released ? null : h('div', {}, h('button', { class: 'btn', onclick: () => retry() }, 'Retry'))));
    } else if (job.state === 'stalled') {
      alerts.append(h('div', { class: 'notice stack', role: 'alert' },
        h('p', {}, job.state_detail === 'gone'
          ? 'The worker for this job is gone, so it is not making progress.'
          : 'No new output for over 15 minutes. The worker looks hung.'),
        h('div', {}, h('button', { class: 'btn', onclick: () => retry() }, 'Retry'))));
    } else if (job.state === 'done' && job.final_pass) {
      const fp = job.final_pass;
      alerts.append(h('div', { class: 'row' },
        h('a', { class: 'btn primary', href: `#/jobs/${encodeURIComponent(id)}/watch` }, 'Watch'),
        h('a', { class: 'btn', href: `#/jobs/${encodeURIComponent(id)}/final-pass` },
          fp.released ? 'Final pass (released)' : `Final pass (${fp.pending} of ${fp.total} left)`),
        h('a', { class: 'btn', href: `#/jobs/${encodeURIComponent(id)}/download` }, 'Download')));
    }

    try {
      const preview = await api(`${url}/preview`);
      if (!stopped && preview.kf) {
        const objects = (preview.objects || []).length;
        caption.textContent = `${preview.kf}: ${objects} object${objects === 1 ? '' : 's'} outlined by the model`;
        frame.hidden = false;
        if (preview.kf !== lastKf) {
          lastKf = preview.kf;
          img.src = preview.frame_url;
        }
        canvas.width = preview.width || 1280;
        canvas.height = preview.height || 720;
        drawPreview(canvas, preview);
      }
    } catch (_) { /* the preview is optional, the stage list above is what matters */ }

    if (!stopped && !['done', 'failed', 'stalled'].includes(job.state)) timer = setTimeout(tick, 1500);
    else if (!stopped && job.state === 'stalled') timer = setTimeout(tick, 10000);
  }

  await tick();
  return () => { stopped = true; clearTimeout(timer); };
}
