import { h, api, postJson, jobHeader, fmtBytes } from '/lib.js';

function fileCard(title, description, url, info, lockedReason) {
  const size = info && info.bytes !== undefined ? fmtBytes(info.bytes) : '';
  const action = lockedReason || !info
    ? h('button', { class: 'btn', disabled: true }, lockedReason ? 'Locked' : 'Not available')
    : h('a', { class: 'btn primary', href: `${url}?download=1`, download: '' }, 'Download');
  return h('section', { class: 'card dl' },
    h('div', { class: 'grow' },
      h('h2', { style: 'margin-bottom:2px' }, title),
      h('p', { class: 'small muted' }, lockedReason || description)),
    h('span', { class: 'small muted mono' }, size),
    action);
}

export async function renderDownload(root, { id }) {
  const url = `/api/jobs/${encodeURIComponent(id)}`;
  const job = await api(url);
  root.append(jobHeader(job, 'download'));
  if (job.state !== 'done') {
    root.append(h('div', { class: 'notice' }, `This job is ${job.state}. Downloads appear when it finishes.`));
    return null;
  }
  const out = job.outputs || {};
  const fp = job.final_pass || {};
  const locked = fp.released ? null
    : `Locked until the final pass is complete and a person releases it (${fp.pending} of ${fp.total} items left).`;
  root.append(h('div', { class: 'stack' },
    fileCard('Annotated video', 'The source video with the AI\'s boxes, masks, phase and narration burned in. AI output, not yet human-checked unless released.',
      `${url}/files/annotated.mp4`, out['annotated.mp4']),
    fileCard('COCO dataset', 'Polygons, boxes and provenance attributes, with the human final pass applied.',
      `${url}/files/released_coco.json`, out['released_coco.json'], locked),
    fileCard('Provenance', 'One row per annotation: model, prompt hash and how it was decided.',
      `${url}/files/provenance.jsonl`, out['provenance.jsonl']),
    locked ? h('div', {}, h('a', { class: 'btn', href: `#/jobs/${encodeURIComponent(id)}/final-pass` }, 'Go to final pass')) : null));
  return null;
}
