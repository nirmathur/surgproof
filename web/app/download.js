import { h, api, jobHeader, fmtBytes } from '/lib.js';

function fileCard(title, description, url, name, info, lockedReason) {
  const size = info && info.bytes !== undefined ? fmtBytes(info.bytes) : '';
  const action = lockedReason || !info
    ? h('button', { class: 'btn', disabled: true }, lockedReason ? 'Locked' : 'Not available')
    : h('a', { class: 'btn primary', href: `${url}/files/${name}?download=1`, download: '' }, 'Download');
  return h('section', { class: 'card dl', 'data-file': name },
    h('div', { class: 'grow' },
      h('h2', { style: 'margin-bottom:2px' }, title),
      h('p', { class: 'small muted' }, lockedReason || description)),
    h('span', { class: 'small muted mono' }, lockedReason ? '' : size),
    action);
}

// Says exactly how much a person looked at, from the released file itself. Most of a dataset is unreviewed AI output.
function reviewedLine(fp, counts) {
  if (!counts) return null;
  const pool = fp.n_audit_pool === null || fp.n_audit_pool === undefined ? '' : ` drawn at random from ${fp.n_audit_pool} of the AI's decisions`;
  return h('section', { class: 'card stack', id: 'dl-reviewed' },
    h('h2', { style: 'margin:0' }, 'What was reviewed'),
    h('p', { class: 'small' },
      `A person reviewed ${fp.total} labels: ${fp.n_amber} the AI was unsure about, and ${fp.n_audit} audit samples${pool}. `
      + `They overruled ${fp.audit_disagreements} of the audit samples.`),
    h('p', { class: 'small' },
      `Of ${counts.n_annotations} annotations in this dataset, ${counts.n_human_reviewed} were individually reviewed by a person. `
      + `The other ${counts.n_annotations - counts.n_human_reviewed}, including ${counts.n_propagated} propagated by optical flow, are AI output that nobody looked at.`));
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
    : `Locked until a person completes the final pass and releases this job (${fp.pending} of ${fp.total} items left). Until then this is an unreviewed AI draft: you can view it on the Watch tab.`;
  root.append(h('div', { class: 'stack' },
    reviewedLine(fp, job.released_counts),
    fileCard('Annotated video', 'The source video with the boxes, masks, phase and AI-draft narration drawn in.',
      url, 'annotated.mp4', out['annotated.mp4'], locked),
    fileCard('COCO dataset', 'Polygons, boxes and provenance attributes, with the final-pass decisions applied. Each annotation says whether a person reviewed it.',
      url, 'released_coco.json', out['released_coco.json'], locked),
    fileCard('Dataset bundle', 'Everything in one zip, so every image in the dataset can be resolved.',
      url, 'dataset_bundle.zip', out['dataset_bundle.zip'], fp.released && !out['dataset_bundle.zip'] ? 'Not produced by this release.' : locked),
    fileCard('Provenance', 'One row per annotation: model, prompt hash and how it was decided.',
      url, 'provenance.jsonl', out['provenance.jsonl'], locked),
    fileCard('Final-pass record', 'Every review click, with who made it and when.',
      url, 'final_pass.jsonl', out['final_pass.jsonl'], locked),
    locked ? h('div', {}, h('a', { class: 'btn', href: `#/jobs/${encodeURIComponent(id)}/final-pass` }, 'Go to final pass')) : null));
  return null;
}
