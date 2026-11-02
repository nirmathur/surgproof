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

const STRATUM_WORDS = {
  accepted: 'accepted by both AI passes',
  amber_accepted: 'accepted by the amber loop',
  ai_absent: 'called absent by the AI',
  propagated: 'frames tracked by optical flow',
};

// Says exactly how much a person looked at, from the released file itself. Most of a dataset is unreviewed AI output.
function reviewedLine(fp, counts) {
  if (!counts) return null;
  const strata = Object.entries(fp.audit_by_stratum || {}).map(([key, n]) => `${n} ${STRATUM_WORDS[key] || key}`).join(', ');
  const parts = [`${fp.n_amber} the AI could not settle`, `${fp.n_audit} audit samples${strata ? ` (${strata})` : ''}`];
  if (fp.n_escalation) parts.push(`${fp.n_escalation} added when the audit stop rule tripped`);
  return h('section', { class: 'card stack', id: 'dl-reviewed' },
    h('h2', { style: 'margin:0' }, 'What was reviewed'),
    h('p', { class: 'small' }, `A person reviewed ${fp.total} labels: ${parts.join(', ')}. They overruled the AI on ${fp.audit_disagreements} of the audit samples.`),
    h('p', { class: 'small' },
      `Of ${counts.n_annotations} annotations in this dataset, ${counts.n_human_reviewed} were individually reviewed by a person. `
      + `The other ${counts.n_annotations - counts.n_human_reviewed}, which include most of the ${counts.n_propagated} propagated by optical flow, are AI output that nobody looked at.`));
}

function retentionLine(job) {
  if (!job.expires) return null;
  return h('div', { class: 'notice', id: 'dl-retention' },
    `This job is deleted automatically on ${new Date(job.expires).toLocaleDateString()}, released dataset included. Download anything you want to keep.`);
}

async function manifestLine(url) {
  const line = h('p', { class: 'small', id: 'dl-signature' }, 'Checking the signature of the released files');
  api(`${url}/release-check`).then((res) => {
    line.textContent = res.ok
      ? 'Signature check passed: the released files match their signed manifest. It detects later edits, and it is not proof against someone holding the key on this machine.'
      : `Signature check FAILED, so these files changed after release: ${res.problems.join('; ')}`;
    line.classList.toggle('bad-text', !res.ok);
  }).catch((err) => { line.textContent = `The signature could not be checked: ${err.message}`; line.classList.add('bad-text'); });
  return line;
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
  const job_ = fp.release_job;
  const locked = fp.released ? null : (job_ && job_.state === 'running')
    ? `The release is running (${Math.round((job_.progress || 0) * 100)}%). These unlock when it finishes.`
    : `Locked until a person completes the final pass and releases this job (${fp.pending} of ${fp.total} items left). Until then this is an unreviewed AI draft: you can view it on the Watch tab.`;
  root.append(h('div', { class: 'stack' },
    retentionLine(job),
    reviewedLine(fp, job.released_counts),
    fp.released ? await manifestLine(url) : null,
    fileCard('Annotated video', job.released_counts && job.released_counts.annotated_rerendered === false
      ? 'Still the unreviewed AI draft, not the reviewed data: the source video was gone at release, so it could not be re-rendered. It is not covered by the signed manifest.'
      : 'The source video with the boxes, masks, phase and AI-draft narration drawn in, rendered from the released data.',
      url, 'annotated.mp4', out['annotated.mp4'], locked),
    fileCard('COCO dataset', 'Polygons, boxes and provenance attributes, with the final-pass decisions applied. Each annotation says whether a person reviewed it.',
      url, 'released_coco.json', out['released_coco.json'], locked),
    fileCard('Dataset bundle', 'One zip: the dataset, video, frames, masks, narration, prompts, versions and a dataset card, so every image resolves.',
      url, 'dataset_bundle.zip', out['dataset_bundle.zip'], locked),
    fileCard('Provenance', 'One row per annotation: model, prompt hash and how it was decided.',
      url, 'provenance.jsonl', out['provenance.jsonl'], locked),
    fileCard('Final-pass record', 'Every review click, with who made it and when.',
      url, 'final_pass.jsonl', out['final_pass.jsonl'], locked),
    fileCard('Signed manifest', 'Hashes of the released files, so later edits can be detected.',
      url, 'release_manifest.json', out['release_manifest.json'], locked),
    locked ? h('div', {}, h('a', { class: 'btn', href: `#/jobs/${encodeURIComponent(id)}/final-pass` }, 'Go to final pass')) : null));
  return null;
}
