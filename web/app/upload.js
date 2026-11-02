import { h, api, ApiError, fmtBytes, fmtTime } from '/lib.js';

const ALLOWED = /\.(mp4|mov)$/i;

function upload(file, attestation, onProgress) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    Object.entries(attestation).forEach(([key, value]) => form.append(key, value));
    form.append('file', file);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/jobs');
    xhr.responseType = 'json';
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    xhr.onload = () => {
      if (xhr.status === 201) resolve(xhr.response);
      else {
        const detail = xhr.response && xhr.response.detail;
        reject(new ApiError(xhr.status, typeof detail === 'string' ? detail : `Upload failed (${xhr.status})`));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, 'Upload failed: the server could not be reached.'));
    xhr.send(form);
  });
}

export async function renderUpload(root) {
  const message = h('div');
  const barFill = h('div', { style: 'width:0' });
  const bar = h('div', { class: 'bar', hidden: true }, barFill);
  const input = h('input', { type: 'file', accept: '.mp4,.mov,video/mp4,video/quicktime', id: 'file-input' });
  const notice = await api('/api/notice');
  const health = await api('/api/health');
  input.disabled = true;
  const drop = h('label', { class: 'drop blocked', htmlFor: 'file-input', id: 'drop' },
    input,
    h('strong', {}, 'Drop a laparoscopic cholecystectomy video'),
    h('span', { class: 'muted' }, 'mp4 or mov, up to 5 minutes. Public or simulator footage only.'));

  // Blocking: nothing can be chosen or dropped until this is answered, for each upload.
  const ticked = h('input', { type: 'checkbox', id: 'attest-rights' });
  const sources = notice.footage_sources.map((value) => h('input', { type: 'radio', name: 'footage-source', value, id: `source-${value}` }));
  const syncGate = () => {
    const open = ticked.checked && sources.some((r) => r.checked);
    input.disabled = !open;
    drop.classList.toggle('blocked', !open);
  };
  ticked.addEventListener('change', syncGate);
  sources.forEach((r) => r.addEventListener('change', syncGate));
  const noticeCard = h('section', { class: 'card stack', id: 'upload-notice' },
    h('h2', { style: 'margin:0' }, 'Before you upload'),
    h('p', { class: 'small', id: 'notice-text' }, notice.text),
    h('fieldset', { class: 'source' },
      h('legend', { class: 'small muted' }, 'This footage is'),
      sources.map((radio) => h('label', { class: 'row', style: 'gap:6px' }, radio, radio.value))),
    h('label', { class: 'attest' }, ticked, h('span', {}, 'I have read this, and I have the right to use this footage and send it to the model provider.')));
  const list = h('div', { class: 'joblist' });
  const listCard = h('section', { class: 'card', hidden: true }, h('h2', {}, 'Jobs'), list);

  root.append(h('div', { class: 'stack' }, noticeCard, drop, bar, message, listCard));

  const showError = (text) => message.replaceChildren(h('div', { class: 'notice error', role: 'alert' }, text));

  let busy = false;
  async function send(file) {
    if (busy || !file) return;
    message.replaceChildren();
    if (input.disabled) { showError('Answer the notice above first.'); return; }
    if (!ALLOWED.test(file.name)) { showError(`${file.name} is not an mp4 or mov file.`); return; }
    if (file.size > health.max_bytes) { showError(`This file is ${fmtBytes(file.size)}, the limit is ${fmtBytes(health.max_bytes)}.`); return; }
    busy = true;
    bar.hidden = false;
    barFill.style.width = '0';
    message.replaceChildren(h('p', { class: 'muted small' }, `Uploading ${file.name} (${fmtBytes(file.size)})`));
    try {
      const attestation = {
        rights_attested: 'yes',
        notice_sha256: notice.sha256,
        footage_source: sources.find((r) => r.checked).value,
      };
      const job = await upload(file, attestation, (fraction) => { barFill.style.width = `${fraction * 100}%`; });
      location.hash = `#/jobs/${encodeURIComponent(job.id)}`;
    } catch (err) {
      showError(err.message);
      bar.hidden = true;
    } finally {
      busy = false;
      input.value = '';
    }
  }

  input.addEventListener('change', () => send(input.files[0]));
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('over');
    send(e.dataTransfer.files[0]);
  });

  async function refresh() {
    const { jobs } = await api('/api/jobs');
    listCard.hidden = jobs.length === 0;
    list.replaceChildren(...jobs.map((job) => h('a', { href: `#/jobs/${encodeURIComponent(job.id)}` },
      h('span', { class: 'grow jobname' }, job.name),
      job.duration ? h('span', { class: 'muted small mono' }, fmtTime(job.duration)) : null,
      job.released ? h('span', { class: 'pill done' }, 'released') : null,
      h('span', { class: `pill ${job.state}` }, job.state))));
  }

  const problems = [];
  const missing = Object.entries(health.tools).filter(([, found]) => !found).map(([tool]) => tool);
  if (missing.length) problems.push(`Not found on this machine: ${missing.join(', ')}.`);
  if (!health.model.ok) problems.push(`Model: ${health.model.detail}.`);
  if (!health.disk.ok) problems.push(`Low disk: ${fmtBytes(health.disk.free_bytes)} free.`);
  if (problems.length) {
    root.prepend(h('div', { class: 'notice', id: 'health-warning', style: 'margin-bottom:16px' }, problems.join(' ')));
  }
  await refresh();
  const timer = setInterval(() => refresh().catch(() => {}), 3000);
  return () => clearInterval(timer);
}
