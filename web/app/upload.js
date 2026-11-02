import { h, api, ApiError, fmtBytes, fmtTime } from '/lib.js';

const ALLOWED = /\.(mp4|mov)$/i;

function upload(file, onProgress) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
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
  const drop = h('label', { class: 'drop', htmlFor: 'file-input', id: 'drop' },
    input,
    h('strong', {}, 'Drop a laparoscopic video'),
    h('span', { class: 'muted' }, 'mp4 or mov, up to 5 minutes'));
  const list = h('div', { class: 'joblist' });
  const listCard = h('section', { class: 'card', hidden: true }, h('h2', {}, 'Jobs'), list);

  root.append(h('div', { class: 'stack' }, drop, bar, message, listCard));

  const showError = (text) => message.replaceChildren(h('div', { class: 'notice error', role: 'alert' }, text));

  let busy = false;
  async function send(file) {
    if (busy || !file) return;
    message.replaceChildren();
    if (!ALLOWED.test(file.name)) { showError(`${file.name} is not an mp4 or mov file.`); return; }
    busy = true;
    bar.hidden = false;
    barFill.style.width = '0';
    message.replaceChildren(h('p', { class: 'muted small' }, `Uploading ${file.name} (${fmtBytes(file.size)})`));
    try {
      const job = await upload(file, (fraction) => { barFill.style.width = `${fraction * 100}%`; });
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
      h('span', { class: 'grow jobname' }, job.filename),
      job.duration ? h('span', { class: 'muted small mono' }, fmtTime(job.duration)) : null,
      job.released ? h('span', { class: 'pill done' }, 'released') : null,
      h('span', { class: `pill ${job.state}` }, job.state))));
  }

  const health = await api('/api/health');
  const missing = ['ffmpeg', 'ffprobe', 'claude'].filter((tool) => !health[tool]);
  if (missing.length) {
    root.prepend(h('div', { class: 'notice', style: 'margin-bottom:16px' },
      `Not found on this machine: ${missing.join(', ')}. The pipeline needs them.`));
  }
  await refresh();
  const timer = setInterval(() => refresh().catch(() => {}), 3000);
  return () => clearInterval(timer);
}
