// Shared helpers. Text always goes in through textContent, so a filename or model output cannot inject markup.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') el.className = value;
    else if (key.startsWith('on')) el.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key in el && key !== 'list') el[key] = value;
    else el.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return el;
}

export class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

export async function api(path, options) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch (_) { /* no JSON body */ }
  if (!res.ok) {
    const detail = body && body.detail;
    throw new ApiError(res.status, typeof detail === 'string' ? detail : `Request failed (${res.status})`);
  }
  return body;
}

export function postJson(path, payload) {
  return api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
}

export function fmtTime(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return '';
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

export function fmtBytes(bytes) {
  if (!bytes && bytes !== 0) return '';
  if (bytes < 1e6) return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
  if (bytes < 1e9) return `${(bytes / 1e6).toFixed(1)} MB`;
  return `${(bytes / 1e9).toFixed(2)} GB`;
}

export function titleCase(text) {
  return String(text || '').replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());
}

const LABEL_COLOURS = {
  gallbladder: '#40c88c', liver: '#d6785c', grasper: '#5aaaff', l_hook: '#fabe46', cystic_duct: '#b58cf0', blood: '#e8607a',
};
export const AMBER = '#f59e0b';
export const labelColour = (label) => LABEL_COLOURS[label] || '#8a909e';
export const LABEL_NAMES = { l_hook: 'L-hook', cystic_duct: 'Cystic duct' };
export const labelName = (label) => LABEL_NAMES[label] || titleCase(label);

export function tabs(job, active) {
  const done = job.state === 'done';
  const base = `#/jobs/${encodeURIComponent(job.id)}`;
  const item = (key, text, href, enabled) => h('a', {
    class: `tab${active === key ? ' on' : ''}${enabled ? '' : ' off'}`,
    href,
    'aria-disabled': enabled ? null : 'true',
    tabIndex: enabled ? 0 : -1,
  }, text);
  return h('nav', { class: 'tabs' },
    item('job', 'Progress', base, true),
    item('watch', 'Watch', `${base}/watch`, done),
    item('final', 'Final pass', `${base}/final-pass`, done),
    item('download', 'Download', `${base}/download`, done));
}

export function jobHeader(job, active) {
  return h('div', {},
    h('div', { class: 'row spread' },
      h('h1', { class: 'jobname' }, job.filename),
      h('span', { class: `pill ${job.state}` }, job.state)),
    tabs(job, active));
}

// Canvas pixels per CSS pixel, so chips and line widths look the same whatever the video's resolution.
export function displayScale(canvas) {
  const shown = canvas.getBoundingClientRect().width;
  return shown > 0 ? canvas.width / shown : 1;
}

// Draw polygons in video pixels. The canvas is sized in the same pixels and scaled by CSS.
export function tracePolygon(ctx, polygon) {
  ctx.beginPath();
  polygon.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
}

export function drawChip(ctx, text, x, y, colour, scale) {
  ctx.font = `500 ${Math.round(13 * scale)}px -apple-system, "Helvetica Neue", sans-serif`;
  const padX = 6 * scale;
  const height = 20 * scale;
  const width = ctx.measureText(text).width + padX * 2;
  const left = Math.min(Math.max(0, x), ctx.canvas.width - width);
  const top = y - height < 0 ? y : y - height;
  ctx.fillStyle = colour;
  ctx.beginPath();
  ctx.roundRect(left, top, width, height, 4 * scale);
  ctx.fill();
  ctx.fillStyle = '#0a0b0e';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, left + padX, top + height / 2);
}
