import { h, ApiError, postJson } from '/lib.js';
import { renderAuth, authState } from '/auth.js';
import { renderUpload } from '/upload.js';
import { renderJob } from '/job.js';
import { renderWatch } from '/watch.js';
import { renderFinalPass } from '/finalpass.js';
import { renderDownload } from '/download.js';

const view = document.getElementById('view');
const session = document.getElementById('session');
let cleanup = null;
let token = 0;

const PAGES = { '': renderJob, watch: renderWatch, 'final-pass': renderFinalPass, download: renderDownload };

function route() {
  const parts = location.hash.replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent);
  if (parts[0] === 'jobs' && parts[1] && (parts[2] || '') in PAGES) {
    return { page: PAGES[parts[2] || ''], params: { id: parts[1] } };
  }
  return { page: renderUpload, params: {} };
}

function showSession(user) {
  session.replaceChildren(...(user ? [
    h('span', { class: 'small muted', id: 'session-user' }, user),
    h('button', {
      class: 'btn small-btn',
      id: 'sign-out',
      onclick: async () => { await postJson('/api/auth/logout', {}); navigate(); },
    }, 'Sign out'),
  ] : []));
}

async function navigate() {
  const mine = ++token;
  if (cleanup) { cleanup(); cleanup = null; }
  const root = h('div', { class: 'enter' });
  view.replaceChildren(root);
  try {
    const state = await authState();
    if (mine !== token) return;
    showSession(state.user);
    if (!state.user) {
      renderAuth(root, state, navigate);
      return;
    }
    const { page, params } = route();
    const dispose = await page(root, { ...params, user: state.user });
    if (mine === token) cleanup = dispose || null;
    else if (dispose) dispose();
  } catch (err) {
    if (mine !== token) return;
    if (err instanceof ApiError && err.status === 401) { navigate(); return; }
    const message = err instanceof ApiError && err.status === 404 ? 'This job does not exist.' : err.message;
    root.replaceChildren(h('div', { class: 'notice error' }, message));
  }
}

window.addEventListener('hashchange', navigate);
navigate();
