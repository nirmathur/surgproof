import { h, api, postJson } from '/lib.js';

// The sign-in or first-run form. Resolves once a session exists.
export function renderAuth(root, state, onDone) {
  const setup = state.setup_required;
  const message = h('div');
  const name = h('input', { class: 'input', id: 'auth-name', autocomplete: 'name', 'aria-label': 'Your name' });
  const password = h('input', { class: 'input', id: 'auth-password', type: 'password', 'aria-label': 'Password', autocomplete: setup ? 'new-password' : 'current-password' });
  const code = h('input', { class: 'input', id: 'auth-code', autocomplete: 'off', 'aria-label': 'Setup code' });
  const field = (label, input, note) => h('label', { class: 'stack field' }, h('span', { class: 'small muted' }, label), input,
    note ? h('span', { class: 'small muted' }, note) : null);

  const form = h('form', { class: 'card stack authcard', id: 'auth-form' },
    h('h1', {}, setup ? 'Create your account' : 'Sign in'),
    h('p', { class: 'small muted' }, setup
      ? 'One local account on this machine. Your name goes on every final-pass decision and release.'
      : 'Your session is what signs your final-pass decisions.'),
    setup ? field('Setup code', code, 'Printed in the terminal where you ran make serve.') : null,
    setup ? field('Your name', name) : null,
    field('Password', password, setup ? 'At least 8 characters.' : null),
    message,
    h('div', {}, h('button', { class: 'btn primary', id: 'auth-submit', type: 'submit' }, setup ? 'Create account' : 'Sign in')));

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    message.replaceChildren();
    try {
      await postJson(setup ? '/api/auth/setup' : '/api/auth/login',
        { name: name.value, password: password.value, setup_code: code.value });
      onDone();
    } catch (err) {
      message.replaceChildren(h('div', { class: 'notice error', role: 'alert' }, err.message));
    }
  });
  root.replaceChildren(form);
  (setup ? code : password).focus();
}

export const authState = () => api('/api/auth/state');
