import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  openDialog, closeDialog, watchDialogEdits, markDialogPristine, isDialogPristine,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/dialog.js';

// One rule for every panel: a click on the page behind a drawer closes it only
// while the operator has not typed into it (#1519).  The rule lives in
// `dialog.js`; the two drawers and the overlay scrim only ask it.
//
// The bug this file exists for: the provider drawer opened from a shim
// prefilled a name and a capability *after* the old snapshot was taken, so the
// form counted as edited from the first frame and the outside click never
// closed it.  A script's write is not an edit.

const here = dirname(fileURLToPath(import.meta.url));
const ADMIN = join(
  here,
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin',
);

/** Read a source file with its comments removed, so a comment cannot satisfy
 *  an assertion that is about wiring. */
function code(path) {
  const src = readFileSync(join(ADMIN, path), 'utf8');
  let out = '';
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    const next = src[i + 1];
    if (ch === '/' && next === '/') {
      const end = src.indexOf('\n', i);
      i = end === -1 ? src.length : end;
      continue;
    }
    if (ch === '/' && next === '*') {
      const end = src.indexOf('*/', i + 2);
      i = end === -1 ? src.length : end + 2;
      continue;
    }
    if (ch === "'" || ch === '"' || ch === '`') {
      const quote = ch;
      out += ch;
      i += 1;
      while (i < src.length) {
        out += src[i];
        if (src[i] === '\\') { out += src[i + 1] ?? ''; i += 2; continue; }
        if (src[i] === quote) { i += 1; break; }
        i += 1;
      }
      continue;
    }
    out += ch;
    i += 1;
  }
  return out;
}

/** One function's own text, so an assertion cannot be satisfied by a line in
 *  a later function (which is how a removed call can still look present). */
function functionBody(src, name) {
  const start = src.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `function ${name} was not found`);
  const end = src.indexOf('\n}\n', start);
  return src.slice(start, end === -1 ? src.length : end);
}

/** The smallest thing `dialog.js` needs: listeners it can register, and a way
 *  to fire them the way a person would. */
function fakePanel() {
  const listeners = {};
  return {
    listeners,
    addEventListener(type, fn) { (listeners[type] ||= []).push(fn); },
    fire(type) { for (const fn of listeners[type] || []) fn({ currentTarget: this }); },
  };
}

test('a panel nobody has touched is pristine', () => {
  const panel = fakePanel();

  watchDialogEdits(panel);

  assert.equal(isDialogPristine(panel), true);
});

test('a value the script writes is not an edit', () => {
  // The shim prefill, the key fetched after the drawer opened, and a pane
  // rebuild all assign `.value`; none of them fires an event a person would.
  const panel = fakePanel();
  watchDialogEdits(panel);

  panel.value = 'suggested-provider-name';
  panel.checked = true;

  assert.equal(isDialogPristine(panel), true);
});

test('typing and picking both count as edits', () => {
  const typing = fakePanel();
  watchDialogEdits(typing);
  typing.fire('input');
  assert.equal(isDialogPristine(typing), false);

  const picking = fakePanel();
  watchDialogEdits(picking);
  picking.fire('change');
  assert.equal(isDialogPristine(picking), false);
});

test('a reset makes the panel pristine again', () => {
  const panel = fakePanel();
  watchDialogEdits(panel);
  panel.fire('input');

  markDialogPristine(panel);

  assert.equal(isDialogPristine(panel), true);
});

test('a missing panel never blocks a close', () => {
  assert.equal(isDialogPristine(null), true);
  assert.equal(isDialogPristine(undefined), true);
});

test('watching a panel twice does not double the listeners', () => {
  const panel = fakePanel();

  watchDialogEdits(panel);
  watchDialogEdits(panel);

  assert.equal(panel.listeners.input.length, 1);
  assert.equal(panel.listeners.change.length, 1);
});

/** Enough of a DOM for `openDialog`: it locks the page and focuses the panel. */
function stubDocument() {
  globalThis.document = {
    scrollingElement: { style: {} },
    addEventListener() {},
    removeEventListener() {},
  };
}

test('an overlay answers for the dialog inside it', () => {
  // The provider drawer's scrim is the overlay; the dialog — and so the
  // listener that records an edit — is the panel inside it.  Reading the
  // pristine flag off the overlay must reach the same panel, which is the bug
  // that made a typed-in provider drawer close on an outside click.
  stubDocument();
  const panel = fakePanel();
  panel.focus = () => {};
  const overlay = { contains: (node) => node === panel };
  openDialog(panel);

  assert.equal(isDialogPristine(overlay), true, 'a fresh dialog is pristine');
  panel.fire('input');
  assert.equal(isDialogPristine(overlay), false, 'typing in it is not');
  markDialogPristine(overlay);
  assert.equal(isDialogPristine(overlay), true, 'and asking the overlay resets it');

  closeDialog(panel);
});

test('a reopened dialog is clean again', () => {
  stubDocument();
  const panel = fakePanel();
  panel.focus = () => {};
  openDialog(panel);
  panel.fire('change');
  assert.equal(isDialogPristine(panel), false);

  closeDialog(panel);
  openDialog(panel);

  assert.equal(isDialogPristine(panel), true);
  closeDialog(panel);
});

test('opening a dialog watches it and starts it clean', () => {
  const body = functionBody(code('js/core/dialog.js'), 'openDialog');

  assert.match(body, /watchDialogEdits\(panel\);/);
  assert.match(body, /markDialogPristine\(panel\);/);
});

test('the overlay scrim asks the rule, not the provider module', () => {
  const init = code('js/core/init.js');

  assert.match(init, /from '\.\/dialog\.js'/);
  assert.match(init, /rule === 'pristine' && !isDialogPristine\(m\)/);
  // The old indirect lookup through `window` let a module's absence read as
  // "pristine", which is how a missing predicate silently disabled the guard.
  assert.doesNotMatch(init, /window\.providerFormPristine/);
});

test('the provider drawer delegates instead of snapshotting', () => {
  const providers = code('js/tabs/providers.js');

  assert.match(providers, /return isDialogPristine\(overlay\);/);
  // The snapshot compared field values taken at open, so a prefill the page
  // wrote afterwards made the form look edited forever.
  assert.doesNotMatch(providers, /_providerFormSnapshot/);
  assert.doesNotMatch(providers, /_providerFormState/);
});

test('the model drawer follows the same rule', () => {
  const models = code('js/tabs/models.js');
  const reset = "markDialogPristine(document.getElementById('modelDrawer'));";

  // Its outside click used to close the panel unconditionally.
  assert.match(
    models,
    /if \(!isDialogPristine\(el\)\) return;\s*closeModelDrawer\(\);/,
  );
  // Neither a pane switch nor a re-render may clear the flag.  Both panes are
  // static DOM toggled with `hidden`, so the form keeps what was typed; and
  // assigning `innerHTML` fires no `input`/`change`, so the summary repaint
  // never dirtied anything to clear.  A reset in either place would let an
  // outside click drop a half-typed model form with no warning.  Each
  // function is checked on its own: the absence in one must not stand in for
  // the absence in the other.
  assert.ok(!functionBody(models, '_setDrawerPane').includes(reset));
  assert.ok(!functionBody(models, '_syncModelDrawer').includes(reset));
});
