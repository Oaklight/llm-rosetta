// dialog.js — the rules a dialog follows, in one place (issue #1472 P5).
//
// A dialog owns the page while it is open: the page behind it does not
// scroll, Tab cycles inside it, Escape closes it, and focus comes back to
// whatever opened it.  The fourteen overlays and the model side panel use
// this module, so a rule changed here changes every dialog at once.

/** What can take focus inside a dialog.  A `tabindex="-1"` container is a
 *  landing point, not a stop: it is never in this list. */
const FOCUSABLE = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',');

/** Open dialogs, innermost last.  Each entry is
 *  `{ panel, prevFocusable, onEscape }`. */
const _stack = [];
let _prevOverflow = null;

const _top = () => _stack[_stack.length - 1];

const _focusableIn = (panel) => [...panel.querySelectorAll(FOCUSABLE)]
  .filter((el) => el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden');

function _lockPage() {
  if (_stack.length !== 1) return;
  const el = document.scrollingElement || document.documentElement;
  _prevOverflow = el.style.overflow;
  el.style.overflow = 'hidden';
}

function _unlockPage() {
  if (_stack.length !== 0) return;
  const el = document.scrollingElement || document.documentElement;
  el.style.overflow = _prevOverflow == null ? '' : _prevOverflow;
  _prevOverflow = null;
}

function _onKeydown(e) {
  const top = _top();
  if (!top) return;
  if (e.key === 'Escape') {
    // A picker or dropdown inside the dialog may consume Escape; only an
    // unconsumed one closes the dialog.
    if (!e.defaultPrevented) top.onEscape?.(e);
    return;
  }
  if (e.key !== 'Tab') return;
  const items = _focusableIn(top.panel);
  if (!items.length) {
    e.preventDefault();
    top.panel.focus();
    return;
  }
  const first = items[0];
  const last = items[items.length - 1];
  const active = document.activeElement;
  // The panel itself holds focus right after opening (it is a landing point
  // and is not in the list), so both directions start from a boundary.
  if (active === top.panel || !top.panel.contains(active)) {
    e.preventDefault();
    (e.shiftKey ? last : first).focus();
  } else if (e.shiftKey && active === first) {
    e.preventDefault();
    last.focus();
  } else if (!e.shiftKey && active === last) {
    e.preventDefault();
    first.focus();
  }
}

function _onFocusIn(e) {
  const top = _top();
  if (!top) return;
  if (e.target === top.panel || top.panel.contains(e.target)) return;
  // Something outside the dialog took focus (a programmatic focus from a
  // module that does not know a dialog is open, or a click that reached the
  // page): bring it back rather than leaving the keyboard outside.
  top.panel.focus();
}

/**
 * Open a dialog: lock the page, focus the panel, trap Tab and Escape.
 *
 * @param {HTMLElement} panel - the element carrying `role="dialog"` and
 *   `tabindex="-1"`.
 * @param {{onEscape?: () => void}} [opts] - `onEscape` is called for an
 *   unconsumed Escape.  It should call {@link closeDialog}.
 */
export function openDialog(panel, opts = {}) {
  if (!panel || _stack.some((d) => d.panel === panel)) return;
  _stack.push({ panel, onEscape: opts.onEscape });
  _lockPage();
  if (_stack.length === 1) {
    document.addEventListener('keydown', _onKeydown, true);
    document.addEventListener('focusin', _onFocusIn);
  }
  panel.focus();
}

/**
 * Close a dialog: unlock the page when it was the last one, and stop
 * listening.  Restoring focus belongs to the opener, which knows what it was.
 *
 * @param {HTMLElement} panel - the panel passed to {@link openDialog}.
 */
export function closeDialog(panel) {
  const i = _stack.findIndex((d) => d.panel === panel);
  if (i < 0) return;
  _stack.splice(i, 1);
  if (_stack.length === 0) {
    document.removeEventListener('keydown', _onKeydown, true);
    document.removeEventListener('focusin', _onFocusIn);
  }
  _unlockPage();
}

/** True while any dialog is open. */
export const isDialogOpen = () => _stack.length > 0;
