// dialog.js — the rules a dialog follows, in one place (issue #1472 P5).
//
// A dialog owns the page while it is open: the page behind it does not
// scroll, Tab cycles inside it, Escape closes it, and focus comes back to
// whatever opened it.  The fourteen overlays and the model side panel use
// this module, so a rule changed here changes every dialog at once.
//
// One of those rules is about dismissal: a click on the page behind a panel
// closes it only while the operator has not typed into it
// (`isDialogPristine`).  A prefill the script wrote is not an edit, so a
// drawer that opened with a suggested name and a pre-ticked capability is
// still dismissable; `✕`, Cancel and Escape always close.

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

/** Panels the operator has typed into since they opened. */
const _edited = new WeakSet();
/** Panels already watched, so a second open adds no second listener. */
const _watched = new WeakSet();

function _markEdited(e) {
  // `input` and `change` are fired by a person; `el.value = …` is not.  That
  // is the whole discriminator: a prefill (a shim pick, a key fetched after
  // the drawer opened, a pane switch) is not an edit, and only an edit makes
  // a panel too precious to throw away with a click on the page behind it.
  if (e.currentTarget) _edited.add(e.currentTarget);
}

/**
 * Record a dialog's content as the operator's to lose.
 *
 * Called when the dialog opens and whenever the code re-renders it.
 *
 * @param {HTMLElement} panel - the element passed to {@link openDialog}.
 */
export function watchDialogEdits(panel) {
  if (!panel || _watched.has(panel)) return;
  _watched.add(panel);
  panel.addEventListener('input', _markEdited);
  panel.addEventListener('change', _markEdited);
}

/** The open dialog an element belongs to.
 *
 * A caller holds whichever element it has: the side panel is the dialog
 * itself, while a modal's scrim is the overlay *around* the dialog.  Both
 * must answer about the same panel, or the scrim would read a panel nobody
 * ever watched and always report it untouched.
 */
function _panelFor(el) {
  if (!el) return null;
  if (_stack.some((d) => d.panel === el)) return el;
  const owner = _stack.find(
    (d) => typeof el.contains === 'function' && el.contains(d.panel),
  );
  return owner ? owner.panel : el;
}

/** Forget the operator's edits: the panel now holds what the page wrote.
 *
 * @param {HTMLElement} el - the panel, or the overlay around it.
 */
export function markDialogPristine(el) {
  const panel = _panelFor(el);
  if (panel) _edited.delete(panel);
}

/**
 * Whether the operator has touched *el*'s dialog since it opened or was reset.
 *
 * A scrim click may throw away a panel that answers `true`, and must not
 * throw away one that answers `false` — `✕`, Cancel and Escape remain the
 * explicit ways out (#1519).
 *
 * @param {HTMLElement} el - the panel, or the overlay around it.
 * @returns {boolean}
 */
export function isDialogPristine(el) {
  const panel = _panelFor(el);
  return !panel || !_edited.has(panel);
}


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
  // Watched before the first event can arrive, and reset here so a panel
  // reopened for a new subject is not still carrying the last subject's edits.
  watchDialogEdits(panel);
  markDialogPristine(panel);
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
