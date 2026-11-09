/**
 * What a bulk model add actually did, as toast parts (#1436).
 *
 * `POST /admin/api/config/models` answers with up to four name lists —
 * `added`, `appended`, `skipped`, `ignored` — and, when nothing changed and
 * nothing was refused, a `message`.  The dialog used to read only `added` and
 * `appended`, so a batch the server refused as `ignored` fell through to "All
 * selected models already exist" (the wrong reason) and a refused entry in a
 * mixed batch disappeared.
 *
 * The whole "response → what to say" step lives here, away from the DOM, so a
 * unit test can reach the failure path too: `api.post` resolves with the parsed
 * body whatever the status, so a `500` arrives as `{error, saved, reloaded}`
 * with **no `ok`** — a caller that only asks `if (res.ok)` treats it as "the
 * batch changed nothing" and shows the success fallback over the error toast.
 *
 * Measured shapes (gateway 0.14 dev, real HTTP against a running server):
 *
 * - all added:    `200 {ok:true, added:[…], appended:[], skipped:[], ignored:[], models:[…]}`
 * - all ignored:  `200 {ok:true, added:[], appended:[], skipped:[], ignored:[…]}`
 *                 — `ok` stays true and `message` is absent, so the client
 *                 cannot rely on either to tell this apart from "all exist"
 * - all exist:    `200 {ok:true, …, ignored:[], message:"All models already exist"}`
 * - appended and ignored together:
 *                 `500 {error:"Config saved but reload failed: …", saved:true, reloaded:false}`
 *                 — an entry the loader cannot represent is also one it cannot
 *                 parse, so a mixed batch commits and then fails its reload
 */

/** How many refused names a toast spells out before it counts the rest. */
const NAMES_SHOWN = 5;

/**
 * The server's wording for "nothing to do"; localized instead of shown.
 *
 * This is a literal shared with `admin/routes/config/bulk.py` (the
 * `response["message"] = "All models already exist"` line, around :278):
 * change one and the other must change with it, or the panel stops
 * recognizing the sentence and shows it untranslated.  Nothing enforces the
 * pair — this comment and the test beside `fallbackMessage` are the guard.
 */
export const ALL_EXIST_MESSAGE = 'All models already exist';

function _names(value) {
  return Array.isArray(value)
    ? value.filter((name) => typeof name === 'string' && name)
    : [];
}

function _describe(names) {
  if (names.length <= NAMES_SHOWN) return names.join(', ');
  return `${names.slice(0, NAMES_SHOWN).join(', ')} (+${names.length - NAMES_SHOWN})`;
}

/**
 * Read a successful bulk-add response into the parts the toast is built from.
 *
 * Args:
 *   res: The parsed response body; `null`/`undefined` is read as "nothing".
 *
 * Returns:
 *   `{added, appended, skipped, ignored, message, parts}` — the name lists,
 *   the server's own message when it sent one, and the toast parts in the
 *   order they are shown.  `parts` is empty only when the server reported no
 *   change and no refusal, which is the one case that may fall back to
 *   "already exist".
 */
export function bulkAddOutcome(res) {
  const payload = res && typeof res === 'object' ? res : {};
  const added = _names(payload.added);
  const appended = _names(payload.appended);
  const skipped = _names(payload.skipped);
  const ignored = _names(payload.ignored);
  const message = typeof payload.message === 'string' ? payload.message : '';

  const parts = [];
  if (added.length) {
    parts.push({ key: 'toast.modelsAdded', count: added.length });
  }
  if (appended.length) {
    parts.push({ key: 'toast.providersAppended', count: appended.length });
  }
  if (ignored.length) {
    parts.push({
      key: 'toast.modelsIgnored',
      count: ignored.length,
      names: _describe(ignored),
    });
  }
  return { added, appended, skipped, ignored, message, parts };
}

/**
 * Decide what a response means before anything is shown.
 *
 * A response is a success only when it says so: `ok:true`.  Everything else —
 * an `{error}` envelope with a status the wrapper does not raise on, or a body
 * without `ok` — is a failure, and the caller must stop and show its error
 * rather than fall through to the "nothing changed" wording.
 *
 * Args:
 *   res: The parsed response body.
 *
 * Returns:
 *   `{ok: true, …}` with the outcome fields, or `{ok: false, error}`.
 */
export function bulkAddResult(res) {
  const payload = res && typeof res === 'object' ? res : {};
  if (payload.ok !== true) {
    return {
      ok: false,
      error: typeof payload.error === 'string' ? payload.error : '',
    };
  }
  return { ok: true, ...bulkAddOutcome(payload) };
}

/**
 * What to say when no toast part was produced.
 *
 * The server's own wording is used only when it says something the panel does
 * not already have: its `All models already exist` default would otherwise
 * reach a Chinese panel in English (#1436 review).
 *
 * Args:
 *   outcome: The result of :func:`bulkAddOutcome`.
 *
 * Returns:
 *   `{key}` for a translatable message, or `{text}` for the server's own.
 */
export function fallbackMessage(outcome) {
  const message = (outcome && outcome.message) || '';
  if (message && message !== ALL_EXIST_MESSAGE) return { text: message };
  return { key: 'toast.modelsAllExist' };
}

/**
 * Whether a set of toast parts is a refusal rather than a success.
 *
 * A batch that changed nothing and was refused says nothing about the models
 * existing, so it is shown as an error; anything that also added, appended or
 * removed stays a success toast with the refusal named in it.
 */
export function isRefusalOnly(parts, removedCount = 0) {
  if (!parts || parts.length === 0) return false;
  if (removedCount > 0) return false;
  return parts.every((part) => part.key === 'toast.modelsIgnored');
}

/**
 * The toast to show for a bulk-add result: its text and its kind.
 *
 * Both ends of the dialog go through here — the failure envelope and the
 * success summary — so the wording and the colour are decided once and can be
 * tested without a DOM or a running panel.  Translation is injected, because
 * this module deliberately imports nothing: pass `t` from `core/i18n.js`.
 *
 * Args:
 *   result: The result of :func:`bulkAddResult`, or an outcome.
 *   removedCount: Models deleted in the same pass; they are part of the
 *     summary and they turn a refusal-only batch into a success.
 *   tr: The translation function `(key, params) => string`.  Without it the
 *     keys are returned as their own text, which is what the tests read.
 *
 * Returns:
 *   `{text, kind}` — `kind` is `'success'` or `'error'`, the two classes the
 *   panel's toast styles.
 */
export function toastFor(result, { removedCount = 0, tr } = {}) {
  const translate = typeof tr === 'function' ? tr : (key) => key;
  if (result && result.ok === false) {
    return { text: result.error || translate('toast.error'), kind: 'error' };
  }

  const parts = (result && result.parts) || [];
  const msgs = parts.map((part) =>
    translate(part.key, { count: part.count, names: part.names }),
  );
  if (removedCount > 0) {
    msgs.push(translate('toast.modelsRemoved', { count: removedCount }));
  }
  if (msgs.length > 0) {
    return {
      text: msgs.join(', '),
      kind: isRefusalOnly(parts, removedCount) ? 'error' : 'success',
    };
  }
  const fallback = fallbackMessage(result);
  return { text: fallback.text || translate(fallback.key), kind: 'success' };
}
