// key-limits.js — what a key's two limit fields accept, apart from the DOM.
//
// The dialog marks the field and the save stops; those are the caller's job.
// These are the rules, kept here so they can be tested without a browser.

/** A model name: the vendor spellings in use, and no whitespace. */
export const MODEL_NAME_RE = /^[A-Za-z0-9._:\-/@+]+$/;

/** The field's tokens, split the way the server splits them. */
export function modelTokens(raw) {
  return String(raw || '').split(/[,\s]+/).map(x => x.trim()).filter(Boolean);
}

/** The first token that is not a model name, or null. `*` is every model. */
export function badModelName(raw) {
  return modelTokens(raw).find(m => m !== '*' && !MODEL_NAME_RE.test(m)) || null;
}
