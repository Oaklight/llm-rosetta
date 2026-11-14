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

/**
 * The limit tags a key row shows, as `{text, title}` pairs — empty is none.
 * `[\"*\"]` and an omitted field both mean every model and show no model
 * tag; `[]` is a real state ("no model") and says so.
 */
export function keyLimitTags(key, t) {
  const tags = [];
  const allowed = key.allowed_models;
  const models = (allowed || ['*']).filter((m) => m !== '*');
  if (allowed && allowed.length === 0) {
    tags.push({text: t('keys.noModel'), title: t('label.keyModels')});
  } else if (models.length) {
    tags.push({
      text: `${models.length} ${t(models.length === 1 ? 'keys.modelUnitOne' : 'keys.modelUnit')}`,
      title: models.join(', '),
    });
  }
  if (key.rate_limit) {
    // The row has one line for a spec that may name two windows: the first
    // shown, the whole string in the tooltip.
    const quotas = String(key.rate_limit).split(',').map((q) => q.trim()).filter(Boolean);
    tags.push({
      text: quotas[0] + (quotas.length > 1 ? ` +${quotas.length - 1}` : ''),
      title: key.rate_limit,
    });
  }
  return tags;
}
