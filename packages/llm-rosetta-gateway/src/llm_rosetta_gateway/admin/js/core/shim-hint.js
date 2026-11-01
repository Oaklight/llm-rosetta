/**
 * shim-hint.js — Resolve a shim's panel hint for the current UI language.
 *
 * The server normalises every `hint_msg` declaration to `{language: text}`
 * (a bare YAML string becomes `{en: ...}`), so the panel only has to choose.
 * DOM-free so `node --test tests/js/` can pin the fallback order.
 */

/**
 * The hint in *lang*, falling back to English, then to the first declared
 * language — a shim with one translation still shows it.
 *
 * @param {object|null|undefined} hint The shim's `hint_msg` mapping.
 * @param {string} lang The current interface language code.
 * @returns {string} The text to show, or '' when the shim declares none.
 */
export function shimHintText(hint, lang) {
  if (!hint || typeof hint !== 'object') return '';
  return hint[lang] || hint.en || Object.values(hint)[0] || '';
}
