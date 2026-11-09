/**
 * themes.js — the admin theme registry, as the gateway injected it.
 *
 * The list of themes lives in `admin/themes.py` and reaches the page as a JSON
 * script tag (`#adminThemes`).  This module is the page's only reader of it, so
 * a new theme is a stylesheet plus a registry entry — not an edit here
 * (issue #1471).
 *
 * The pure helpers take the registry as an argument so the node tests can
 * exercise them without a DOM; the wrappers below read the injected one.
 */

const REGISTRY_ID = 'adminThemes';
const SHEET_ID = 'themeSheet';
const LEGACY_NAME = 'minimal';

let _registry = null;

/**
 * Normalise the injected registry.
 *
 * @param {any} raw Parsed JSON (or anything else).
 * @returns {{default: string, themes: Array<{name: string, label: string, url: string}>}}
 */
export function parseRegistry(raw) {
  const themes = Array.isArray(raw && raw.themes)
    ? raw.themes
      .filter((theme) => theme && typeof theme.name === 'string' && theme.name)
      .map((theme) => ({
        name: theme.name,
        label: typeof theme.label === 'string' && theme.label ? theme.label : theme.name,
        url: typeof theme.url === 'string' ? theme.url : '',
      }))
    : [];
  const fallback = themes.length ? themes[0].name : '';
  const name = typeof (raw && raw.default) === 'string' ? raw.default : fallback;
  return { default: themes.some((theme) => theme.name === name) ? name : fallback, themes };
}

/** The injected registry, parsed once. */
export function registry() {
  if (_registry) return _registry;
  let raw = null;
  try {
    const el = typeof document !== 'undefined' && document.getElementById(REGISTRY_ID);
    if (el) raw = JSON.parse(el.textContent);
  } catch {
    // A malformed registry must not break the panel: the seed colours keep it
    // readable and `default` being empty leaves every setter a no-op.
    raw = null;
  }
  _registry = parseRegistry(raw);
  return _registry;
}

/** The theme the panel uses when nothing (valid) is stored. */
export function defaultThemeName() { return registry().default; }

/** Whether *name* is one of the injected themes. */
export function themeByName(name) {
  return registry().themes.find((theme) => theme.name === name) || null;
}

/**
 * A stored scheme name, or the default when the build no longer ships it.
 *
 * @param {{default: string, themes: Array<{name: string}>}} reg
 * @param {string|null} stored
 * @returns {string}
 */
export function resolveSchemeIn(reg, stored) {
  return reg.themes.some((theme) => theme.name === stored) ? stored : reg.default;
}

/** {@link resolveSchemeIn} against the injected registry. */
export function resolveScheme(stored) { return resolveSchemeIn(registry(), stored); }

/** The Settings picker's options, in registry order. */
export function schemeChoices() {
  return registry().themes.map((theme) => ({ name: theme.name, label: theme.label }));
}

/**
 * The theme to fall back to after a load failure, or `null` when there is none
 * left to try.  The default is tried once; a default that fails too is not
 * re-requested, so a broken install cannot spin.
 *
 * @param {Set<string>} attempted Theme names already tried.
 * @param {string} defaultName
 * @returns {string|null}
 */
export function pickFallback(attempted, defaultName) {
  if (!defaultName) return null;
  return attempted && attempted.has(defaultName) ? null : defaultName;
}

/** Point the theme sheet at *name* (validated, falling back to the default). */
export function setThemeSheet(name) {
  const link = document.getElementById(SHEET_ID);
  const theme = themeByName(resolveScheme(name));
  if (!link || !theme) return;
  if (link.getAttribute('href') !== theme.url) link.setAttribute('href', theme.url);
}

/**
 * Keep the panel styled when a theme does not load.
 *
 * Two failures are possible: the request fails (`error`), or it answers with
 * something that is not CSS (`load` with `sheet === null`).  Both fall back to
 * the default once; the seed colours in the page keep it readable if even that
 * fails.
 */
export function installThemeFallback() {
  const link = document.getElementById(SHEET_ID);
  if (!link || link.dataset.fallbackWired) return;
  link.dataset.fallbackWired = '1';
  const attempted = new Set();
  const fallback = () => {
    const name = pickFallback(attempted, defaultThemeName());
    if (!name) return;
    attempted.add(name);
    const theme = themeByName(name);
    if (!theme || link.getAttribute('href') === theme.url) return;
    link.setAttribute('href', theme.url);
    document.documentElement.setAttribute('data-scheme', name);
  };
  link.addEventListener('error', fallback);
  link.addEventListener('load', () => {
    if (!link.sheet) fallback();
  });
}

/**
 * Fill the Settings picker from the registry.
 *
 * @param {HTMLSelectElement|null} select
 * @param {(name: string, label: string) => string} labelFor
 *   Translates a theme's label; return *label* unchanged when there is no
 *   translation, and the option keeps the registry's English label.
 */
export function populateSchemeSelect(select, labelFor) {
  if (!select || select.dataset.built) return;
  select.dataset.built = '1';
  select.textContent = '';
  for (const { name, label } of schemeChoices()) {
    const option = document.createElement('option');
    option.value = name;
    const text = labelFor ? labelFor(name, label) : label;
    option.textContent = text;
    // Only a real translation is re-applied on a language change; a theme
    // without one keeps the registry's label rather than rendering its key.
    if (text !== label) option.dataset.i18n = 'scheme.' + name;
    select.appendChild(option);
  }
}

/** The legacy scheme name, for migrations that need a concrete default. */
export const legacySchemeName = LEGACY_NAME;
