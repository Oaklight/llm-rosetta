/**
 * Which i18n keys the panel's JS asks for, and whether both dictionaries have
 * them (#1441).
 *
 * `t()` falls back to the key itself when a sentence is missing, so a
 * misspelled or deleted key is a silent failure: the operator sees
 * `toast.modelsIgnored` on screen.  Nothing else in the suite notices — the
 * component tests inject a fake translator that returns the key by design.
 *
 * This module is the scanner; `i18n-keys.test.mjs` is the guard.  It lives in
 * `tests/` because it reads the sources as text (i18n.js imports state.js,
 * which touches localStorage, so Node cannot import the dictionary).
 *
 * What is checked:
 *
 * - every string literal passed to `t(...)`, single or double quoted;
 * - every `key: '…'` in an object literal whose first segment is a dictionary
 *   namespace — the shape `toastFor()` returns and the components read;
 * - literals inside a `t(ternary …)` argument, when they are key-shaped.
 *
 * What is listed rather than checked: any `t(...)` whose argument is not a
 * literal (`t(key)`, `t(EVENT_KEYS[e.type])`, `t('toast.' + name)`).  They are
 * reported by file and line — a guard that quietly checks nothing is the
 * failure mode this module exists to prevent.
 */

/** Split i18n.js into its two dictionary blocks. */
export function dictionarySections(source) {
  const enStart = source.indexOf('  en: {');
  const zhStart = source.indexOf('  zh: {');
  const end = source.lastIndexOf('\n};');
  if (enStart < 0 || zhStart < 0 || end < zhStart) {
    throw new Error('i18n.js: cannot find the en/zh dictionary blocks');
  }
  return { en: source.slice(enStart, zhStart), zh: source.slice(zhStart, end) };
}

/** The keys a dictionary block declares, in order of appearance. */
export function keysIn(section) {
  return new Set(
    [...section.matchAll(/'([A-Za-z][A-Za-z0-9_.]*)'\s*:/g)]
      .map((m) => m[1])
      .filter((key) => key.includes('.')),
  );
}

/**
 * The keys the two dictionaries disagree on.
 *
 * This half needs no JS reference: a key that only one language declares is a
 * defect wherever it is used — when the panel is in the other language, `t()`
 * renders the key itself.  Returned as `{onlyEn, onlyZh}`, sorted, so a
 * failure reads like a list and a passing case reads as two empty arrays.
 */
export function dictionaryDiff(en, zh) {
  const onlyEn = [...en].filter((key) => !zh.has(key)).sort();
  const onlyZh = [...zh].filter((key) => !en.has(key)).sort();
  return { onlyEn, onlyZh };
}

/**
 * Keys a dictionary block declares more than once, with every value.
 *
 * A later declaration silently wins, so the earlier sentence is dead text that
 * a translator keeps maintaining for nothing — and the two values can differ
 * in meaning, not just in wording.  Values are returned as written (escapes
 * intact): the point is to show what was overridden, and the file is where a
 * reader will go looking.
 */
export function duplicateKeys(section) {
  const seen = new Map();
  for (const match of section.matchAll(
    /'([A-Za-z][A-Za-z0-9_.]*)'\s*:\s*("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/g,
  )) {
    const key = match[1];
    if (!key.includes('.')) continue;
    if (!seen.has(key)) seen.set(key, []);
    seen.get(key).push(match[2]);
  }
  return new Map([...seen].filter(([, values]) => values.length > 1));
}

/** `"a.b.c"` is a key; `"free"` and `"some-id"` are not. */
export function isKeyShaped(text, namespaces) {
  if (!text.includes('.')) return false;
  return namespaces.has(text.split('.')[0]);
}

function lineOf(text, index) {
  let line = 1;
  for (let i = 0; i < index && i < text.length; i += 1) {
    if (text[i] === '\n') line += 1;
  }
  return line;
}

/** The text of `t(`'s first argument, paren-balanced, plus its offset. */
function firstArgument(text, openParen) {
  let depth = 0;
  let quote = null;
  for (let i = openParen; i < text.length; i += 1) {
    const ch = text[i];
    if (quote) {
      if (ch === '\\') i += 1;
      else if (ch === quote) quote = null;
      continue;
    }
    if (ch === "'" || ch === '"' || ch === '`') {
      quote = ch;
      continue;
    }
    if (ch === '(') depth += 1;
    else if (ch === ')') {
      depth -= 1;
      if (depth === 0) return { text: text.slice(openParen + 1, i), start: openParen + 1 };
    } else if (ch === ',' && depth === 1) {
      return { text: text.slice(openParen + 1, i), start: openParen + 1 };
    }
  }
  return { text: text.slice(openParen + 1), start: openParen + 1 };
}

/**
 * Read one source file.
 *
 * Returns `{literals, unresolved}`; every entry carries `line` so a failure
 * names the place an operator would look.
 */
export function scanSource(source, namespaces) {
  const literals = [];
  const unresolved = [];

  for (const match of source.matchAll(/\bt\(/g)) {
    const openParen = match.index + 1;
    let cursor = openParen + 1;
    while (cursor < source.length && /\s/.test(source[cursor])) cursor += 1;
    const first = source[cursor];

    if (first === "'" || first === '"') {
      let end = cursor + 1;
      while (end < source.length) {
        if (source[end] === '\\') end += 2;
        else if (source[end] === first) break;
        else end += 1;
      }
      // `'toast.' + name` builds the key; checking the fragment would report a
      // key nobody asks for.  It is a computed key, so it is listed instead.
      const after = source.slice(end + 1);
      if (/^\s*\+/.test(after) || first === '`') {
        const arg = firstArgument(source, openParen);
        unresolved.push({
          expr: `t(${arg.text.trim().slice(0, 60)})`,
          line: lineOf(source, match.index),
        });
        continue;
      }
      literals.push({
        key: source.slice(cursor + 1, end),
        line: lineOf(source, match.index),
      });
      continue;
    }

    if (first === '`') {
      unresolved.push({ expr: 't(`…`)', line: lineOf(source, match.index) });
      continue;
    }

    const arg = firstArgument(source, openParen);
    const expr = arg.text.trim();
    unresolved.push({ expr: `t(${expr.slice(0, 60)})`, line: lineOf(source, match.index) });
    // A ternary still spells its keys out: check the ones that are key-shaped.
    for (const inner of arg.text.matchAll(/'((?:\\.|[^'])*)'/g)) {
      if (isKeyShaped(inner[1], namespaces)) {
        literals.push({
          key: inner[1],
          line: lineOf(source, arg.start + (inner.index || 0)),
        });
      }
    }
  }

  for (const match of source.matchAll(/\bkey:\s*(['"])([^'"]+)\1/g)) {
    if (isKeyShaped(match[2], namespaces)) {
      literals.push({ key: match[2], line: lineOf(source, match.index) });
    }
  }

  return { literals, unresolved };
}

/**
 * Compare references against the two dictionaries.
 *
 * Returns `{missing, unresolved}`: `missing` is `[{key, languages, file, line}]`
 * with the languages that do not declare it; `unresolved` is `[{file, line,
 * expr}]`.
 */
export function checkSources(sources, dictionaries) {
  const namespaces = new Set(
    [...dictionaries.en, ...dictionaries.zh].map((key) => key.split('.')[0]),
  );
  const missing = [];
  const unresolved = [];
  for (const [file, source] of sources) {
    const { literals, unresolved: dynamics } = scanSource(source, namespaces);
    for (const { key, line } of literals) {
      const languages = [];
      if (!dictionaries.en.has(key)) languages.push('en');
      if (!dictionaries.zh.has(key)) languages.push('zh');
      if (languages.length > 0) missing.push({ key, languages, file, line });
    }
    for (const entry of dynamics) unresolved.push({ file, ...entry });
  }
  return { missing, unresolved };
}
