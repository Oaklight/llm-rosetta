/**
 * Scanner behind innerhtml-lint.test.mjs.
 *
 * Finds every innerHTML / outerHTML assignment and every insertAdjacentHTML
 * call in a source text and returns the interpolations that reach the markup
 * without an escaper.  It is a text scanner (strings, comments, nested
 * template literals and balanced brackets are understood), not a parser: it
 * reads what the assignment spells out and nothing it reaches through a
 * variable.
 */

const SAFE_CALL = /^(?:esc|_esc|jsArg|raw)\s*\(/;
const SLASH = '/';
const STAR = '*';

function skipString(src, i) {
  const q = src[i];
  i++;
  while (i < src.length && src[i] !== q) i += src[i] === '\\' ? 2 : 1;
  return i + 1;
}

function skipComment(src, i) {
  if (src[i] === SLASH && src[i + 1] === SLASH) {
    const n = src.indexOf('\n', i);
    return n < 0 ? src.length : n;
  }
  if (src[i] === SLASH && src[i + 1] === STAR) {
    const n = src.indexOf(STAR + SLASH, i + 2);
    return n < 0 ? src.length : n + 2;
  }
  return i;
}

/** Parse a template literal at `i` (the opening backtick).
 *  Returns {end, interps: [{expr, templates}]}; `templates` are the template
 *  literals found inside that interpolation. */
function parseTemplate(src, i) {
  const interps = [];
  i++;
  let chunk = i; // where the literal text before the next interpolation starts
  while (i < src.length && src[i] !== '`') {
    if (src[i] === '\\') {
      i += 2;
    } else if (src[i] === '$' && src[i + 1] === '{') {
      const start = i + 2;
      const before = src.slice(chunk, i);
      const r = scanExpr(src, start, '}');
      interps.push({
        expr: src.slice(start, r.end).replace(/\s+/g, ' ').trim(),
        templates: r.templates,
        before,
      });
      i = r.end + 1;
      chunk = i;
    } else {
      i++;
    }
  }
  return { end: i + 1, interps };
}

/** Scan from `i` to the closing `stop` char (`;`, `,`, `)` or `}`) at depth 0.
 *  Returns {end, templates} — end indexes the stop char. */
function scanExpr(src, i, stop) {
  const templates = [];
  let depth = 0;
  while (i < src.length) {
    const c = src[i];
    const skipped = skipComment(src, i);
    if (skipped !== i) {
      i = skipped;
      continue;
    }
    if (c === '"' || c === "'") {
      i = skipString(src, i);
      continue;
    }
    if (c === '`') {
      const t = parseTemplate(src, i);
      // An html tagged template escapes its own interpolations.
      if (!/(?:^|[^\w$])html$/.test(src.slice(Math.max(0, i - 5), i))) templates.push(t);
      i = t.end;
      continue;
    }
    if (depth === 0 && c === stop) return { end: i, templates };
    if (c === '(' || c === '[' || c === '{') depth++;
    else if (c === ')' || c === ']' || c === '}') {
      if (depth === 0) return { end: i, templates };
      depth--;
    }
    i++;
  }
  return { end: i, templates };
}

/** Whether an interpolation is one call to an escaper covering all of it, or
 *  a nested html template. */
export function isSafeExpr(expr) {
  const e = expr.trim();
  if (e.startsWith('html`')) return parseTemplate(e, 4).end === e.length;
  if (!SAFE_CALL.test(e)) return false;
  let depth = 0;
  for (let i = 0; i < e.length; i++) {
    if (e[i] === '"' || e[i] === "'") i = skipString(e, i) - 1;
    else if (e[i] === '`') i = parseTemplate(e, i).end - 1;
    else if (e[i] === '(') depth++;
    else if (e[i] === ')') {
      depth--;
      if (depth === 0) return i === e.length - 1;
    }
  }
  return false;
}

/** The unescaped interpolations of a parsed template list.  An interpolation
 *  that holds a template of its own (a map over rows) is judged by the
 *  interpolations inside that template. */
function offenders(templates, out) {
  for (const t of templates) {
    for (const it of t.interps) {
      if (isSafeExpr(it.expr)) continue;
      if (it.templates.length) offenders(it.templates, out);
      else out.push(it.expr);
    }
  }
  return out;
}

const SINKS = /\.(?:innerHTML|outerHTML)\s*\+?=(?!=)|\.insertAdjacentHTML\s*\(/g;

/** Every unescaped interpolation in `src`: [{line, expr}]. */
export function scanSource(src) {
  const found = [];
  SINKS.lastIndex = 0;
  let m;
  while ((m = SINKS.exec(src))) {
    const lineStart = src.lastIndexOf('\n', m.index) + 1;
    if (/^\s*(?:\/\/|\*|\/\*)/.test(src.slice(lineStart, m.index + 1))) continue;
    const line = src.slice(0, m.index).split('\n').length;
    let start = m.index + m[0].length;
    let r;
    if (m[0].includes('insertAdjacentHTML')) {
      start = scanExpr(src, start, ',').end + 1; // skip the position argument
      r = scanExpr(src, start, ')');
    } else {
      r = scanExpr(src, start, ';');
    }
    for (const expr of offenders(r.templates, [])) found.push({ line, expr });
  }
  return found;
}

const HTML_TAG = /(^|[^\w$`])html`/g;

/** Interpolations of an html`` template that sit in an unquoted attribute
 *  value position (`title=${v}`): [{line, expr}].  `esc` leaves whitespace and
 *  `=` alone, so such a value can add attributes of its own; the position
 *  must be quoted (`title="${v}"`). */
export function scanUnquoted(src) {
  const found = [];
  HTML_TAG.lastIndex = 0;
  let m;
  while ((m = HTML_TAG.exec(src))) {
    const tick = m.index + m[0].length - 1;
    const lineStart = src.lastIndexOf('\n', tick) + 1;
    if (/^\s*(?:\/\/|\*|\/\*)/.test(src.slice(lineStart, tick))) continue;
    const line = src.slice(0, tick).split('\n').length;
    for (const it of parseTemplate(src, tick).interps) {
      if (/=\s*$/.test(it.before)) found.push({ line, expr: it.expr });
    }
  }
  return found;
}

/** `file::expr` -> count, for a list of {file, expr}. */
export function keyCounts(items) {
  const counts = {};
  for (const { file, expr } of items) {
    const k = `${file}::${expr}`;
    counts[k] = (counts[k] || 0) + 1;
  }
  return counts;
}

/**
 * Find every `esc(t(...))` / `_esc(t(...))` call in a source text.
 *
 * Returns `{keys, dynamic}`: `keys` are the string-literal keys (with line),
 * including both arms of a ternary; `dynamic` are the sites whose key is built
 * at run time (`t('jobs.' + action)`) and so cannot be resolved here.
 */
export function scanEscT(src) {
  const keys = [];
  const dynamic = [];
  const re = /(?<![\w$.])_?esc\(\s*t\(/g;
  let m;
  while ((m = re.exec(src))) {
    const start = m.index + m[0].length;
    const line = src.slice(0, m.index).split('\n').length;
    // The first argument of t(): up to the `,` or `)` at depth 0.
    let i = start;
    let depth = 0;
    const lits = [];
    let hasOther = false;
    while (i < src.length) {
      const c = src[i];
      if (c === '"' || c === "'") {
        const end = skipString(src, i);
        lits.push(src.slice(i + 1, end - 1));
        i = end;
        continue;
      }
      if (c === '(' || c === '[' || c === '{') depth++;
      else if (c === ')' || c === ']' || c === '}') {
        if (depth === 0) break;
        depth--;
      } else if (c === ',' && depth === 0) break;
      else if (c === '+' && depth === 0) hasOther = true;
      i++;
    }
    if (hasOther || !lits.length) dynamic.push({ line });
    else for (const key of lits) keys.push({ key, line });
  }
  return { keys, dynamic };
}
