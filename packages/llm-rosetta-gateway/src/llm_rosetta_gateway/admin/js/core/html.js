// html.js — text → markup, with no DOM.
//
// The admin escaped through a detached <div>: `textContent` in, `innerHTML`
// out.  That is correct, and it made every module that renders markup
// untestable outside a browser — the lane controls among them, so their
// escaping could only be read, never run.  The escaping is small and fixed;
// it is written out here once and `utils.js` re-exports it, so the browser
// behaviour does not change.
//
// `esc` escapes the five characters that end a text node or a quoted
// attribute — & < > " — and then the apostrophe, because the admin also puts
// these strings inside single-quoted JS in an attribute
// (`onclick="fn('…')"`).  The DOM version escaped & < > only, so a name with a
// double quote left `aria-label="…"` early; the comment there claimed
// otherwise.  That single-quote case is still the wrong escape for the
// position, which is what `jsArg` is for.

/** Escape a value for HTML text or a quoted attribute. */
export function esc(s) {
  if (s === null || s === undefined) return '';
  // `&` first, or the `&` of the entities below is encoded again.
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** A JS string literal, escaped to sit inside a double-quoted HTML attribute.
 *
 *  `esc` is the wrong tool for handler arguments: it turns `'` into `&#39;`,
 *  which the HTML parser decodes back to `'` before the handler is compiled,
 *  so a name like `a'-alert(1)-'b` closes the literal and runs as code.
 *  `jsArg` builds the literal with JSON and escapes the double quotes the
 *  attribute carries, so the parsed attribute is always valid JS. */
export function jsArg(v) {
  const literal = JSON.stringify(v === undefined ? null : v);
  return literal.replace(/&/g, '&amp;').replace(/"/g, '&quot;');
}
