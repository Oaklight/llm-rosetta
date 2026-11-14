/**
 * Code blocks: syntax colour and a copy control in the frame's corner.
 *
 * The panel shows code in three places (the API-keys samples, a routing
 * rule's JSON, an error dump's body) and had no highlighter: every block was
 * one colour, so a shell line and its own flags read as the same string.
 * There is no build step and no third-party request allowed (#1522), so the
 * lexer is here: a per-language rule list, scanned left to right, with each
 * token escaped on its way out.
 *
 * The rules are deliberately shallow — a sample block is short and its
 * language is known — but they are ordered: a string is recognised before a
 * comment, so `#` inside a quoted URL is not a comment.
 */

/** The languages a block may declare. */
export const CODE_LANGS = ['json', 'shell', 'python', 'js'];

/** @type {Record<string, Array<[RegExp, string]>>} */
const RULES = {
  json: [
    [/"(?:\\.|[^"\\])*"(?=\s*:)/y, 'key'],
    [/"(?:\\.|[^"\\])*"/y, 'str'],
    [/\b(?:true|false|null)\b/y, 'lit'],
    [/-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/y, 'num'],
  ],
  shell: [
    [/'[^']*'|"(?:\\.|[^"\\])*"/y, 'str'],
    [/#[^\n]*/y, 'com'],
    [/\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*/y, 'var'],
    [/--?[A-Za-z][\w-]*/y, 'flag'],
    [/\b(?:export|cd|curl|setx|set|python|python3|node|npm|pip|uv|llm-rosetta-gateway)\b/y, 'kw'],
  ],
  python: [
    [/(?:'''[\s\S]*?'''|"""[\s\S]*?""")/y, 'str'],
    [/'(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*"/y, 'str'],
    [/#[^\n]*/y, 'com'],
    [/@[A-Za-z_][\w.]*/y, 'dec'],
    [/\b(?:import|from|def|return|if|else|elif|for|in|while|with|as|try|except|finally|class|raise|None|True|False)\b/y, 'kw'],
    [/\b\d+(?:\.\d+)?\b/y, 'num'],
  ],
  js: [
    [/\/\*[\s\S]*?\*\//y, 'com'],
    [/\/\/[^\n]*/y, 'com'],
    [/'(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*"|`(?:\\.|[^`\\])*`/y, 'str'],
    [/\b(?:const|let|var|function|return|await|async|if|else|new|import|from|export|class|throw|try|catch|of)\b/y, 'kw'],
    [/\b\d+(?:\.\d+)?\b/y, 'num'],
  ],
};

/**
 * Escape text for HTML.
 *
 * @param {string} text
 * @returns {string}
 */
export function escapeCode(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Colour one block.
 *
 * Scans left to right: at each position the earliest-starting rule wins, and
 * a position no rule matches is copied through as one character.  The output
 * is HTML with `tok-*` spans, every piece escaped.
 *
 * @param {string} code
 * @param {string} lang One of {@link CODE_LANGS}; anything else is plain.
 * @returns {string}
 */
export function highlight(code, lang) {
  const text = String(code ?? '');
  const rules = RULES[lang];
  if (!rules) return escapeCode(text);
  let out = '';
  let i = 0;
  while (i < text.length) {
    let best = null;
    for (const [pattern, cls] of rules) {
      pattern.lastIndex = i;
      const m = pattern.exec(text);
      if (m && m.index === i && m[0].length > 0) {
        // The first rule in the list wins at the same position, which is what
        // makes "string before comment" hold.
        best = { text: m[0], cls };
        break;
      }
    }
    if (best) {
      out += `<span class="tok-${best.cls}">${escapeCode(best.text)}</span>`;
      i += best.text.length;
    } else {
      out += escapeCode(text[i]);
      i += 1;
    }
  }
  return out;
}

/** The copy glyph: two sheets, the front one offset. */
const COPY_ICON =
  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ' +
  'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';

/**
 * A framed block with the copy control in its top-right corner.
 *
 * @param {string} code The text to show; it is highlighted, never executed.
 * @param {string} lang One of {@link CODE_LANGS}.
 * @param {{label?: string, className?: string}} [opts]
 * @returns {string}
 */
export function codeBlockHtml(code, lang, opts = {}) {
  const label = opts.label ? escapeCode(opts.label) : '';
  const extra = opts.className ? ' ' + opts.className : '';
  return (
    `<div class="code-block-wrap">` +
    `<button type="button" class="icon-btn code-copy" onclick="copyCodeBlock(this)"` +
    `${label ? ` title="${label}" aria-label="${label}"` : ' title="Copy" aria-label="Copy"'}>` +
    `${COPY_ICON}</button>` +
    `<pre class="code-block${extra}" data-lang="${escapeCode(lang)}">` +
    `<code>${highlight(code, lang)}</code></pre>` +
    `</div>`
  );
}

/**
 * Copy the block that owns the button.
 *
 * The button is icon-only, so the feedback is the shared toast rather than a
 * label swap that would move the corner.
 *
 * @param {HTMLElement} btn
 */
export function copyCodeBlock(btn) {
  const wrap = btn.closest('.code-block-wrap');
  const code = wrap && wrap.querySelector('code');
  if (!code) return;
  window.copyText?.(code.textContent);
}

// Guarded: the lexer is imported by `node --test`, which has no window.
if (typeof window !== 'undefined') Object.assign(window, { copyCodeBlock });
