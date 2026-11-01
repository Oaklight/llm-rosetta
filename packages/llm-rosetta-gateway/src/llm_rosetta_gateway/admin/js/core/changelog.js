/**
 * changelog.js — read one release out of the bundled changelog.
 *
 * The file is the docs changelog verbatim: `## vX.Y.Z — date` headings, an
 * optional `### Added` / `### 新增` under each, and bullet entries whose
 * inline markdown is bold, code and links only.  Parsing and rendering are
 * pure string functions so they can be tested without a DOM; everything
 * rendered goes through `escapeHtml` first, so the output cannot carry markup
 * the file did not have.
 */

/** `v0.13.0+g1a2b` and `0.13.0` name the same release for a changelog. */
export function normaliseVersion(text) {
  return String(text || '')
    .trim()
    .replace(/^v/i, '')
    .split('+')[0];
}

/**
 * Find one release's section.
 *
 * @param {string} markdown - The whole changelog.
 * @param {string} version - The running version, with or without the `v`.
 * @returns {{title: string, date: string, groups: {name: string, items: string[]}[]}|null}
 *   `null` when the file has no section for that version.
 */
export function parseChangelog(markdown, version) {
  const target = normaliseVersion(version);
  if (!target) return null;
  const lines = String(markdown || '').split('\n');
  const groups = [];
  let current = null;
  let found = false;
  let title = '';
  let date = '';
  for (const line of lines) {
    const heading = /^##\s+(.*)$/.exec(line);
    if (heading) {
      if (found) break; // the next release begins; this one ends
      const text = heading[1].trim();
      const parsed = /^v?(\d[^\s—–-]*)\s*(?:[—–-]\s*(.+))?$/.exec(text);
      if (parsed && normaliseVersion(parsed[1]) === target) {
        found = true;
        title = text;
        date = (parsed[2] || '').trim();
      }
      continue;
    }
    if (!found) continue;
    const group = /^###\s+(.*)$/.exec(line);
    if (group) {
      current = {name: group[1].trim(), items: []};
      groups.push(current);
      continue;
    }
    const bullet = /^-\s+(.*)$/.exec(line);
    if (bullet) {
      if (!current) {
        current = {name: '', items: []};
        groups.push(current);
      }
      current.items.push(bullet[1].trim());
      continue;
    }
    // A wrapped bullet continues on the next line; join it so a long entry
    // reads as one paragraph rather than losing its tail.
    if (current && current.items.length && line.trim()) {
      current.items[current.items.length - 1] += ' ' + line.trim();
    }
  }
  if (!found) return null;
  return {title, date, groups};
}

function escapeHtml(text) {
  return String(text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Render one changelog entry's inline markdown as HTML.
 *
 * Only the three forms the file uses: code spans, bold, and links to http(s).
 * The text is escaped first, so an entry that mentions `<code>` shows the
 * word rather than starting a tag.
 */
export function renderEntryHtml(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(
      /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>'
    );
}
