/**
 * shim-selection.js — the `shims.include` / `shims.exclude` rule, in one place.
 *
 * A package is installed whole by default; a selection narrows it to the shims
 * the operator ticked.  Both halves have to agree with the gateway, which reads
 * the same section: the panel shows what is kept, and writes what is ticked.
 */

/**
 * Whether a selection holds an `fnmatch` pattern rather than plain names.
 *
 * A pattern is the operator's own edit of the config file; the checkboxes can
 * render the package's declared shims but not expand a pattern, so the card
 * says so instead of pretending the selection is a list of names.
 *
 * @param {string[]} list - One `include` or `exclude` entry.
 * @returns {boolean} True when the entry cannot be shown as checkboxes.
 */
export function hasPattern(list) {
  return (list || []).some((x) => /[*?[]/.test(String(x)));
}

/**
 * Which of a package's declared shims a saved selection keeps.
 *
 * @param {string[]} names - The shims the package declares.
 * @param {{include?: string[], exclude?: string[]}} selection - Its saved entry.
 * @returns {string[]} The shims that load, in declaration order.
 */
export function keptShims(names, selection) {
  const include = Array.isArray(selection && selection.include) ? selection.include : [];
  const exclude = Array.isArray(selection && selection.exclude) ? selection.exclude : [];
  let kept = Array.isArray(names) ? names : [];
  if (include.length) kept = kept.filter((n) => include.includes(n));
  return kept.filter((n) => !exclude.includes(n));
}

/**
 * A package's saved selection, from whichever carrier holds it.
 *
 * A preset-installed package is narrowed by the maps keyed on its preset.  A
 * pinned package (an explicit remote) has no preset key, so the loader reads
 * the entry's own `include` / `exclude` — and the card must read the same
 * place, or a narrowed pinned package paints every box ticked and Save would
 * look like it did nothing.
 *
 * @param {object} section - The saved `shims` config section.
 * @param {{key?: string, source_repo?: string, remote?: object}} pkg
 *   A catalogue entry.
 * @returns {{include?: string[], exclude?: string[]}} The saved selection.
 */
export function selectionOf(section, pkg) {
  const current = section || {};
  const key = pkg && pkg.key;
  const url = (pkg && pkg.remote && pkg.remote.url) || (pkg && pkg.source_repo) || '';
  const remotes = Array.isArray(current.remotes) ? current.remotes : [];
  const entry = remotes.find((r) => r && sameRepo(r.url, url));
  if (entry) return { include: entry.include, exclude: entry.exclude };
  return {
    include: current.include && current.include[key],
    exclude: current.exclude && current.exclude[key],
  };
}

/**
 * The `include` / `exclude` maps that add *key* with *checked* of *total*.
 *
 * Everything ticked clears the package's entry, so the default stays "all of
 * it" — a shim the package adds later then switches itself on.  Anything less
 * is a closed `include` list, which is what the operator picked.  Every other
 * package's entry is carried over untouched, and a stale `exclude` for this
 * package is dropped: the two maps must not both speak for one key.
 *
 * @param {object} section - The saved `shims` config section.
 * @param {string} key - The catalogue key being saved.
 * @param {string[]} checked - The ticked shim names.
 * @param {number} total - How many the card offered.
 * @returns {{include: object, exclude: object}} The maps to write.
 */
export function selectionMaps(section, key, checked, total) {
  const include = { ...((section && section.include) || {}) };
  const exclude = { ...((section && section.exclude) || {}) };
  if (checked.length === total) {
    delete include[key];
  } else {
    include[key] = [...checked];
  }
  delete exclude[key];
  return { include, exclude };
}

/**
 * A commit shortened the way git prints it; a tag stays as declared.
 *
 * A remote source is pinned to a commit SHA, but a preset may pin a tag, and
 * then the pin is that tag.  Truncating a tag would hide which release the
 * gateway runs, so only an all-hex string is cut.
 *
 * @param {string|null} ref - The pinned ref, or a repository head.
 * @returns {string} Seven characters for a SHA, the ref itself otherwise.
 */
export function shortRef(ref) {
  if (!ref) return '';
  return /^[0-9a-f]{7,40}$/i.test(ref) ? ref.slice(0, 7) : ref;
}

/**
 * The mark for one vendor of a catalogue package.
 *
 * A catalogue record keys its marks by the vendor the manifest names, while a
 * shim name carries the lane's own prefix — `alcf_metis--…` belongs to the
 * `alcf` mark — so the leading segment is tried before giving up.  A package
 * with exactly one mark paints it on every lane, which is what a manifest
 * above the lanes means.
 *
 * @param {{logos?: object}} pkg - A catalogue entry.
 * @param {string} vendor - The vendor name taken from a shim name.
 * @param {string|null} [known] - The mark the inventory holds, if any.
 * @returns {string|null} A URL, or null when the vendor has no mark.
 */
export function packageLogoUrl(pkg, vendor, known = null) {
  const logos = (pkg && pkg.logos) || {};
  if (logos[vendor]) return logos[vendor];
  const head = String(vendor).split('_')[0];
  if (logos[head]) return logos[head];
  if (known) return known;
  const keys = Object.keys(logos);
  return keys.length === 1 ? logos[keys[0]] : null;
}

/**
 * Whether a catalogue package answers a search box's query.
 *
 * The Installed view lists what is registered, so a shim that is available
 * but not installed is invisible there — searching for it would answer "no
 * shims" for something one click away.  The same predicate decides both: the
 * Available view's filter, and the "not installed" section the Installed view
 * adds under a query.  Every whitespace-separated term must match, so
 * "alcf metis" narrows instead of widening.
 *
 * @param {{key?: string, label?: string, shims?: string[], formats?: string[]}} pkg
 *   A catalogue entry.
 * @param {string} query - What the operator typed.
 * @returns {boolean} True when the package matches.
 */
export function catalogueMatches(pkg, query) {
  const q = String(query || '').trim().toLowerCase();
  if (!q) return true;
  const hay = [pkg && pkg.key, pkg && pkg.label,
    ...((pkg && pkg.shims) || []), ...((pkg && pkg.formats) || [])]
    .filter(Boolean).join(' ').toLowerCase();
  return q.split(/\s+/).every((term) => hay.includes(term));
}

/**
 * The `shims.remotes` entry that loads a package at one chosen ref.
 *
 * A preset pins what the catalogue ships, and names the commit-addressed
 * mirror that release published.  Asking for another ref is a different
 * request: the loader fills the mirror's `{ref}` with the declared ref, and
 * only the catalogue's own tag has a `dist/<ref>/` directory — a branch, a
 * commit or another tag has none, so the mirror would 404 and the loader
 * would fall back to codeload.  The entry therefore drops the mirror and
 * names the ref directly, rather than carrying a leg that is certain to fail.
 * An empty ref falls back to the repository head, which is what an omitted
 * `ref` already means to the loader.
 *
 * @param {{key?: string, source_repo?: string, remote?: object}} pkg
 *   A catalogue entry.
 * @param {string} ref - A tag, branch or commit.
 * @returns {object} The remote entry to append to `shims.remotes`.
 */
export function pinnedRemote(pkg, ref) {
  const remote = { ...((pkg && pkg.remote) || {}) };
  if (!remote.url && pkg && pkg.source_repo) remote.url = pkg.source_repo;
  // Per-release bytes: a digest for another ref would fail the fetch.
  delete remote.mirror;
  const value = String(ref || '').trim();
  if (value) remote.ref = value;
  else delete remote.ref;
  return remote;
}

/**
 * The load report's entry for *url*, when that source failed.
 *
 * A ref that does not exist fails the fetch, not the write: the config already
 * holds it, and the next start would drop the package without a word.  The
 * report names which source failed, so a caller can undo exactly that write —
 * and must key on the URL, not on the failure count, or another package's
 * bad source would undo a ref that loaded fine.
 *
 * @param {Array<{location?: string, ok?: boolean, error?: string}>} sources
 *   The `status.sources` list a reload reported.
 * @param {string|null|undefined} url - The source that was just written.
 * @returns {{location?: string, error?: string}|null} The failing entry.
 */
export function failedSource(sources, url) {
  if (!url || !Array.isArray(sources)) return null;
  return sources.find((s) => s && s.ok === false && s.location === url) || null;
}

/**
 * Whether two source URLs name one repository.
 *
 * The loader keys sources on its own normal form — lowercase, no trailing
 * slash, no `.git` — so two spellings of one repository are one source to the
 * gateway.  The panel has to use the same rule, or a `.git` entry would be
 * appended beside the plain one instead of replacing it.
 *
 * @param {string|null|undefined} a - One URL.
 * @param {string|null|undefined} b - The other.
 * @returns {boolean} True when both name the same repository.
 */
export function sameRepo(a, b) {
  const norm = (value) => String(value || '').trim().toLowerCase()
    .replace(/\/+$/, '').replace(/\.git$/, '');
  if (!a || !b) return false;
  return norm(a) === norm(b);
}

/**
 * The catalogue packages a query names that the gateway does not run.
 *
 * The Installed view lists what is registered, so an uninstalled package read
 * as missing.  Extracted from the page so the rule can be tested: a query is
 * required (without one this would be the whole Available view), the tier
 * filter applies, and so does the origin filter — every catalogue package is
 * an external source, so a *builtin* or *local* filter matches none of them.
 *
 * @param {Array<object>} packages - The catalogue entries.
 * @param {{q?: string, tier?: string, origin?: string}} filters - The toolbar.
 * @param {(pkg: object, q: string) => boolean} [matches] - The search rule.
 * @returns {Array<object>} The matching, not-installed packages.
 */
export function availableNotInstalled(packages, filters, matches = catalogueMatches) {
  const f = filters || {};
  const q = String(f.q || '').trim();
  if (!q) return [];
  if (f.origin && f.origin !== 'remote') return [];
  return (packages || []).filter((pkg) => {
    if (pkg.installed) return false;
    if (f.tier && (pkg.tier || (pkg.free ? 'free' : 'paid')) !== f.tier) return false;
    return matches(pkg, q);
  });
}

/**
 * The `shims` section a ref install writes, from the section it has now.
 *
 * The preset leaves `presets` (it pins the catalogue's own release) and an
 * explicit remote takes the package's place: same URL and path, the chosen
 * ref, no mirror.  An existing remote for the same repository is replaced
 * rather than duplicated, under the loader's own URL normalisation.
 *
 * @param {object} section - The saved `shims` section.
 * @param {object} pkg - The catalogue entry being loaded.
 * @param {string} ref - The ref the operator typed.
 * @returns {{presets: string[], remotes: object[]}} The section to write.
 */
export function refSourcesWrite(section, pkg, ref) {
  const current = section || {};
  const key = pkg && pkg.key;
  const url = (pkg && pkg.remote && pkg.remote.url) || (pkg && pkg.source_repo) || '';
  const presets = Array.isArray(current.presets) ? [...current.presets] : [];
  const remotes = Array.isArray(current.remotes) ? [...current.remotes] : [];
  const include = { ...(current.include || {}) };
  const exclude = { ...(current.exclude || {}) };
  // A package can be pinned twice: the first pin moves the selection off the
  // keyed map and onto the entry, and the second pin replaces that entry. So
  // the selection is read from whichever carrier holds it — reading only the
  // keyed map would rebuild the entry empty and widen the package to every
  // shim it ships.
  const saved = selectionOf(current, pkg);
  const entry = pinnedRemote(pkg, ref);
  if (Array.isArray(saved.include) && saved.include.length) {
    entry.include = [...saved.include];
  }
  if (Array.isArray(saved.exclude) && saved.exclude.length) {
    entry.exclude = [...saved.exclude];
  }
  delete include[key];
  delete exclude[key];
  return {
    presets: presets.filter((k) => k !== key),
    remotes: [...remotes.filter((r) => !(r && sameRepo(r.url, url))), entry],
    include,
    exclude,
  };
}

/**
 * The `shims` section that removes a package, whichever shape installed it.
 *
 * A package can be present as a preset key *or* as an explicit remote (a ref
 * install), and the page's Remove only rewrote `presets` — so a pinned
 * package stayed loaded while the toast said it was removed. Both carriers go,
 * with the selection maps that name the package, and the entry the panel is
 * about to write keeps every other package untouched.
 *
 * @param {object} section - The saved `shims` section.
 * @param {object} pkg - The catalogue entry being removed.
 * @returns {{presets: string[], remotes: object[], include: object, exclude: object}}
 *   The section to write.
 */
export function removeSourcesWrite(section, pkg) {
  const current = section || {};
  const key = pkg && pkg.key;
  const url = (pkg && pkg.remote && pkg.remote.url) || (pkg && pkg.source_repo) || '';
  const include = { ...(current.include || {}) };
  const exclude = { ...(current.exclude || {}) };
  delete include[key];
  delete exclude[key];
  return {
    presets: (Array.isArray(current.presets) ? current.presets : [])
      .filter((k) => k !== key),
    remotes: (Array.isArray(current.remotes) ? current.remotes : [])
      .filter((r) => !(r && sameRepo(r.url, url))),
    include,
    exclude,
  };
}

/**
 * Run one ref install, with its rollback, against an injected config.
 *
 * The write lands before the fetch, so a bad ref must be undone — and that
 * ordering, not the helpers it calls, is the part a page mutation could
 * silently delete. It lives here with `io` injected so a test can watch the
 * calls it makes:
 *
 * @param {object} opts
 * @param {object} opts.section - The saved `shims` section (the snapshot).
 * @param {object} opts.pkg - The catalogue entry.
 * @param {string} opts.ref - The ref the operator typed.
 * @param {{writeShims: Function, reload: Function, readConfig: Function}} opts.io
 *   The config calls the page would make.
 * @returns {Promise<{ok: boolean, error?: string, next: object}>} The outcome;
 *   `next` is the section a successful install left on disk.
 */
export async function executeRefInstall({ section, pkg, ref, io }) {
  // The snapshot is every key the write can touch: the ref write clears the
  // keyed maps as it moves the selection onto the entry, so restoring only
  // `presets` and `remotes` would bring a narrowed package back at full
  // width — the exact loss the selection move exists to prevent.
  const before = {
    presets: Array.isArray(section && section.presets) ? [...section.presets] : [],
    remotes: Array.isArray(section && section.remotes) ? [...section.remotes] : [],
    include: { ...((section && section.include) || {}) },
    exclude: { ...((section && section.exclude) || {}) },
  };
  const next = refSourcesWrite(section, pkg, ref);
  await io.writeShims(next);
  const report = await io.reload();
  const action = refInstallAction(report, pkg && pkg.remote && pkg.remote.url);
  if (action.restore) {
    await io.writeShims(before);
    await io.reload();
    await io.readConfig();
    return { ok: false, error: action.error || null, next: before };
  }
  await io.readConfig();
  return { ok: true, next };
}

/**
 * What a ref install should do once the reload has answered.
 *
 * The write lands before the fetch, so a ref the repository does not have
 * leaves the bad ref on disk.  This is the decision that undoes it, kept here
 * so it is testable: roll back only for *this* source, and carry the forge's
 * own message rather than a generic one.
 *
 * @param {object|null|undefined} report - The reload's response body.
 * @param {string|null|undefined} url - The source the install just wrote.
 * @returns {{restore: boolean, error: string|null}} The action to take.
 */
export function refInstallAction(report, url) {
  const broken = failedSource(report && report.status && report.status.sources, url);
  if (!broken) return { restore: false, error: null };
  return { restore: true, error: broken.error || null };
}

/**
 * The `shims` body that saves a package's selection, in the shape it has now.
 *
 * A preset-installed package keeps its selection in `include` / `exclude`,
 * keyed by preset.  A pinned package (an explicit remote) has no preset key
 * for the map to bite on, so the selection belongs on the entry itself — the
 * same place `refSourcesWrite` puts it.  Without this, **Save selection** on a
 * pinned package writes a map nothing reads and reports success.
 *
 * @param {object} section - The saved `shims` section.
 * @param {object} pkg - The catalogue entry.
 * @param {{include: object, exclude: object}} maps - The maps the panel built.
 * @returns {object} The body to write (only the keys that changed).
 */
export function selectionWrite(section, pkg, maps) {
  const current = section || {};
  const key = pkg && pkg.key;
  const url = (pkg && pkg.remote && pkg.remote.url) || (pkg && pkg.source_repo) || '';
  const remotes = Array.isArray(current.remotes) ? [...current.remotes] : [];
  const index = remotes.findIndex((r) => r && sameRepo(r.url, url));
  const body = { include: maps.include, exclude: maps.exclude };
  if (index === -1) return body;
  // Pinned: the entry carries the selection, and the map entry would be dead.
  const entry = { ...remotes[index] };
  const wanted = maps.include[key];
  const dropped = maps.exclude[key];
  if (Array.isArray(wanted) && wanted.length) entry.include = [...wanted];
  else delete entry.include;
  if (Array.isArray(dropped) && dropped.length) entry.exclude = [...dropped];
  else delete entry.exclude;
  remotes[index] = entry;
  const include = { ...body.include };
  const exclude = { ...body.exclude };
  delete include[key];
  delete exclude[key];
  return { include, exclude, remotes };
}
