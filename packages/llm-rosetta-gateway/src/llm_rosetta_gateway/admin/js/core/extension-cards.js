// extension-cards.js — what the extension card says about an external program.
//
// An extension that drives an agent CLI (`claude`, `agy`) declares it; the
// listing (`GET /admin/api/extensions`) checks it and says one of three things
// that matter — missing, older than the extension's minimum, ready — plus
// `broken` when the program starts and fails (its Node.js no longer runs) and
// `unknown` when the check ran but reported no version.  This module turns the
// listing into the cards the panel paints and decides which need attention.
// It is DOM-free: the panel builds the nodes, and the tests call these directly.

/** The card statuses, in the order a card with several programs reports them. */
export const STATUSES = ['missing', 'broken', 'too_old', 'unknown', 'ready'];

/** The status a program shows; an unknown spelling is `unknown`, never `ready`. */
export function programStatus(program) {
  const raw = program && program.status;
  return STATUSES.includes(raw) ? raw : 'unknown';
}

/** One program row of a card. */
export function programRow(program) {
  const p = program && typeof program === 'object' ? program : {};
  return {
    name: String(p.name || ''),
    status: programStatus(p),
    found: p.found || null,
    minimum: p.minimum || null,
    // The pinned version `ext install-cli` installs, when the extension says.
    pinned: p.install && p.install.version ? String(p.install.version) : null,
    // The command is the server's: it is only there when running it helps.
    command: typeof p.install_command === 'string' && p.install_command ? p.install_command : null,
    hint: p.install_hint || null,
    detail: p.detail || null,
  };
}

/** The worst status of a card's programs: the one the card is badged with. */
export function cardStatus(programs) {
  for (const status of STATUSES) {
    if (programs.some((p) => p.status === status)) return status;
  }
  return 'ready';
}

/**
 * The cards of the listing: one per extension that declares an external
 * program.  Cards that need attention come first, then by id.
 */
export function extensionCards(listing) {
  const rows = Array.isArray(listing) ? listing : [];
  const cards = rows
    .filter((e) => e && Array.isArray(e.external) && e.external.length)
    .map((e) => {
      const programs = e.external.map(programRow);
      return {
        id: String(e.id),
        version: e.version || null,
        state: e.state || null,
        reason: e.reason || null,
        programs,
        status: cardStatus(programs),
      };
    });
  const rank = (c) => (c.status === 'ready' ? 1 : 0);
  return cards.sort((a, b) => rank(a) - rank(b) || a.id.localeCompare(b.id));
}
