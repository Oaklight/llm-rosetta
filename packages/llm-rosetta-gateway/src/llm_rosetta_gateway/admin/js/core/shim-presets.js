/**
 * The preload-preset selection, kept free of the DOM so it can be tested.
 *
 * The Shims settings pane shows one chip per standalone package; clicking a
 * chip toggles its catalog key in `shims.presets`.  The declaration is still a
 * comma-separated list on the wire, so these helpers are what bridges the two.
 */

/** Split a declared value into preset keys, dropping blanks and duplicates. */
export function parsePresets(value) {
  const seen = new Set();
  const out = [];
  for (const raw of String(value || '').split(/[\s,]+/)) {
    const key = raw.trim();
    if (key && !seen.has(key)) {
      seen.add(key);
      out.push(key);
    }
  }
  return out;
}

/**
 * Return *current* with *key* selected or not, keeping declaration order.
 *
 * Selecting appends (the order is what the operator picked), deselecting
 * removes every occurrence.
 */
export function togglePreset(current, key, on) {
  const rest = parsePresets(current).filter(k => k !== key);
  if (on) rest.push(key);
  return rest;
}

/** Render the selected keys back to the comma-separated value Save reads. */
export function formatPresets(keys) {
  return parsePresets((keys || []).join(', ')).join(', ');
}

/**
 * The chips to draw: every known package, flagged with whether it is chosen.
 *
 * A key that is declared but missing from the catalog gets no row, so the
 * value can only keep it: it stays in the declaration and the trigger still
 * names it.  The key may belong to a catalog older or newer than this build,
 * and silently deleting it would rewrite a config the operator never touched.
 */
export function presetChips(presets, selected) {
  const chosen = new Set(parsePresets(selected));
  return (presets || []).map(p => ({
    key: p.key,
    label: p.label || p.key,
    count: Array.isArray(p.shims) ? p.shims.length : 0,
    repo: p.source_repo || '',
    free: p.free === true,
    selected: chosen.has(p.key),
  }));
}

/**
 * Split the chips into the two groups the dropdown shows.
 *
 * A free resource is a different decision from a paid one — selecting it may
 * put prompts on a third party and costs nothing — so the two are never mixed
 * in one list.  The free group comes first; an empty group is omitted.
 */
export function presetGroups(presets, selected) {
  const chips = presetChips(presets, selected);
  const groups = [
    { key: 'free', items: chips.filter(c => c.free) },
    { key: 'paid', items: chips.filter(c => !c.free) },
  ];
  return groups.filter(g => g.items.length);
}
