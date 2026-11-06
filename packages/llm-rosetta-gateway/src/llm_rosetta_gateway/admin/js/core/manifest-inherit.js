// manifest-inherit.js — the vendor manifest editor's inheritance view.
//
// A manifest declares connection defaults that every leaf below it inherits,
// field by field; `headers` merges per key.  The editor has to say who
// inherits what before Save, so this module turns the server's per-field map
// into rows and marks the rows an unsaved edit would touch.  It is DOM-free:
// the panel builds the nodes, and the tests call these functions directly.

/** One row per field, flattening `headers` into one row per header name. */
export function inheritanceRows(fields) {
  const source = fields && typeof fields === 'object' ? fields : {};
  const rows = [];
  for (const key of Object.keys(source)) {
    const value = source[key];
    if (key === 'headers' && value && typeof value === 'object') {
      for (const header of Object.keys(value)) {
        rows.push({
          id: `headers.${header}`,
          label: `headers.${header}`,
          field: 'headers',
          header,
          inherited: _names(value[header] && value[header].inherited_by),
          overridden: _names(value[header] && value[header].overridden_by),
        });
      }
      continue;
    }
    rows.push({
      id: key,
      label: key,
      field: key,
      header: null,
      inherited: _names(value && value.inherited_by),
      overridden: _names(value && value.overridden_by),
    });
  }
  return rows;
}

function _names(list) {
  return Array.isArray(list) ? list.slice() : [];
}

/**
 * Mark the rows a proposed edit touches.
 *
 * A change is keyed the same way a row is: a plain field name, or
 * `headers.<name>` for one header.  A leaf that overrides the field is
 * unaffected and stays out of `affected` — that is the whole point of showing
 * the blast radius, since the operator cannot see it from the manifest file.
 */
export function annotateBlast(rows, changes) {
  const byId = new Map();
  for (const change of Array.isArray(changes) ? changes : []) {
    const id = change.field === 'headers'
      ? `headers.${change.header}`
      : change.field;
    byId.set(id, change);
  }
  return rows.map(row => {
    const change = byId.get(row.id);
    return change
      ? { ...row, touched: true, removed: !!change.removed, affected: _names(change.leaves) }
      : { ...row, touched: false, removed: false, affected: [] };
  });
}

/** The leaves any part of the edit changes, deduplicated and ordered. */
export function changedLeaves(changes) {
  const seen = new Set();
  for (const change of Array.isArray(changes) ? changes : []) {
    for (const name of _names(change.leaves)) seen.add(name);
  }
  return [...seen].sort();
}

/** A comma list truncated to `limit` names, with the remainder counted. */
export function nameList(names, limit = 5) {
  const list = Array.isArray(names) ? names : [];
  if (list.length <= limit) return list.join(', ');
  return `${list.slice(0, limit).join(', ')} +${list.length - limit}`;
}
