// manifest-inherit.js — the vendor manifest's inheritance view.
//
// A manifest declares connection defaults that every leaf below it inherits,
// field by field; `headers` merges per key.  This module turns the server's
// per-field map into rows saying who inherits what.  It is DOM-free: the panel
// builds the nodes, and the tests call these functions directly.

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

/** A comma list truncated to `limit` names, with the remainder counted. */
export function nameList(names, limit = 5) {
  const list = Array.isArray(names) ? names : [];
  if (list.length <= limit) return list.join(', ');
  return `${list.slice(0, limit).join(', ')} +${list.length - limit}`;
}
