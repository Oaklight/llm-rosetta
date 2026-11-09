// lanes.js — the identity of a routing lane: (provider, upstream_model).
//
// A model may route through the same provider twice, once per upstream model
// (#1430).  The provider name alone no longer addresses a row on the Models
// page: the config endpoints answer 409 when it names more than one lane, and
// a control that sends only the name would toggle or remove the wrong one.
//
// An absent upstream is a value, not a wildcard — the lane whose provider
// serves the model's own name.  So `null`, `undefined` and `''` are one key.

/** The lane as a comparison key.  Never shown: `laneLabel` is. */
export function laneKey(name, upstream) {
  return `${name || ''}\u0000${upstream || ''}`;
}

/** The lane of one provider detail, as `{name, upstream}`. */
export function laneOf(detail) {
  return {
    name: (detail && detail.name) || '',
    upstream: (detail && detail.upstream_model) || '',
  };
}

/** The detail row for `(name, upstream)`, or null when the model has no such lane. */
export function findLane(details, name, upstream) {
  const key = laneKey(name, upstream);
  return (details || []).find((d) => laneKey(d.name, d.upstream_model) === key) || null;
}

/** How a lane is named in a confirmation or a toast. */
export function laneLabel(name, upstream) {
  return upstream ? `${name} → ${upstream}` : name;
}

/** A request body that addresses one lane.
 *
 *  `upstream_model` is always sent, as `''` for the lane that declares none.
 *  An absent field means "not specified", which the endpoints answer with a
 *  409 when the provider appears more than once — so leaving the empty value
 *  out would make the undeclared lane impossible to address from the panel. */
export function laneBody(name, upstream, fields) {
  return { provider: name, upstream_model: upstream || '', ...(fields || {}) };
}
