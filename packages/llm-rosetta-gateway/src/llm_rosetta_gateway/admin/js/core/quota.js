// quota.js — "N/unit" request quotas, written the one way both places read them.
//
// The settings rate-limit fields and a key's own limit take the same strings
// ("60/m", "30/m, 500/h"), so they parse and validate here rather than once
// per panel.

export const RL_UNIT_MAP = {
  s: 1, sec: 1, second: 1,
  m: 60, min: 60, minute: 60,
  h: 3600, hr: 3600, hour: 3600,
  d: 86400, day: 86400,
};

/** "100/m" → {limit, period}; null for blank; {error} for anything else. */
export function parseQuota(q) {
  if (!q || !q.trim()) return null;
  const m = q.trim().match(/^(\d+)\s*(?:\/|per)\s*([a-z]+)(?:\s+burst\s+(\d+))?$/i);
  if (!m) return {error: 'Invalid format. Use: N/unit (e.g. 100/m, 60/s, 10000/d)'};
  let unit = m[2].toLowerCase();
  if (unit.length > 1 && unit.endsWith('s')) unit = unit.slice(0, -1);
  if (!(unit in RL_UNIT_MAP)) return {error: 'Unknown unit: ' + m[2]};
  return {limit: parseInt(m[1]), period: RL_UNIT_MAP[unit]};
}

/** The comma list an input holds, trimmed and without empties. */
export function splitQuotas(raw) {
  if (!raw || !raw.trim()) return [];
  return raw.split(',').map(s => s.trim()).filter(Boolean);
}

/** A stored quota (string or list) as the text an input shows. */
export function quotaToDisplay(v) {
  if (!v) return '';
  return Array.isArray(v) ? v.join(', ') : String(v);
}

/** A field's text as the config payload: none, one string, or a list. */
export function quotaToPayload(raw) {
  const parts = splitQuotas(raw);
  if (parts.length === 0) return null;
  return parts.length === 1 ? parts[0] : parts;
}

/**
 * Validate a whole field: every part must parse and allow at least one request.
 * Returns {ok, parts, error} — error names the offending part.
 *
 * The floor is the gateway's own: a window of zero is not "no limit" — it
 * refuses every request — and the limiter refuses to be built with it, so a
 * blank field is the way to say "none".
 */
export function validateQuotas(raw) {
  const parts = splitQuotas(raw);
  for (const part of parts) {
    const p = parseQuota(part);
    if (p && p.error) return {ok: false, parts, error: `"${part}": ${p.error}`};
    if (p && p.limit < 1) {
      return {
        ok: false,
        parts,
        error: `"${part}": allow at least one request, or leave the field blank`,
      };
    }
  }
  return {ok: true, parts, error: null};
}
