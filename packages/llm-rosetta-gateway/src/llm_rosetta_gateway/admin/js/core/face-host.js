// face-host.js — one endpoint's face: reading it and writing it.
//
// A provider entry is an account; each typed endpoint is a face of it and may
// stand on another vendor's host (#1595).  The rule that decides an endpoint's
// host used to live inline in the drawer, where six mutations of it left the
// whole JS suite green — including the line that writes the face's `base_url`
// back.  It is here so those cases can run it instead of reading it.

/** One endpoint setting, the way the runtime reads it.
 *
 *  The block (`embeddings: {shim, format, path, enabled, base_url}`) wins; an
 *  empty value in it falls through to the flat `<type>_*` key a config written
 *  before the block still carries.  `enabled` decides by presence, so `false`
 *  is a value there — the same rule as `config/typed.py::face_value`.
 */
export function readFace(cfg, typeName, field, flatKey) {
  if (!cfg) return undefined;
  const block = cfg[typeName];
  if (block && typeof block === 'object' && !Array.isArray(block)) {
    const value = block[field];
    if (value !== undefined && (field === 'enabled' || value)) return value;
  }
  if (field === 'enabled') return cfg[`${typeName}_enabled`];
  return flatKey ? cfg[flatKey] : undefined;
}

/** The block the drawer writes for one endpoint.
 *
 *  `base_url` is only written when the operator typed one: blank means "inherit
 *  the provider's", and storing `""` would look like a deliberate host.  An
 *  endpoint the operator switched off carries `enabled: false`; an enabled one
 *  carries no `enabled` key at all, which is the default.
 */
export function faceBlock({ shim = '', format = '', path = '', baseUrl = '', enabled = true } = {}) {
  const block = {};
  if (shim) block.shim = shim;
  block.format = format || '';
  block.path = path || '';
  const host = String(baseUrl || '').trim();
  if (host) block.base_url = host;
  if (!enabled) block.enabled = false;
  return block;
}

/** The face state one endpoint row shows.
 *
 *  The drawer wired these facts inline, and four mutations of that wiring — the
 *  field never reading the stored host, the marker never firing, the save
 *  posting an empty host — left the whole JS suite green (#1595 review).  One
 *  builder feeds the row and the save from the payload, so a case can drive it.
 *
 *  `entry` is the provider entry as the payload carries it, `desc` the model
 *  type's descriptor, `shims` the registered shims.  A face naming only a
 *  format is compared against every shim of that format — the same set the
 *  runtime's guardrail uses, so the two agree about "another vendor".
 */
export function faceRow({ entry, desc, shims, inherited } = {}) {
  const cfg = entry || {};
  const name = desc?.name || '';
  const shim = readFace(cfg, name, 'shim', desc?.config_shim_key || '') || '';
  const format = readFace(cfg, name, 'format', desc?.config_format_key || '') || '';
  const stored = readFace(cfg, name, 'base_url', `${name}_base_url`) || '';
  const base = String(inherited ?? (cfg.base_url || ''));
  const candidates = faceCandidateHosts(shims, {
    modelType: name,
    format: format || (shims || []).find(s => s.name === shim)?.format,
    shim,
  });
  return {
    host: String(stored),
    inherited: base,
    candidates,
    marked: !String(stored).trim() && faceHostMismatch(base, candidates),
    shim,
    format,
  };
}

/** The block one endpoint row saves.
 *
 *  A blank host is an inheritance, so it is not written: storing `""` would
 *  look like a deliberate host.  This exists as its own export because
 *  `faceBlock({..., baseUrl: field.value})` inline is exactly the line that a
 *  mutation can delete with the suite still green.
 */
export function faceSaveBlock({ host = '', shim = '', format = '', path = '', enabled = true } = {}) {
  return faceBlock({ shim, format, path, baseUrl: host, enabled });
}

/** The hosts an endpoint's declaration would forward to.
 *
 *  A face naming a shim takes that shim's host.  A face naming only a *format*
 *  is served by every registered shim of that `(model_type, format)` pair —
 *  `embeddings: {format: "openai"}` by OpenAI's, SiliconFlow's, Zhipu's and
 *  three more — so all of them answer.  Same rule as the runtime's guardrail
 *  (`config/typed.py::_endpoint_hosts`): accepting only the first made the
 *  panel and the gateway disagree, and made four vendors that route correctly
 *  look like a misroute (#1595 review).
 */
export function faceCandidateHosts(shims, { modelType, format, shim } = {}) {
  const list = shims || [];
  if (shim) {
    const def = list.find(s => s.name === shim);
    return def && def.default_base_url ? [def.default_base_url] : [];
  }
  const hosts = [];
  for (const s of list) {
    if (s.model_type !== modelType || s.format !== format) continue;
    if (s.default_base_url && !hosts.includes(s.default_base_url)) hosts.push(s.default_base_url);
  }
  return hosts;
}

/** Whether an endpoint inherits a host its declaration says belongs elsewhere.
 *
 *  The comparison is `origin`, not hostname: a face on `http://` under an
 *  entry on `https://`, or on another port of the same host, is a different
 *  place, and comparing hostnames alone called both a match (#1595 review).
 *  A value that is not an absolute URL compares as itself, so a typo still
 *  differs from a real host rather than throwing.  *hosts* is one host or the
 *  list `faceCandidateHosts` returns; an empty list means nothing is known
 *  about the endpoint's destination, and nothing is reported.
 */
export function faceHostMismatch(inherited, hosts) {
  const a = _origin(inherited);
  const candidates = (Array.isArray(hosts) ? hosts : [hosts]).map(_origin).filter(Boolean);
  if (!a || !candidates.length) return false;
  return !candidates.includes(a);
}

function _origin(url) {
  const text = String(url || '').trim();
  if (!text) return '';
  try {
    return new URL(text).origin;
  } catch {
    return text.replace(/\/+$/, '').toLowerCase();
  }
}
