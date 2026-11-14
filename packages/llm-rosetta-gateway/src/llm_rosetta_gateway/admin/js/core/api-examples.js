// The snippets one wire example turns into.
//
// Kept out of the tab module so a node test can run them: the panel is only
// as honest as the text it puts on the clipboard, and a snippet that drifts
// from the catalog's path or the gateway's auth header sends the operator to
// a 401.

/** The env var a snippet reads when no key is selected. */
export const DEFAULT_ENV = 'LLM_ROSETTA_API_KEY';

/**
 * The env var a selected key is read from.
 *
 * The gateway keeps only a hash, so the panel cannot print the secret.  It
 * names a variable after the key's label instead — the operator exports the
 * value wherever it was saved at creation, and the snippet says which key
 * that value belongs to.
 *
 * @param {{label?: string, id?: string}|null} key
 * @returns {string} An upper-snake env var name.
 */
export function envNameFor(key) {
  if (!key) return DEFAULT_ENV;
  const stem = String(key.label || key.id || '')
    .toUpperCase()
    .replace(/[^A-Z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return stem ? `LLM_ROSETTA_KEY_${stem}` : DEFAULT_ENV;
}

const LANGS = ['curl', 'python', 'node'];

function _json(value) {
  return JSON.stringify(value, null, 2);
}

/** Single-quote a string for a POSIX shell. */
function _shellQuote(text) {
  return `'${String(text).replace(/'/g, `'\\''`)}'`;
}

/** How a language spells "the secret in this env var". */
function _secretRef(lang, env) {
  if (lang === 'python') return `os.environ["${env}"]`;
  if (lang === 'node') return `process.env.${env}`;
  return `$${env}`;
}

/** The auth header value, keeping the catalog's scheme in front of the key. */
function _authValue(lang, auth, env) {
  const template = (auth && auth.value) || '{key}';
  const ref = _secretRef(lang, env);
  if (!template.includes('{key}')) return template;
  const [before, after] = template.split('{key}');
  if (lang === 'python') {
    return (before || after) ? `"${before}" + ${ref}` + (after ? ` + "${after}"` : '') : ref;
  }
  if (lang === 'node') {
    return (before || after) ? '`' + before + '${' + ref + '}' + after + '`' : ref;
  }
  return before + ref + after;
}

/**
 * Build the three snippets for one example.
 *
 * @param {{path: string, auth?: {name: string, value: string},
 *          content_type?: string, body: unknown}} example
 * @param {{base: string, env: string}} opts  Gateway origin and env var name.
 * @returns {{curl: string, python: string, node: string}}
 */
export function buildSnippets(example, opts) {
  const base = (opts.base || '').replace(/\/+$/, '');
  const env = opts.env || DEFAULT_ENV;
  const url = `${base}${example.path}`;
  const auth = example.auth || { name: 'Authorization', value: 'Bearer {key}' };
  const ctype = example.content_type || 'application/json';
  const body = _json(example.body);
  const out = {};

  out.curl = [
    `curl -X POST ${url} \\`,
    `  -H "content-type: ${ctype}" \\`,
    `  -H "${auth.name}: ${_authValue('curl', auth, env)}" \\`,
    `  -d ${_shellQuote(body)}`,
  ].join('\n');

  // The body rides in a triple-quoted string so the snippet shows the exact
  // JSON: a Python literal would need true/false/null translated, and a
  // translated body is one more thing that can drift.
  const triple = `'''${body.replace(/'''/g, "\\'\\'\\'")}'''`;
  out.python = [
    'import json',
    'import os',
    '',
    'import requests',
    '',
    `payload = json.loads(${triple})`,
    '',
    'resp = requests.post(',
    `    "${url}",`,
    '    headers={',
    `        "content-type": "${ctype}",`,
    `        "${auth.name}": ${_authValue('python', auth, env)},`,
    '    },',
    '    json=payload,',
    '    timeout=60,',
    ')',
    'resp.raise_for_status()',
    'print(json.dumps(resp.json(), indent=2))',
  ].join('\n');

  out.node = [
    `const payload = ${body};`,
    '',
    `const resp = await fetch("${url}", {`,
    '  method: "POST",',
    '  headers: {',
    `    "content-type": "${ctype}",`,
    `    "${auth.name}": ${_authValue('node', auth, env)},`,
    '  },',
    '  body: JSON.stringify(payload),',
    '});',
    'console.log(JSON.stringify(await resp.json(), null, 2));',
  ].join('\n');

  return out;
}

/** The language keys a card offers, in display order. */
export function snippetLangs() {
  return LANGS.slice();
}

/**
 * The two export lines that point a tool at this gateway.
 *
 * A tool is redirected by the base URL and the key in its environment, not by
 * an HTTP request, so this is the sample most operators copy.  The key is a
 * placeholder: the gateway stores a hash and cannot print the value, and a
 * sample that printed a fake one would be pasted as-is.
 *
 * @param {{base: string, basePath?: string, baseEnv: string, keyEnv: string,
 *          platform?: 'unix'|'windows', keyPlaceholder?: string}} opts
 * @returns {string}
 */
export function buildShell(opts) {
  const base = (opts.base || '').replace(/\/+$/, '');
  const url = `${base}${opts.basePath || ''}`;
  const key = opts.keyPlaceholder || '<your gateway key>';
  if (opts.platform === 'windows') {
    return [
      `$env:${opts.baseEnv} = "${url}"`,
      `$env:${opts.keyEnv} = "${key}"`,
    ].join('\n');
  }
  return [
    `export ${opts.baseEnv}=${url}`,
    `export ${opts.keyEnv}=${key}`,
  ].join('\n');
}
