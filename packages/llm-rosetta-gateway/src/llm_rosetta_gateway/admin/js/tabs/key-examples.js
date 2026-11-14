// API keys — the connect panel.
//
// The page is static: it shows every format the gateway speaks, with
// `<model>` where the operator's model id goes, so it reads the same on a
// deployment with one chat model and on one with every type configured.
//
// One client API at a time: the base URL, the model, then the sample that
// puts them together.  The request bodies come from
// `/admin/api/keys/examples`, which builds them with the converters the
// gateway itself runs, so a sample cannot describe a call it would refuse.

import { S } from '../core/state.js';
import { t, typeLabel } from '../core/i18n.js';
import { api, esc, copyText } from '../core/utils.js';
import { jsArg } from '../core/html.js';
import { buildSnippets, buildShell, snippetLangs } from '../core/api-examples.js';
import { codeBlockHtml } from '../core/code-block.js';

const OPEN_KEY = 'llm-rosetta-access-open';
const MODEL_TEMPLATE = '<model-id>';

/**
 * What a client needs beyond the format itself: where its SDK appends its
 * own route (so the base URL stops earlier) and which variables it reads.
 * Keyed `type/format`; anything unlisted falls back to the generic pair,
 * because the gateway accepts a bearer token everywhere.
 */
const WIRING = {
  'llm/chat_completions': { basePath: '/v1', baseEnv: 'OPENAI_BASE_URL', keyEnv: 'OPENAI_API_KEY' },
  'llm/open_responses': { basePath: '/v1', baseEnv: 'OPENAI_BASE_URL', keyEnv: 'OPENAI_API_KEY' },
  'llm/messages': { basePath: '', baseEnv: 'ANTHROPIC_BASE_URL', keyEnv: 'ANTHROPIC_API_KEY' },
  'llm/google_generate': { basePath: '', baseEnv: 'GOOGLE_GEMINI_BASE_URL', keyEnv: 'GEMINI_API_KEY' },
  'llm/google_interactions': { basePath: '', baseEnv: 'GOOGLE_GEMINI_BASE_URL', keyEnv: 'GEMINI_API_KEY' },
  'embeddings/openai': { basePath: '/v1', baseEnv: 'OPENAI_BASE_URL', keyEnv: 'OPENAI_API_KEY' },
  'embeddings/cohere': { basePath: '', baseEnv: 'COHERE_BASE_URL', keyEnv: 'COHERE_API_KEY' },
  'embeddings/jina': { basePath: '', baseEnv: 'JINA_BASE_URL', keyEnv: 'JINA_API_KEY' },
  'embeddings/voyage': { basePath: '', baseEnv: 'VOYAGE_BASE_URL', keyEnv: 'VOYAGE_API_KEY' },
  'embeddings/minimax': { basePath: '/v1', baseEnv: 'MINIMAX_BASE_URL', keyEnv: 'MINIMAX_API_KEY' },
  'embeddings/google_generate': { basePath: '', baseEnv: 'GOOGLE_GEMINI_BASE_URL', keyEnv: 'GEMINI_API_KEY' },
  'rerank/jina': { basePath: '/v1', baseEnv: 'JINA_BASE_URL', keyEnv: 'JINA_API_KEY' },
  'rerank/cohere': { basePath: '', baseEnv: 'COHERE_BASE_URL', keyEnv: 'COHERE_API_KEY' },
  'rerank/voyage': { basePath: '', baseEnv: 'VOYAGE_BASE_URL', keyEnv: 'VOYAGE_API_KEY' },
  'decisions/systemone': { basePath: '/v1', baseEnv: 'LLM_ROSETTA_BASE_URL', keyEnv: 'LLM_ROSETTA_API_KEY' },
  'decisions/openai_decisions': { basePath: '/v1', baseEnv: 'OPENAI_BASE_URL', keyEnv: 'OPENAI_API_KEY' },
};

const DEFAULT_WIRING = { basePath: '/v1', baseEnv: 'LLM_ROSETTA_BASE_URL', keyEnv: 'LLM_ROSETTA_API_KEY' };

const FORMAT_TITLES = {
  chat_completions: 'Chat Completions',
  open_responses: 'Responses',
  messages: 'Messages',
  google_generate: 'generateContent',
  google_interactions: 'Interactions',
  openai: 'OpenAI',
  cohere: 'Cohere',
  jina: 'Jina',
  voyage: 'Voyage',
  minimax: 'MiniMax',
  systemone: 'System One',
  openai_decisions: 'OpenAI Decisions',
};

function _formatTitle(format) {
  return FORMAT_TITLES[format] || format;
}

/** Every sample, in the order the server sent them (type, then picker). */
function _dialects() {
  const data = S._accessExamples;
  if (!data || !data.examples) return [];
  return data.examples.map(ex => ({
    id: `${ex.model_type}/${ex.format}`,
    type: ex.model_type,
    format: ex.format,
    example: ex,
    ...(WIRING[`${ex.model_type}/${ex.format}`] || DEFAULT_WIRING),
  }));
}

function _dialect() {
  const all = _dialects();
  return all.find(d => d.id === S._accessDialect) || all[0] || null;
}

function _langs() {
  return ['shell'].concat(snippetLangs());
}

function _open() {
  return S._accessOpen === true;
}

function toggleAccessPanel() {
  S._accessOpen = !_open();
  try { localStorage.setItem(OPEN_KEY, S._accessOpen ? '1' : '0'); } catch { /* private window */ }
  _applyOpen();
}

function _applyOpen() {
  const area = document.getElementById('keyExamplesArea');
  const btn = document.getElementById('accessToggle');
  if (area) area.style.display = _open() ? '' : 'none';
  if (btn) btn.setAttribute('aria-expanded', _open() ? 'true' : 'false');
  const section = document.getElementById('keysAccessSection');
  if (section) section.classList.toggle('is-collapsed', !_open());
  if (_open()) loadKeyExamples();
}

function _baseUrl(dialect) {
  return `${location.origin}${dialect.basePath}`;
}

/** The one-line description of the selected API, format first then type. */
function _desc(d) {
  const key = `keys.dialect.${d.format}Desc`;
  const text = t(key);
  return text === key ? t(`keys.dialect.${d.type}Desc`) : text;
}

function _row(label, control, hint, copy) {
  const copyBtn = copy
    ? `<button type="button" class="icon-btn is-bordered" onclick="copyAccessValue(${jsArg(copy)})" title="${esc(t('btn.copy'))}" aria-label="${esc(t('btn.copy'))}"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button>`
    : '';
  return `<div class="access-row">
    <span class="access-label">${esc(label)}</span>
    <div class="access-control">${control}${copyBtn}</div>
    ${hint ? `<p class="hint access-hint">${hint}</p>` : ''}
  </div>`;
}

/** Chips for every format, under the model type that owns them. */
function _dialectChips() {
  const all = _dialects();
  const active = _dialect();
  const groups = [];
  for (const d of all) {
    let g = groups.find(x => x.type === d.type);
    if (!g) { g = { type: d.type, items: [] }; groups.push(g); }
    g.items.push(d);
  }
  return groups.map(g => {
    const chips = g.items.map(d => {
      const on = active && d.id === active.id ? 'active' : '';
      return `<label class="${on}">` +
        `<input type="radio" name="accessDialect" value="${esc(d.id)}"${on ? ' checked' : ''} onchange="setAccessDialect(${jsArg(d.id)})">` +
        `<span>${esc(_formatTitle(d.format))}</span></label>`;
    }).join('');
    return `<div class="access-group"><span class="access-group-label">${esc(typeLabel(g.type))}</span>` +
      `<div class="facet-chips">${chips}</div></div>`;
  }).join('');
}

function _fillAndRender() {
  const area = document.getElementById('keyExamplesArea');
  if (!area) return;
  const d = _dialect();
  if (!d) {
    area.innerHTML = `<p class="empty">${esc(t('keys.examplesFailed'))}</p>`;
    return;
  }
  const lang = S._exampleLang || 'shell';
  const platform = S._examplePlatform || 'unix';
  const snippets = buildSnippets(d.example, { base: location.origin, env: d.keyEnv });
  const shell = buildShell({
    base: location.origin,
    basePath: d.basePath,
    baseEnv: d.baseEnv,
    keyEnv: d.keyEnv,
    platform,
  });

  const langSeg = `<div class="seg-control access-lang" role="tablist">` +
    _langs().map(l => `<button type="button" role="tab" data-lang="${l}" class="${l === lang ? 'active' : ''}" aria-selected="${l === lang}" onclick="setExampleLang(${jsArg(l)})">${esc(_langLabel(l))}</button>`).join('') +
    `</div>`;

  const platformSeg = lang === 'shell'
    ? `<div class="seg-control access-platform" role="tablist">` +
      ['unix', 'windows'].map(p => `<button type="button" role="tab" data-platform="${p}" class="${p === platform ? 'active' : ''}" aria-selected="${p === platform}" onclick="setExamplePlatform(${jsArg(p)})">${esc(p === 'unix' ? t('keys.accessUnix') : t('keys.accessWindows'))}</button>`).join('') +
      `</div>`
    : '';

  const code = lang === 'shell' ? shell : snippets[lang];

  area.innerHTML = `<div class="access-panel">
    ${_row(t('keys.accessApi'), `<div class="access-groups" id="accessDialects" role="radiogroup" aria-label="${esc(t('keys.accessApi'))}">${_dialectChips()}</div>`, esc(_desc(d)))}
    ${_row('Base URL', `<code class="access-value" id="accessBaseUrl">${esc(_baseUrl(d))}</code>`, esc(t('keys.accessBaseHint', { env: d.baseEnv })), _baseUrl(d))}
    ${_row(t('keys.exampleModel'), _modelControl(d), esc(t('keys.accessModelHint')), S._accessModel || MODEL_TEMPLATE)}
    <div class="access-row access-row-samples">
      <span class="access-label">${esc(t('keys.accessSamples'))}</span>
      <div class="access-control access-sample-controls">${langSeg}${platformSeg}</div>
    </div>
    ${codeBlockHtml(code, lang === 'node' ? 'js' : lang, {label: t('btn.copy'), className: 'example-code'})}
    <p class="hint access-hint access-foot">${esc(t('keys.accessFoot'))}</p>
  </div>`;
}

/**
 * The model field: the template by default, the configured models of this
 * type as an offer.  A deployment with none of that type gets the template
 * alone, which is why this is a select and not a list of links.
 */
function _modelControl(d) {
  const models = _modelsOf(d.type);
  if (!models.length) {
    return `<code class="access-value" id="accessModel">${esc(MODEL_TEMPLATE)}</code>`;
  }
  const current = S._accessModel || '';
  return `<select id="accessModel" onchange="setAccessModel(this.value)">` +
    `<option value=""${current ? '' : ' selected'}>${esc(MODEL_TEMPLATE)}</option>` +
    models.map(m => `<option value="${esc(m)}"${m === current ? ' selected' : ''}>${esc(m)}</option>`).join('') +
    `</select>`;
}

function _langLabel(lang) {
  if (lang === 'shell') return t('keys.accessEnv');
  if (lang === 'curl') return 'curl';
  if (lang === 'python') return 'Python';
  return 'Node';
}

/** Configured models of *type*, or an empty list on a deployment without. */
function _modelsOf(type) {
  const models = (S.configData && S.configData.models) || {};
  return Object.entries(models)
    .filter(([, cfg]) => ((cfg && cfg.type) || 'llm') === type)
    .map(([name]) => name)
    .sort();
}

async function loadKeyExamples() {
  const area = document.getElementById('keyExamplesArea');
  if (!area || !_open()) return;
  if (!S.configData) {
    // The sample works without it: the template is the default, and the
    // configured models are only an offer.
    try { S.configData = await api.get('/admin/api/config'); } catch (e) { console.error(e); }
  }
  if (S._accessExamples && S._accessExamples.model === (S._accessModel || MODEL_TEMPLATE)) {
    _fillAndRender();
    return;
  }
  try {
    const q = S._accessModel ? `?model=${encodeURIComponent(S._accessModel)}` : '';
    S._accessExamples = await api.get(`/admin/api/keys/examples${q}`);
  } catch (e) {
    console.error('Failed to load examples:', e);
    area.innerHTML = `<p class="empty">${esc(t('keys.examplesFailed'))}</p>`;
    return;
  }
  _fillAndRender();
}

function setAccessModel(model) {
  S._accessModel = model || '';
  loadKeyExamples();
}

function renderKeyExamples() {
  _fillAndRender();
}

function setAccessDialect(id) {
  if (!_dialects().some(d => d.id === id)) return;
  S._accessDialect = id;
  _fillAndRender();
}

function setExampleLang(lang) {
  if (!_langs().includes(lang)) return;
  S._exampleLang = lang;
  _fillAndRender();
}

function setExamplePlatform(p) {
  if (p !== 'unix' && p !== 'windows') return;
  S._examplePlatform = p;
  _fillAndRender();
}

function copyAccessValue(value) {
  copyText(String(value || ''), t('keys.exampleCopied'));
}

S._exampleLang = S._exampleLang || 'shell';
S._examplePlatform = S._examplePlatform || 'unix';
/** Collapsed until the reader opens it: the panel is for setting a tool up
 *  once, and it is the tallest thing on a tab whose job is the key list. */
try {
  S._accessOpen = localStorage.getItem(OPEN_KEY) === '1';
} catch { S._accessOpen = false; }

Object.assign(window, {
  loadKeyExamples, renderKeyExamples, setAccessDialect, toggleAccessPanel,
  setAccessModel, setExampleLang, setExamplePlatform, copyAccessValue,
});

export {
  loadKeyExamples, renderKeyExamples, setAccessDialect, toggleAccessPanel,
  setAccessModel, setExampleLang, setExamplePlatform, copyAccessValue, _applyOpen,
};
