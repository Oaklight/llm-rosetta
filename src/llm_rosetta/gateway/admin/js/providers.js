/**
 * providers.js — Provider management module for the admin panel.
 *
 * Handles provider CRUD, provider modal, multi-key management,
 * provider filtering/view toggling, server settings, network
 * diagnostics, and config loading.
 */

import { S, _CAP_ICONS } from './state.js';
import { t } from './i18n.js';
import { api, showToast, closeModal, esc, copyText, inlineConfirm } from './core.js';

// ── Module-local state ──────────────────────────────────────────────

let _editingProviderName = null; // original name when editing (for rename)
let _providerViewMode = localStorage.getItem('provider-view') || 'grid';
let _pendingDeleteProvider = '';
let _keyFieldIsMulti = false;
let _keyFieldVisible = false; // track visibility toggle for multi-key inputs

// ── Provider modal ──────────────────────────────────────────────────

function openProviderModal(name, baseUrl, apiKey, proxy, provType) {
  _editingProviderName = name || null;
  document.getElementById('providerModalTitle').textContent = name ? t('modal.editProvider') : t('modal.addProvider');
  document.getElementById('provName').value = name || '';
  document.getElementById('provName').readOnly = false;

  const registeredShims = (S.configData && S.configData.registered_shims) || [];
  const shimMap = Object.fromEntries(registeredShims.map(s => [s.name, s]));

  // Populate Provider Type dropdown from registered shims
  const typeSel = document.getElementById('provType');
  typeSel.innerHTML = '<option value="">' + t('label.selectOne', '— select —') + '</option>';
  for (const s of registeredShims) {
    const opt = document.createElement('option');
    opt.value = s.name; opt.textContent = s.name;
    if (s.name === provType) opt.selected = true;
    typeSel.appendChild(opt);
  }

  // Update the logo preview beside the dropdown
  function _updateTypeLogo() {
    const logo = document.getElementById('provTypeLogo');
    const s = shimMap[typeSel.value];
    if (s && s.logo) { logo.src = s.logo; logo.style.display = ''; }
    else { logo.src = ''; logo.style.display = 'none'; }
  }
  _updateTypeLogo();

  // Auto-fill base URL and API key placeholder when selecting a type
  let _prevShimUrl = (shimMap[provType || ''] || {}).default_base_url || '';
  typeSel.onchange = () => {
    _updateTypeLogo();
    const s = shimMap[typeSel.value];
    if (!s) return;
    const urlInput = document.getElementById('provBaseUrl');
    const cur = urlInput.value.trim();
    // Only auto-fill when the field is empty or still matches the previous shim's default
    if (s.default_base_url && (!cur || cur === _prevShimUrl)) {
      urlInput.value = s.default_base_url;
    }
    _prevShimUrl = s.default_base_url || '';
    const keyInput = document.getElementById('provApiKey');
    if (!keyInput.value && s.default_api_key_env) keyInput.value = '${' + s.default_api_key_env + '}';
    // Update custom tools checkbox to match new shim default
    const provCfg = _editingProviderName && S.configData ? S.configData.providers[_editingProviderName] : null;
    if (!provCfg || !('supports_custom_tools' in provCfg)) {
      document.getElementById('provCustomTools').checked = !!s.supports_custom_tools;
    }
    if (!provCfg || !('hoist_system_messages' in provCfg)) {
      document.getElementById('provHoistSystem').checked = s.hoist_system_messages !== false;
    }
    if (!provCfg || !('preflight_token_count' in provCfg)) {
      document.getElementById('provPreflightTokens').checked = false;
    }
  };

  document.getElementById('provBaseUrl').value = baseUrl || '';
  const keyInput = document.getElementById('provApiKey');
  keyInput.type = 'password';
  keyInput.readOnly = false;
  _resetKeyField();
  // Hide eye/copy buttons when credential visibility is off
  document.getElementById('provKeyToggleBtn').style.display = S._credentialVisible ? '' : 'none';
  document.getElementById('provKeyCopyBtn').style.display = S._credentialVisible ? '' : 'none';
  if (!S._credentialVisible) {
    // Don't reveal even masked key; leave empty so user can type a new one
    keyInput.value = '';
    keyInput.placeholder = name ? t('label.keyUnchangedHint', 'Leave blank to keep current key') : '${OPENAI_API_KEY}';
  } else {
    keyInput.value = apiKey || '';
    keyInput.placeholder = '${OPENAI_API_KEY}';
    _syncKeyField();
  }
  document.getElementById('provProxy').value = proxy || '';

  // URL template overrides
  const provCfg = name && S.configData ? S.configData.providers[name] : null;
  document.getElementById('provUrlTemplate').value = (provCfg && provCfg.url_template) || '';
  document.getElementById('provStreamUrlTemplate').value = (provCfg && provCfg.stream_url_template) || '';
  // Custom tools checkbox: config value > shim default
  const customToolsCb = document.getElementById('provCustomTools');
  if (provCfg && 'supports_custom_tools' in provCfg) {
    customToolsCb.checked = !!provCfg.supports_custom_tools;
  } else {
    const shimDefault = (shimMap[provType || ''] || {}).supports_custom_tools;
    customToolsCb.checked = !!shimDefault;
  }
  // Hoist system messages checkbox: config value > shim default
  const hoistCb = document.getElementById('provHoistSystem');
  if (provCfg && 'hoist_system_messages' in provCfg) {
    hoistCb.checked = !!provCfg.hoist_system_messages;
  } else {
    const shimHoist = (shimMap[provType || '' ] || {}).hoist_system_messages;
    hoistCb.checked = shimHoist !== false;
  }
  // Preflight token count checkbox: config value only (no shim default)
  const preflightCb = document.getElementById('provPreflightTokens');
  preflightCb.checked = !!(provCfg && provCfg.preflight_token_count);
  document.getElementById('provTimeout').value = (provCfg && provCfg.timeout != null) ? provCfg.timeout : '';
  document.getElementById('provTimeout').placeholder = (S.configData.server && S.configData.server.upstream_timeout) || 300;
  if (window._detectedHostIp) document.getElementById('provProxy').placeholder = `e.g. http://${window._detectedHostIp}:7890`;
  // Populate embedding/rerank capability checkboxes
  const provCaps = provCfg ? _getProviderCaps(provCfg, name) : ['llm'];
  document.getElementById('provCapLlm').checked = provCaps.includes('llm');
  document.getElementById('provCapEmbedding').checked = provCaps.includes('embedding');
  document.getElementById('provCapRerank').checked = provCaps.includes('rerank');
  // Populate format dropdowns
  const embFmts = S.configData?.embedding_formats || ['openai','cohere','jina','voyage'];
  const rrFmts = S.configData?.rerank_formats || ['jina','cohere','voyage'];
  const embFmtSel = document.getElementById('provEmbeddingFormat');
  embFmtSel.innerHTML = embFmts.map(f => `<option value="${f}">${f}</option>`).join('');
  const rrFmtSel = document.getElementById('provRerankFormat');
  rrFmtSel.innerHTML = rrFmts.map(f => `<option value="${f}">${f}</option>`).join('');
  if (provCfg) {
    if (provCfg.embedding_format) embFmtSel.value = provCfg.embedding_format;
    document.getElementById('provEmbeddingPath').value = provCfg.embedding_path || '/v1/embeddings';
    if (provCfg.rerank_format) rrFmtSel.value = provCfg.rerank_format;
    document.getElementById('provRerankPath').value = provCfg.rerank_path || '/v1/rerank';
  } else {
    document.getElementById('provEmbeddingPath').value = '';
    document.getElementById('provRerankPath').value = '';
  }
  toggleProvCapSection();
  document.getElementById('providerModal').classList.add('open');
  // When editing, fetch the real (unmasked) key
  if (name && S._credentialVisible) {
    api.get(`/admin/api/config/providers/${encodeURIComponent(name)}/key`).then(res => {
      if (res.api_key) {
        keyInput.value = res.api_key;
        _syncKeyField();
      }
    }).catch(() => {});
  }
}

function toggleProvCapSection() {
  const llm = document.getElementById('provCapLlm').checked;
  const embed = document.getElementById('provCapEmbedding').checked;
  const rerank = document.getElementById('provCapRerank').checked;
  document.getElementById('provLlmSection').classList.toggle('visible', llm);
  document.getElementById('provEmbeddingSection').classList.toggle('visible', embed);
  document.getElementById('provRerankSection').classList.toggle('visible', rerank);
  // Populate format dropdowns if shown
  if (embed) _populateFormatDropdown('provEmbeddingFormat', S.configData?.embedding_formats || ['openai','cohere','jina','voyage']);
  if (rerank) _populateFormatDropdown('provRerankFormat', S.configData?.rerank_formats || ['jina','cohere','voyage']);
}

function _populateFormatDropdown(selectId, formats) {
  const sel = document.getElementById(selectId);
  if (!sel) return;
  sel.innerHTML = formats.map(f => `<option value="${f}">${f}</option>`).join('');
}

// ── Multi-key management ────────────────────────────────────────────

function _syncKeyField() {
  if (!_keyFieldIsMulti) {
    // Currently in single-input mode — check if value has commas
    const val = document.getElementById('provApiKey').value;
    if (val.includes(',')) {
      const keys = val.split(',').map(k => k.trim()).filter(Boolean);
      _showMultiKeyUI(keys);
    }
  }
  // In multi mode, _syncKeyField is a no-op (entries manage themselves)
}

function _promoteToMultiKey() {
  const input = document.getElementById('provApiKey');
  const currentKey = input.value.trim();
  const keys = currentKey ? [currentKey, ''] : ['', ''];
  _showMultiKeyUI(keys);
  // Focus the new empty entry
  const inputs = document.querySelectorAll('#provApiKeyMultiBox .multi-key-row input');
  if (inputs.length > 1) inputs[inputs.length - 1].focus();
}

function _showMultiKeyUI(keys) {
  document.getElementById('provApiKeyRow').style.display = 'none';
  document.getElementById('provApiKeySingleFooter').style.display = 'none';
  document.getElementById('provApiKeyMultiBox').style.display = '';
  _keyFieldIsMulti = true;
  _keyFieldVisible = document.getElementById('provApiKey').type === 'text';
  _renderMultiKeys(keys);
}

function _renderMultiKeys(keys) {
  const box = document.getElementById('provApiKeyMultiBox');
  const inputType = _keyFieldVisible ? 'text' : 'password';
  const delSvg = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
  const eyeSvg = _keyFieldVisible
    ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/><path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/><line x1="1" y1="1" x2="23" y2="23"/></svg>'
    : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';

  let html = '<div class="multi-key-list">';
  keys.forEach((k, i) => {
    html += `<div class="multi-key-row">
      <input type="${inputType}" value="${esc(k)}" data-key-idx="${i}" placeholder="API key #${i+1}">
      <button type="button" class="key-btn" onclick="_removeKeyEntry(${i})" title="Remove">${delSvg}</button>
    </div>`;
  });
  html += '</div>';
  html += `<div class="multi-key-footer">
    <button type="button" class="btn btn-sm" onclick="_addKeyEntry()">+ ${t('label.addKey')}</button>
    <button type="button" class="key-btn" onclick="toggleModalKeyVisibility()" title="Toggle visibility">${eyeSvg}</button>
    <button type="button" class="key-btn" onclick="copyModalKey()" title="Copy"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button>
    <span class="key-count">${t('label.keyCount', {count: keys.length})}</span>
  </div>`;
  box.innerHTML = html;
}

function _getMultiKeys() {
  const inputs = document.querySelectorAll('#provApiKeyMultiBox .multi-key-row input');
  return [...inputs].map(el => el.value.trim()).filter(Boolean);
}

function _addKeyEntry() {
  const keys = _getMultiKeys();
  keys.push('');
  _renderMultiKeys(keys);
  // Focus the new empty input
  const inputs = document.querySelectorAll('#provApiKeyMultiBox .multi-key-row input');
  if (inputs.length) inputs[inputs.length - 1].focus();
}

function _removeKeyEntry(idx) {
  const keys = _getMultiKeys();
  keys.splice(idx, 1);
  if (keys.length <= 1) {
    // Switch back to single input
    const input = document.getElementById('provApiKey');
    input.value = keys[0] || '';
    input.type = _keyFieldVisible ? 'text' : 'password';
    document.getElementById('provApiKeyRow').style.display = '';
    document.getElementById('provApiKeySingleFooter').style.display = '';
    document.getElementById('provApiKeyMultiBox').style.display = 'none';
    _keyFieldIsMulti = false;
  } else {
    _renderMultiKeys(keys);
  }
}

function _resetKeyField() {
  document.getElementById('provApiKeyRow').style.display = '';
  document.getElementById('provApiKeySingleFooter').style.display = '';
  document.getElementById('provApiKeyMultiBox').style.display = 'none';
  document.getElementById('provApiKeyMultiBox').innerHTML = '';
  _keyFieldIsMulti = false;
  _keyFieldVisible = false;
}

function _getKeyFieldValue() {
  if (_keyFieldIsMulti) {
    return _getMultiKeys().join(',');
  }
  return document.getElementById('provApiKey').value.trim();
}

function toggleModalKeyVisibility() {
  if (_keyFieldIsMulti) {
    _keyFieldVisible = !_keyFieldVisible;
    const inputs = document.querySelectorAll('#provApiKeyMultiBox .multi-key-row input');
    inputs.forEach(el => { el.type = _keyFieldVisible ? 'text' : 'password'; });
    // Re-render to update eye icon
    _renderMultiKeys(_getMultiKeys());
  } else {
    const input = document.getElementById('provApiKey');
    input.type = input.type === 'password' ? 'text' : 'password';
  }
}

async function copyModalKey() {
  const val = _getKeyFieldValue();
  try {
    await navigator.clipboard.writeText(val);
    showToast('API key copied');
  } catch(e) {
    showToast('Copy failed', 'error');
  }
}

// ── Provider filter / view ──────────────────────────────────────────

function _activateSegChild(el) {
  el.parentElement.querySelectorAll(':scope > div').forEach(s => s.classList.remove('active'));
  el.classList.add('active');
}

function switchProviderFilter(el, filter) {
  _activateSegChild(el);
  S._providerFilter = filter;
  renderProviders();
}

function setProviderView(mode) {
  _providerViewMode = mode;
  localStorage.setItem('provider-view', mode);
  const grid = document.getElementById('providerGrid');
  grid.classList.toggle('list-view', mode === 'list');
  _updateViewToggle();
}

function _updateViewToggle() {
  const gridBtn = document.getElementById('viewGridBtn');
  const listBtn = document.getElementById('viewListBtn');
  if (gridBtn) gridBtn.classList.toggle('active', _providerViewMode === 'grid');
  if (listBtn) listBtn.classList.toggle('active', _providerViewMode === 'list');
}

// ── Provider rendering helpers ──────────────────────────────────────

function _getProviderCaps(cfg, provName) {
  const caps = [];
  // LLM if the provider has LLM models, or has url_template, or has no embedding/rerank fields at all
  const hasEmbedOrRerank = cfg.embedding_format || cfg.rerank_format;
  const hasLlmModels = provName && S.configData && S.configData.models && Object.values(S.configData.models).some(m => {
    const p = typeof m === 'string' ? m : m.provider;
    const t = typeof m === 'object' ? (m.type || 'llm') : 'llm';
    return p === provName && t === 'llm';
  });
  if (hasLlmModels || cfg.url_template || cfg.stream_url_template || !hasEmbedOrRerank) caps.push('llm');
  if (cfg.embedding_format) caps.push('embedding');
  if (cfg.rerank_format) caps.push('rerank');
  if (caps.length === 0) caps.push('llm');
  return caps;
}

function _capBadgesHtml(caps) {
  return caps.map(c => {
    const cls = c === 'llm' ? 'cap-badge-llm' : c === 'embedding' ? 'cap-badge-embedding' : 'cap-badge-rerank';
    return `<span class="cap-badge ${cls}">${_CAP_ICONS[c] || ''}${esc(c.toUpperCase())}</span>`;
  }).join('');
}

function _countModelsForProvider(provName) {
  if (!S.configData) return 0;
  let count = 0;
  for (const [, info] of Object.entries(S.configData.models || {})) {
    const prov = typeof info === 'string' ? info : info.provider;
    if (prov === provName) count++;
  }
  return count;
}

function goToModelsForProvider(provName) {
  const modelsTab = document.querySelector('[data-tab="models"]');
  if (modelsTab) { modelsTab.click(); }
  S._modelDomain = 'all';
  const allSeg = document.querySelector('#modelTypeSeg > div:first-child');
  if (allSeg) _activateSegChild(allSeg);
  window.renderModels();
  const filterSelect = document.getElementById('modelProviderFilter');
  filterSelect.value = provName;
  window.renderModels();
}

function goToProviderFromModel(provName) {
  const provTab = document.querySelector('[data-tab="providers"]');
  if (provTab) { provTab.click(); }
  S._providerFilter = 'all';
  const allSeg = document.querySelector('#providerSeg > div:first-child');
  if (allSeg) _activateSegChild(allSeg);
  renderProviders();
  requestAnimationFrame(() => {
    const card = document.querySelector(`[data-provider="${provName}"]`);
    if (card) {
      card.scrollIntoView({ behavior: 'smooth', block: 'center' });
      card.classList.add('highlight');
      setTimeout(() => card.classList.remove('highlight'), 2000);
    }
  });
}

// ── Main provider rendering ─────────────────────────────────────────

function renderProviders() {
  const grid = document.getElementById('providerGrid');
  const providers = S.configData.providers || {};
  const allEntries = Object.entries(providers);
  const totalCount = allEntries.length;

  // Show search input only when > 6 providers
  document.getElementById('providerSearch').style.display = totalCount > 6 ? '' : 'none';

  // Apply view mode class
  grid.classList.toggle('list-view', _providerViewMode === 'list');
  _updateViewToggle();

  if (totalCount === 0) {
    grid.innerHTML = `<p style="color:var(--text-dim)">${t('empty.providers')}</p>`;
    document.getElementById('providerCount').textContent = '';
    return;
  }

  // Filter by search query
  const query = (document.getElementById('providerSearch').value || '').trim().toLowerCase();
  let entries = allEntries;
  if (query) {
    entries = entries.filter(([name, cfg]) => {
      const typeName = cfg.type || name;
      return name.toLowerCase().includes(query) ||
        typeName.toLowerCase().includes(query) ||
        (cfg.base_url || '').toLowerCase().includes(query);
    });
  }

  // Filter by capability
  if (S._providerFilter !== 'all') {
    entries = entries.filter(([name, cfg]) => _getProviderCaps(cfg, name).includes(S._providerFilter));
  }

  // Update count
  const countEl = document.getElementById('providerCount');
  if (totalCount > 6) {
    countEl.textContent = query ? t('provider.count', {shown: entries.length, total: totalCount})
                                : t('provider.countTotal', {total: totalCount});
  } else {
    countEl.textContent = '';
  }

  const shimLogo = Object.fromEntries(
    (S.configData.registered_shims || []).map(s => [s.name, s.logo])
  );

  if (entries.length === 0) {
    grid.innerHTML = `<p style="color:var(--text-dim)">${t('empty.searchResults')}</p>`;
    return;
  }

  const isList = _providerViewMode === 'list';
  grid.innerHTML = entries.map(([name, cfg]) => {
    const enabled = cfg.enabled !== false;
    const typeName = cfg.type || name;
    const logo = shimLogo[typeName];
    const logoHtml = logo ? `<img class="provider-logo" src="${esc(logo)}" alt="">` : '';
    const fieldsHtml = isList
      ? `<div class="field" title="Type: ${esc(typeName)}"><code>${esc(typeName)}</code></div>
         <div class="field" title="${esc(cfg.base_url || '')}"><code>${esc(cfg.base_url || '')}</code></div>
         ${S._credentialVisible ? `<div class="field" title="${esc(cfg.api_key || '')}"><code>${esc(cfg.api_key || '')}</code></div>` : '<div class="field"></div>'}`
      : `<div class="field">Type: <code>${esc(typeName)}</code></div>
         <div class="field">Base URL: <code>${esc(cfg.base_url || '')}</code></div>
         ${S._credentialVisible ? `<div class="field">API Key: <code>${esc(cfg.api_key || '')}</code></div>` : ''}`;
    const provCaps = _getProviderCaps(cfg, name);
    const capBadges = _capBadgesHtml(provCaps);
    const modelCount = _countModelsForProvider(name);
    const modelLink = modelCount > 0
      ? `<span class="model-link" onclick="goToModelsForProvider('${esc(name)}')">${modelCount} model${modelCount !== 1 ? 's' : ''} →</span>`
      : '';
    // Endpoint details are in Edit modal — badges on card are sufficient
    return `
    <div class="provider-card${enabled ? '' : ' disabled'}" data-provider="${esc(name)}" style="display:flex;flex-direction:column">
      <div class="card-header">
        <div class="name" style="display:flex;align-items:center;gap:6px">${logoHtml}${esc(name)} ${capBadges}</div>
        <label class="toggle" title="${enabled ? t('provider.enabled') : t('provider.disabled')}">
          <input type="checkbox" ${enabled ? 'checked' : ''} onchange="toggleProvider('${esc(name)}')">
          <span class="slider"></span>
        </label>
      </div>
      ${fieldsHtml}
      <div class="actions" style="margin-top:auto;padding-top:12px">
        <button class="btn btn-sm" onclick="copyProviderEntry('${esc(name)}')">${t('btn.clone')}</button>
        <button class="btn btn-sm" onclick="editProvider('${esc(name)}')">${t('btn.edit')}</button>
        <button class="btn btn-sm btn-danger" onclick="deleteProvider('${esc(name)}')">${t('btn.delete')}</button>
        ${modelLink}
      </div>
    </div>`;
  }).join('');
}

// ── Provider CRUD ───────────────────────────────────────────────────

async function saveProvider() {
  const name = document.getElementById('provName').value.trim();
  const provType = document.getElementById('provType').value;
  const baseUrl = document.getElementById('provBaseUrl').value.trim();
  const apiKey = _getKeyFieldValue();
  const proxy = document.getElementById('provProxy').value.trim();
  const isLlm = document.getElementById('provCapLlm').checked;
  if (!name || (isLlm && !provType)) { showToast(t('error.fieldsRequired'), 'error'); return; }
  const urlTemplate = document.getElementById('provUrlTemplate').value.trim();
  const streamUrlTemplate = document.getElementById('provStreamUrlTemplate').value.trim();
  const body = {base_url: baseUrl, proxy};
  if (isLlm) body.type = provType;
  if (urlTemplate) body.url_template = urlTemplate;
  if (streamUrlTemplate) body.stream_url_template = streamUrlTemplate;
  body.supports_custom_tools = document.getElementById('provCustomTools').checked;
  body.hoist_system_messages = document.getElementById('provHoistSystem').checked;
  body.preflight_token_count = document.getElementById('provPreflightTokens').checked;
  // Embedding/rerank endpoint config
  if (document.getElementById('provCapEmbedding').checked) {
    body.embedding_format = document.getElementById('provEmbeddingFormat').value;
    body.embedding_path = document.getElementById('provEmbeddingPath').value.trim() || '/v1/embeddings';
  } else {
    body.embedding_format = '';
    body.embedding_path = '';
  }
  if (document.getElementById('provCapRerank').checked) {
    body.rerank_format = document.getElementById('provRerankFormat').value;
    body.rerank_path = document.getElementById('provRerankPath').value.trim() || '/v1/rerank';
  } else {
    body.rerank_format = '';
    body.rerank_path = '';
  }
  const provTimeoutVal = document.getElementById('provTimeout').value.trim();
  if (provTimeoutVal) body.timeout = parseFloat(provTimeoutVal);
  // When api_key is empty and we're editing, omit it so backend keeps the original
  if (apiKey) body.api_key = apiKey;
  // If editing and name changed, include rename_from so backend updates model references
  if (_editingProviderName && _editingProviderName !== name) {
    body.rename_from = _editingProviderName;
  }
  const res = await api.put(`/admin/api/config/providers/${encodeURIComponent(name)}`, body);
  if (res.ok) { showToast(t('toast.providerSaved',{name})); closeModal('providerModal'); loadConfig(); }
  else { showToast(res.error || 'Failed', 'error'); }
}

function deleteProvider(name) {
  _pendingDeleteProvider = name;
  document.getElementById('deleteConfirmMsg').innerHTML = t('confirm.typeToConfirm', {name: '<strong>' + esc(name) + '</strong>'});
  document.getElementById('deleteConfirmInput').value = '';
  document.getElementById('deleteConfirmBtn').disabled = true;
  // Reset button style
  const btn = document.getElementById('deleteConfirmBtn');
  btn.style.background = '#ccc'; btn.style.color = '#888'; btn.style.cursor = 'not-allowed';
  // Show affected models if any
  const affected = Object.entries(S.configData.models || {})
    .filter(([, v]) => (typeof v === 'object' ? v.provider : v) === name)
    .map(([m]) => m);
  const el = document.getElementById('deleteAffectedModels');
  if (affected.length > 0) {
    el.innerHTML = `<div style="color:#cf222e;font-weight:600;margin-bottom:4px">${t('confirm.cascadeWarning', {count: affected.length})}</div>`
      + affected.map(m => `<div style="color:var(--text-dim)">• ${esc(m)}</div>`).join('');
    el.style.display = 'block';
  } else {
    el.style.display = 'none';
  }
  document.getElementById('deleteConfirmModal').classList.add('open');
  document.getElementById('deleteConfirmInput').focus();
}

function onDeleteConfirmInput() {
  const btn = document.getElementById('deleteConfirmBtn');
  const matched = document.getElementById('deleteConfirmInput').value === _pendingDeleteProvider;
  btn.disabled = !matched;
  if (matched) {
    btn.style.background = '#cf222e';
    btn.style.color = '#fff';
    btn.style.cursor = 'pointer';
  } else {
    btn.style.background = '#ccc';
    btn.style.color = '#888';
    btn.style.cursor = 'not-allowed';
  }
}

function onDeleteConfirmClick() {
  const btn = document.getElementById('deleteConfirmBtn');
  if (btn.disabled) {
    const input = document.getElementById('deleteConfirmInput');
    input.style.outline = '2px solid #cf222e';
    input.focus();
    setTimeout(() => { input.style.outline = ''; }, 1200);
    return;
  }
  confirmDeleteProvider();
}

async function confirmDeleteProvider() {
  const name = _pendingDeleteProvider;
  if (!name) return;
  const res = await api.del(`/admin/api/config/providers/${encodeURIComponent(name)}?cascade=true`);
  closeModal('deleteConfirmModal');
  if (res.ok) {
    const cascaded = res.cascade_deleted_models || [];
    if (cascaded.length > 0) {
      showToast(t('toast.providerDeleted',{name}) + ` (+${cascaded.length} models)`);
    } else {
      showToast(t('toast.providerDeleted',{name}));
    }
    loadConfig();
  } else { showToast(res.error || 'Failed', 'error'); }
}

async function toggleProvider(name) {
  const res = await api.post(`/admin/api/config/providers/${encodeURIComponent(name)}/toggle`);
  if (res.ok) {
    const stateKey = res.enabled ? 'provider.enabled' : 'provider.disabled';
    showToast(`${name}: ${t(stateKey)}`);
    loadConfig();
  } else { showToast(res.error || 'Failed', 'error'); }
}

function editProvider(name) {
  const cfg = S.configData.providers[name] || {};
  openProviderModal(name, cfg.base_url, cfg.api_key, cfg.proxy, cfg.type);
}

function copyProviderEntry(name) {
  const cfg = S.configData.providers[name] || {};
  // Open Add Provider modal pre-filled with non-sensitive fields; API key is intentionally omitted
  openProviderModal('', cfg.base_url, '', cfg.proxy, cfg.type || name);
}

// ── Server settings ─────────────────────────────────────────────────

async function saveServerSettings() {
  const proxy = document.getElementById('globalProxy').value.trim();
  const res = await api.put('/admin/api/config/server', {proxy});
  if (res.ok) { showToast(t('toast.serverSaved')); loadConfig(); }
  else { showToast(res.error || 'Failed', 'error'); }
}

async function runNetDiag() {
  const el = document.getElementById('netDiagResults');
  el.innerHTML = `
    <span class="diag-item"><span class="diag-dot loading"></span>${t('diag.ip')}: ${t('diag.testing')}</span>
    <span class="diag-item"><span class="diag-dot loading"></span>${t('diag.host')}: ${t('diag.testing')}</span>
    <span class="diag-item"><span class="diag-dot loading"></span>${t('diag.google')}: ${t('diag.testing')}</span>`;
  try {
    const data = await api.get('/admin/api/diagnostics/network');
    const ipInfo = data.ip || {};
    const hostInfo = data.host || {};
    const googleInfo = data.google || {};
    // Store detected host IP for provider form hints
    if (hostInfo.ok) window._detectedHostIp = hostInfo.ip;
    const proxyText = data.proxy ? `(via ${esc(data.proxy)})` : `(${t('diag.direct')})`;
    const ipText = ipInfo.ok ? `${ipInfo.ip} (${ipInfo.city}, ${ipInfo.country})` : (ipInfo.error || t('diag.unknown'));
    const googleText = googleInfo.ok ? 'OK' : (googleInfo.error || 'unreachable');
    el.innerHTML = `
      <span class="diag-item" style="color:var(--text-dim);font-size:11px">${proxyText}</span>
      <span class="diag-item"><span class="diag-dot ${ipInfo.ok ? 'ok' : 'fail'}"></span>${t('diag.ip')}: ${esc(ipText)}</span>
      <span class="diag-item"><span class="diag-dot ${hostInfo.ok ? 'ok' : 'fail'}"></span>${t('diag.host')}: ${hostInfo.ok ? esc(hostInfo.ip) : esc(hostInfo.error || t('diag.unknown'))}</span>
      <span class="diag-item"><span class="diag-dot ${googleInfo.ok ? 'ok' : 'fail'}"></span>${t('diag.google')}: ${esc(googleText)}</span>`;
  } catch(e) {
    el.innerHTML = `<span class="diag-item"><span class="diag-dot fail"></span>Error: ${esc(String(e))}</span>`;
  }
}

// ── Config loading ──────────────────────────────────────────────────

async function loadConfig() {
  try {
    S.configData = await api.get('/admin/api/config');
    S._credentialVisible = S.configData.credential_visible !== false;
    document.getElementById('configPath').textContent = S.configData.config_path || '';
    document.getElementById('globalProxy').value = (S.configData.server && S.configData.server.proxy) || '';
    renderProviders();
    window.renderModels();
    // Detect host IP in background (for proxy placeholder hints)
    if (!window._detectedHostIp) {
      api.get('/admin/api/diagnostics/host-ip').then(data => {
        if (data.ok) {
          window._detectedHostIp = data.ip;
          document.getElementById('globalProxy').placeholder = `e.g. http://${data.ip}:7890`;
        }
      }).catch(() => {});
    }
  } catch(e) {
    console.error('Failed to load config:', e);
  }
}

// ── Window bindings ─────────────────────────────────────────────────

Object.assign(window, {
  openProviderModal, toggleProvCapSection, toggleModalKeyVisibility, copyModalKey,
  _syncKeyField, _promoteToMultiKey, _addKeyEntry, _removeKeyEntry,
  switchProviderFilter, setProviderView, renderProviders,
  saveProvider, deleteProvider, onDeleteConfirmInput, onDeleteConfirmClick,
  confirmDeleteProvider, toggleProvider, editProvider, copyProviderEntry,
  saveServerSettings, runNetDiag, loadConfig,
  goToModelsForProvider, goToProviderFromModel,
  _activateSegChild,
});

export { renderProviders, loadConfig, _activateSegChild, _getProviderCaps };
