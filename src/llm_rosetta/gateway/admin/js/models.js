// models.js — Model management: modal, CRUD, rendering, filtering, bulk ops
import { S, _CAP_ICONS } from './state.js';
import { t } from './i18n.js';
import { api, showToast, closeModal, esc, inlineConfirm } from './core.js';

// ── helpers (module-private) ──

function _activateSegChild(el) {
  el.parentElement.querySelectorAll(':scope > div').forEach(s => {
    s.classList.remove('active');
    if (s.hasAttribute('role')) { s.setAttribute('aria-checked', 'false'); s.setAttribute('tabindex', '-1'); }
  });
  el.classList.add('active');
  if (el.hasAttribute('role')) { el.setAttribute('aria-checked', 'true'); el.setAttribute('tabindex', '0'); }
}

function _getModelType(info) {
  if (typeof info === 'string') return 'llm';
  if (info.type) return info.type;
  const caps = info.capabilities || ['text'];
  if (caps.includes('embedding') && !caps.includes('text')) return 'embedding';
  return 'llm';
}

// ── Model modal ──

function switchModelTab(clicked, showId) {
  const modal = clicked.closest('.modal');
  modal.querySelectorAll('.model-tab-content').forEach(el => el.classList.remove('active'));
  modal.querySelector('#' + showId).classList.add('active');
  clicked.parentElement.querySelectorAll('.model-tab').forEach(t => t.classList.remove('active'));
  clicked.classList.add('active');
}

function segModelType(el, value) {
  el.parentElement.querySelectorAll(':scope > *').forEach(l => l.classList.remove('active'));
  el.classList.add('active');
  el.querySelector('input').checked = true;
  onModelTypeChange();
}

function toggleCap(el, checkboxId) {
  const cb = document.getElementById(checkboxId);
  cb.checked = !cb.checked;
  el.classList.toggle('active', cb.checked);
  if (checkboxId === 'capReasoning') updateReasoningVisibility();
}

function toggleStreamUrl(el) {
  const group = document.getElementById('modelStreamUrlGroup');
  const icon = el.querySelector('span:first-child');
  if (group.style.display === 'none') {
    group.style.display = 'block';
    icon.textContent = '−';
  } else {
    group.style.display = 'none';
    icon.textContent = '+';
  }
}

function openModelModal(model, provider, capabilities, upstreamModel, sourceModel) {
  // sourceModel: when cloning, the original model name to read reasoning from
  // (the visible name field is left blank so the user picks a new name).
  document.getElementById('modelModalTitle').textContent = model ? t('modal.editModel') : t('modal.addModel');
  document.getElementById('modelName').value = model || '';
  document.getElementById('modelName').dataset.originalName = model || '';
  document.getElementById('modelUpstream').value = upstreamModel || '';
  // Populate provider dropdown from config
  const sel = document.getElementById('modelProvider');
  sel.innerHTML = '';
  if (S.configData) {
    for (const p of Object.keys(S.configData.providers)) {
      const opt = document.createElement('option');
      opt.value = p; opt.textContent = p;
      if (p === provider) opt.selected = true;
      sel.appendChild(opt);
    }
  }
  // Reset to General tab
  const modal = document.getElementById('modelModal');
  modal.querySelectorAll('.model-tab-content').forEach(el => el.classList.remove('active'));
  modal.querySelector('#modelTab-general').classList.add('active');
  modal.querySelectorAll('.model-tab').forEach(t => t.classList.remove('active'));
  modal.querySelector('.model-tab').classList.add('active');
  // Reset stream URL expand
  document.getElementById('modelStreamUrlGroup').style.display = 'none';
  const expandIcon = modal.querySelector('.expand-link span:first-child');
  if (expandIcon) expandIcon.textContent = '+';

  // Set model type via seg-control
  const caps = capabilities || ['text', 'tools'];
  const isEmbedding = caps.includes('embedding') && !caps.includes('text');
  const isRerank = caps.includes('rerank') && !caps.includes('text');
  const detectedType = isRerank ? 'rerank' : isEmbedding ? 'embedding' : 'llm';
  const segLabels = document.getElementById('modelModalTypeSeg').querySelectorAll(':scope > div');
  segLabels.forEach(l => l.classList.remove('active'));
  const typeIdx = detectedType === 'rerank' ? 2 : detectedType === 'embedding' ? 1 : 0;
  segLabels[typeIdx].classList.add('active');
  document.querySelector('input[name="modelType"][value="' + detectedType + '"]').checked = true;
  // Set capability chips
  const chipMap = {capText: 'text', capVision: 'vision', capTools: 'tools', capReasoning: 'reasoning'};
  for (const [id, cap] of Object.entries(chipMap)) {
    const cb = document.getElementById(id);
    cb.checked = caps.includes(cap);
    cb.closest('.chip').classList.toggle('active', caps.includes(cap));
  }
  // flatten_system — read from model config, or infer auto-detect for gemini
  const fsModel = model && S.configData ? S.configData.models[model] : null;
  const fsCfg = fsModel && fsModel.flatten_system;
  const fsAuto = model && /gemini/i.test(model);
  const modelCfg = model && S.configData.models[model];
  document.getElementById('modelTimeout').value = (typeof modelCfg === 'object' && modelCfg && modelCfg.timeout != null) ? modelCfg.timeout : '';
  document.getElementById('flattenSystem').checked = fsCfg != null ? !!fsCfg : !!fsAuto;
  // URL template overrides
  document.getElementById('modelUrlTemplate').value = (fsModel && fsModel.url_template) || '';
  const streamTpl = (fsModel && fsModel.stream_url_template) || '';
  document.getElementById('modelStreamUrlTemplate').value = streamTpl;
  if (streamTpl) {
    document.getElementById('modelStreamUrlGroup').style.display = 'block';
    const ei = modal.querySelector('.expand-link span:first-child');
    if (ei) ei.textContent = '−';
  }
  onModelTypeChange();

  // Populate reasoning config section.
  // Read from the edited model, or the source model when cloning.
  const lookupName = model || sourceModel;
  const info = lookupName && S.configData ? S.configData.models[lookupName] : null;
  const reasoning = (info && info.reasoning) || {};
  const configOverride = (info && info.reasoning_override) || {};
  const srcBadge = document.getElementById('reasoningSourceBadge');
  srcBadge.textContent = reasoning.source || 'none';
  srcBadge.className = 'badge badge-sm' + (reasoning.source === 'config' ? ' badge-cap-tools' : reasoning.source === 'model_override' ? ' badge-cap-reasoning' : '');
  // Update inherit option labels: show "value (inherited)" or just "(inherit)"
  const inh = t('label.inherited');
  const ttInherit = document.querySelector('#reasoningThinkingType option[value=""]');
  ttInherit.textContent = reasoning.thinking_type ? reasoning.thinking_type + ' (' + inh + ')' : '(' + inh + ')';
  const br = document.getElementById('reasoningBudgetRatio');
  br.placeholder = reasoning.budget_tokens_default_ratio != null ? reasoning.budget_tokens_default_ratio + ' (' + inh + ')' : '(' + inh + ')';
  const dsInherit = document.querySelector('#reasoningDisabled option[value=""]');
  dsInherit.textContent = reasoning.disabled ? reasoning.disabled + ' (' + inh + ')' : '(' + inh + ')';
  // Set values — config override if present, otherwise blank (inheriting)
  document.getElementById('reasoningThinkingType').value = configOverride.thinking_type || '';
  document.getElementById('reasoningBudgetRatio').value = configOverride.budget_tokens_default_ratio != null ? configOverride.budget_tokens_default_ratio : '';
  document.getElementById('reasoningDisabled').value = configOverride.disabled || '';

  openModal('modelModal');
}

function onModelTypeChange() {
  const isLLM = document.querySelector('input[name="modelType"]:checked').value === 'llm';
  document.getElementById('modelCapsGroup').style.display = isLLM ? '' : 'none';
  updateReasoningVisibility();
}

function updateReasoningVisibility() {
  const isLLM = document.querySelector('input[name="modelType"]:checked').value === 'llm';
  const hasReasoning = isLLM && document.getElementById('capReasoning').checked;
  document.getElementById('modelReasoningGroup').style.display = hasReasoning ? '' : 'none';
}

// ── Model filter / sort / domain ──

function switchModelDomain(el, domain) {
  _activateSegChild(el);
  S._modelDomain = domain;
  renderModels();
}

function resetModelFilters() {
  S._modelDomain = 'all';
  const allSeg = document.querySelector('#modelTypeSeg > div:first-child');
  if (allSeg) _activateSegChild(allSeg);
  document.getElementById('modelProviderFilter').value = '';
  document.getElementById('modelSearch').value = '';
  renderModels();
}

function sortModels(key) {
  if (S._modelSortKey === key) {
    S._modelSortDir = S._modelSortDir === 'asc' ? 'desc' : 'asc';
  } else {
    S._modelSortKey = key;
    S._modelSortDir = 'asc';
  }
  renderModels();
}

// ── Cross-tab navigation ──

function goToModelsForProvider(provName) {
  const modelsTab = document.querySelector('[data-tab="models"]');
  if (modelsTab) { modelsTab.click(); }
  S._modelDomain = 'all';
  const allSeg = document.querySelector('#modelTypeSeg > div:first-child');
  if (allSeg) _activateSegChild(allSeg);
  renderModels();
  const filterSelect = document.getElementById('modelProviderFilter');
  filterSelect.value = provName;
  renderModels();
}

function goToProviderFromModel(provName) {
  const provTab = document.querySelector('[data-tab="providers"]');
  if (provTab) { provTab.click(); }
  S._providerFilter = 'all';
  const allSeg = document.querySelector('#providerSeg > div:first-child');
  if (allSeg) _activateSegChild(allSeg);
  window.renderProviders();
  requestAnimationFrame(() => {
    const card = document.querySelector(`[data-provider="${provName}"]`);
    if (card) {
      card.scrollIntoView({ behavior: 'smooth', block: 'center' });
      card.classList.add('highlight');
      setTimeout(() => card.classList.remove('highlight'), 2000);
    }
  });
}

// ── Model rendering ──

function renderModels() {
  const tbody = document.getElementById('modelTable');
  const models = S.configData.models || {};
  const totalCount = Object.keys(models).length;

  // Populate provider filter dropdown (preserve selection)
  const provFilter = document.getElementById('modelProviderFilter');
  const prevProv = provFilter.value;
  const provSet = new Set();
  for (const [, info] of Object.entries(models)) {
    provSet.add(typeof info === 'string' ? info : info.provider);
  }
  provFilter.innerHTML = `<option value="">${t('filter.allProviders')}</option>` +
    [...provSet].sort().map(p => `<option value="${esc(p)}"${p === prevProv ? ' selected' : ''}>${esc(p)}</option>`).join('');

  // Update sort header indicators
  document.getElementById('sortName').className = 'sortable' + (S._modelSortKey === 'name' ? ` ${S._modelSortDir}` : '');
  document.getElementById('sortProvider').className = 'sortable' + (S._modelSortKey === 'provider' ? ` ${S._modelSortDir}` : '');

  if (totalCount === 0) {
    tbody.innerHTML = `<tr><td colspan="4" style="color:var(--text-dim)">${t('empty.models')}</td></tr>`;
    document.getElementById('modelCount').textContent = '';
    return;
  }

  // Filter by provider dropdown and search query
  const selectedProv = provFilter.value;
  const query = (document.getElementById('modelSearch').value || '').trim().toLowerCase();
  let entries = Object.entries(models).map(([name, info]) => {
    const prov = typeof info === 'string' ? info : info.provider;
    return {name, prov, info};
  });
  if (selectedProv) {
    entries = entries.filter(e => e.prov === selectedProv);
  }
  if (query) {
    entries = entries.filter(e => e.name.toLowerCase().includes(query) || e.prov.toLowerCase().includes(query));
  }

  // Filter by model type (domain)
  if (S._modelDomain !== 'all') {
    entries = entries.filter(e => _getModelType(e.info) === S._modelDomain);
  }

  // Show/hide reset button
  const hasFilters = S._modelDomain !== 'all' || selectedProv || query;
  const resetBtn = document.getElementById('modelResetBtn');
  if (resetBtn) resetBtn.classList.toggle('visible', !!hasFilters);

  // Sort
  const dir = S._modelSortDir === 'asc' ? 1 : -1;
  entries.sort((a, b) => {
    const va = S._modelSortKey === 'provider' ? a.prov : a.name;
    const vb = S._modelSortKey === 'provider' ? b.prov : b.name;
    return va.localeCompare(vb) * dir;
  });

  // Update count
  const countEl = document.getElementById('modelCount');
  countEl.textContent = t('model.count', {shown: entries.length, total: totalCount});

  if (entries.length === 0) {
    tbody.innerHTML = `<tr><td colspan="4" style="color:var(--text-dim)">${t('empty.models')}</td></tr>`;
    return;
  }

  // Determine which providers are disabled
  const providers = S.configData.providers || {};
  const disabledProviders = new Set(
    Object.entries(providers).filter(([,cfg]) => cfg.enabled === false).map(([n]) => n)
  );

  tbody.innerHTML = entries.map(({name, prov, info}) => {
    const provDisabled = disabledProviders.has(prov);
    const modelEnabled = typeof info === 'object' ? info.enabled !== false : true;
    const rowDimmed = provDisabled || !modelEnabled;
    const caps = typeof info === 'string' ? ['text'] : (info.capabilities || ['text']);
    const modelType = _getModelType(info);
    const capBadges = modelType === 'llm' ? caps.map(c => {
      const cls = c === 'vision' ? 'badge-cap-vision' : c === 'tools' ? 'badge-cap-tools' : c === 'embedding' ? 'badge-cap-embed' : c === 'reasoning' ? 'badge-cap-reasoning' : 'badge-cap';
      return `<span class="badge ${cls}">${esc(c)}</span>`;
    }).join('') : '';
    const hasVision = caps.includes('vision');
    const hasTools = caps.includes('tools');
    const hasReasoning = caps.includes('reasoning');
    const isEmbedding = caps.includes('embedding');
    const typeBadgeCls = modelType === 'llm' ? 'cap-badge-llm' : modelType === 'embedding' ? 'cap-badge-embedding' : 'cap-badge-rerank';
    const typeBadge = `<span class="cap-badge ${typeBadgeCls}">${_CAP_ICONS[modelType] || ''}${esc(modelType.toUpperCase())}</span>`;
    const upstream = (typeof info === 'object' && info.upstream_model) ? info.upstream_model : '';
    const upstreamTag = upstream ? ` <span style="font-size:11px;color:var(--text-dim)" title="Upstream: ${esc(upstream)}">→ ${esc(upstream)}</span>` : '';
    const urlTpl = (typeof info === 'object' && info.url_template) ? info.url_template : '';
    const urlTplTag = urlTpl ? ` <span style="font-size:10px;color:var(--text-dim);background:var(--bg-card);padding:1px 4px;border-radius:3px;border:1px solid var(--border)" title="URL Template: ${esc(urlTpl)}">⛓ url</span>` : '';
    return `<tr${rowDimmed ? ' class="model-disabled"' : ''}>
      <td><input type="checkbox" class="row-check" data-model="${esc(name)}" onchange="updateModelBulk()"></td>
      <td class="model-name-cell">
        <code class="copyable" title="${esc(name)}" onclick="copyText('${esc(name)}')" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;display:block">${esc(name)}</code>
        ${upstreamTag || urlTplTag ? `<div style="margin-top:2px">${upstreamTag}${urlTplTag}</div>` : ''}
      </td>
      <td style="text-align:center">${typeBadge}</td>
      <td style="text-align:center">${capBadges || '<span style="color:var(--text-dim);font-size:11px">—</span>'}</td>
      <td><span class="provider-link" onclick="goToProviderFromModel('${esc(prov)}')">${esc(prov)}</span>${provDisabled ? ` <span style="color:var(--text-dim);font-size:11px">(${t('provider.disabled')})</span>` : ''}</td>
      <td style="text-align:right;white-space:nowrap;position:relative">
        <div class="pill-toggle ${modelEnabled ? 'is-on' : 'is-off'}" role="switch" aria-checked="${modelEnabled}" aria-label="${esc(name)}" onclick="toggleModel('${esc(name)}')" title="${modelEnabled ? t('model.enabled') : t('model.disabled')}" style="vertical-align:middle;margin-right:4px"><span class="pill-on">${t('label.on')}</span><span class="pill-off">${t('label.off')}</span></div>
        <div class="test-group" style="display:inline-block">
          <button class="btn btn-sm btn-test${modelType !== 'llm' ? ' btn-test-embed' : ''}" onclick="runTest('${esc(name)}','${modelType === 'embedding' ? 'embedding' : modelType === 'rerank' ? 'rerank' : 'text'}')">${t('btn.test')}</button>
          <button class="btn btn-sm btn-caret" onclick="toggleTestMenu(this)">&#9662;</button>
          <div class="test-menu">
            ${modelType === 'embedding' ? `
            <div class="test-menu-item" onclick="runTest('${esc(name)}','embedding')">${t('test.embedding')}</div>
            <div class="test-menu-item" onclick="runTest('${esc(name)}','embed_batch')">${t('test.embedBatch')}</div>
            <div class="test-menu-item" onclick="promptMatryoshka('${esc(name)}')">${t('test.matryoshka')}</div>
            <div class="test-menu-item" onclick="runTest('${esc(name)}','embed_multimodal')">${t('test.embedMultimodal')}</div>
            ` : modelType === 'rerank' ? `
            <div class="test-menu-item" onclick="runTest('${esc(name)}','rerank')">${t('test.rerank')}</div>
            <div class="test-menu-item" onclick="runTest('${esc(name)}','rerank_batch')">${t('test.rerankBatch')}</div>
            ` : `
            <div class="test-menu-item" onclick="runTest('${esc(name)}','text')">${t('test.text')}</div>
            <div class="test-menu-item" onclick="runTest('${esc(name)}','stream')">${t('test.stream')}</div>
            <div class="test-menu-item${hasTools ? '' : ' disabled'}" onclick="${hasTools ? `runTest('${esc(name)}','tools')` : ''}">${t('test.tools')}</div>
            <div class="test-menu-item${hasVision ? '' : ' disabled'}" onclick="${hasVision ? `runTest('${esc(name)}','vision')` : ''}">${t('test.vision')}</div>
            <div class="test-menu-item${hasReasoning ? '' : ' disabled'}" onclick="${hasReasoning ? `runTest('${esc(name)}','reasoning')` : ''}">${t('test.reasoning')}</div>
            `}
          </div>
        </div>
        <button class="btn btn-sm" aria-label="${t('btn.edit')} ${esc(name)}" onclick="editModel('${esc(name)}','${esc(prov)}')">${t('btn.edit')}</button>
        <button class="btn btn-sm" onclick="toggleMoreMenu(this)" style="padding:3px 6px">⋯</button>
        <div class="more-menu" style="display:none;position:absolute;right:0;top:calc(100% + 4px);min-width:110px;background:var(--bg-card);border:1px solid var(--border);border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.12);z-index:10;overflow:hidden">
          <div style="padding:7px 14px;font-size:13px;cursor:pointer" onmouseenter="this.style.background='var(--bg)'" onmouseleave="this.style.background=''" aria-label="${t('btn.clone')} ${esc(name)}" onclick="this.closest('.more-menu').style.display='none';cloneModel('${esc(name)}')">${t('btn.clone')}</div>
          <div style="border-top:1px solid var(--border);margin:2px 0"></div>
          <div style="padding:7px 14px;font-size:13px;cursor:pointer;color:var(--danger, #cf222e)" onmouseenter="this.style.background='#fff1f0'" onmouseleave="this.style.background=''" aria-label="${t('btn.delete')} ${esc(name)}" onclick="this.closest('.more-menu').style.display='none';deleteModel('${esc(name)}', this)">${t('btn.delete')}</div>
        </div>
      </td>
    </tr>`;
  }).join('');

  // Update filter dropdowns
  window.updateFilterOptions();
}

// ── Model CRUD ──

async function saveModel() {
  const name = document.getElementById('modelName').value.trim();
  const provider = document.getElementById('modelProvider').value;
  if (!name || !provider) { showToast(t('error.modelRequired'), 'error'); return; }
  const modelType = document.querySelector('input[name="modelType"]:checked').value;
  let capabilities;
  if (modelType === 'embedding') {
    capabilities = ['embedding'];
  } else if (modelType === 'rerank') {
    capabilities = ['rerank'];
  } else {
    capabilities = [];
    if (document.getElementById('capText').checked) capabilities.push('text');
    if (document.getElementById('capVision').checked) capabilities.push('vision');
    if (document.getElementById('capTools').checked) capabilities.push('tools');
    if (document.getElementById('capReasoning').checked) capabilities.push('reasoning');
    if (capabilities.length === 0) capabilities.push('text');
  }
  const body = {provider, capabilities};
  if (modelType !== 'llm') body.type = modelType;
  const upstreamModel = document.getElementById('modelUpstream').value.trim();
  if (upstreamModel) body.upstream_model = upstreamModel;
  // Collect reasoning override if reasoning capability is enabled
  if (capabilities.includes('reasoning')) {
    const thinkingType = document.getElementById('reasoningThinkingType').value || null;
    const budgetRatioStr = document.getElementById('reasoningBudgetRatio').value;
    const disabledStrategy = document.getElementById('reasoningDisabled').value || null;
    const override = {};
    if (thinkingType) override.thinking_type = thinkingType;
    if (budgetRatioStr !== '') override.budget_tokens_default_ratio = parseFloat(budgetRatioStr);
    if (disabledStrategy) override.disabled = disabledStrategy;
    if (Object.keys(override).length > 0) body.reasoning_override = override;
  }
  // URL template overrides
  const urlTpl = document.getElementById('modelUrlTemplate').value.trim();
  const streamUrlTpl = document.getElementById('modelStreamUrlTemplate').value.trim();
  if (urlTpl) body.url_template = urlTpl;
  if (streamUrlTpl) body.stream_url_template = streamUrlTpl;
  // flatten_system
  const modelTimeoutVal = document.getElementById('modelTimeout').value.trim();
  if (modelTimeoutVal) body.timeout = parseFloat(modelTimeoutVal);
  body.flatten_system = document.getElementById('flattenSystem').checked;
  const originalName = document.getElementById('modelName').dataset.originalName;
  if (originalName && originalName !== name) {
    body.rename_from = originalName;
  }
  const res = await api.put(`/admin/api/config/models/${encodeURIComponent(name)}`, body);
  if (res.ok) { showToast(t('toast.modelSaved',{name})); closeModal('modelModal'); window.loadConfig(); }
  else { showToast(res.error || 'Failed', 'error'); }
}

async function _doDeleteModel(name) {
  const res = await api.del(`/admin/api/config/models/${encodeURIComponent(name)}`);
  if (res.ok) { showToast(t('toast.modelDeleted',{name})); window.loadConfig(); }
  else { showToast(res.error || 'Failed', 'error'); }
}

function deleteModel(name, btn) {
  if (!btn) { _doDeleteModel(name); return; }
  inlineConfirm(btn, () => _doDeleteModel(name));
}

async function toggleModel(name) {
  const res = await api.post(`/admin/api/config/models/${encodeURIComponent(name)}/toggle`);
  if (res.ok) {
    const stateKey = res.enabled ? 'model.enabled' : 'model.disabled';
    showToast(`${name}: ${t(stateKey)}`);
    window.loadConfig();
  } else { showToast(res.error || 'Failed', 'error'); }
}

function editModel(name, provider) {
  const info = S.configData.models[name];
  const caps = (typeof info === 'object' && info.capabilities) ? info.capabilities : ['text'];
  const upstream = (typeof info === 'object' && info.upstream_model) ? info.upstream_model : '';
  openModelModal(name, provider, caps, upstream);
}

function cloneModel(name) {
  // Open the model modal pre-filled from an existing model, with a blank
  // name so the user picks a new one. Mirrors copyProviderEntry's behavior.
  const models = S.configData.models || {};
  const info = models[name];
  const provider = typeof info === 'string' ? info : (info.provider || '');
  const caps = typeof info === 'string' ? ['text'] : (info.capabilities || ['text']);
  const upstream = (typeof info === 'object' && info.upstream_model) ? info.upstream_model : '';
  // name blank ('') → modal opens in "add" mode; sourceModel feeds reasoning config
  openModelModal('', provider, caps, upstream, name);
}

// ── Bulk operations ──

function selectAllModels(headerCb) {
  document.querySelectorAll('#modelTable .row-check').forEach(cb => cb.checked = headerCb.checked);
  updateModelBulk();
}

function updateModelBulk() {
  const checked = document.querySelectorAll('#modelTable .row-check:checked');
  const bar = document.getElementById('modelBulkBar');
  document.getElementById('modelBulkCount').textContent = checked.length;
  bar.style.display = checked.length > 0 ? 'flex' : 'none';
}

async function bulkModels(action) {
  const checked = document.querySelectorAll('#modelTable .row-check:checked');
  const names = [...checked].map(cb => cb.dataset.model);
  if (!names.length) return;
  if (action === 'delete' && !confirm(t('confirm.bulkDelete', {count: names.length}))) return;
  const res = await api.post('/admin/api/config/models/bulk', {action, models: names});
  if (res.ok) {
    showToast(t('toast.bulkDone', {action, count: res.affected.length}));
    window.loadConfig();
  } else { showToast(res.error || 'Failed', 'error'); }
}

function toggleMoreMenu(btn) {
  const menu = btn.nextElementSibling;
  document.querySelectorAll('.more-menu').forEach(m => { if (m !== menu) m.style.display = 'none'; });
  menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
}

// ── Register on window for cross-module and inline-handler access ──

Object.assign(window, {
  switchModelTab, segModelType, toggleCap, toggleStreamUrl,
  openModelModal, onModelTypeChange, updateReasoningVisibility,
  switchModelDomain, resetModelFilters, sortModels,
  renderModels, saveModel, toggleModel, editModel, cloneModel,
  selectAllModels, updateModelBulk, bulkModels, toggleMoreMenu,
  deleteModel, goToModelsForProvider, goToProviderFromModel,
});

export { renderModels };
