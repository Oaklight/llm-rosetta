// models.js — Model management: modal, CRUD, rendering, filtering, bulk ops
import { S, _CAP_ICONS } from '../core/state.js';
import { t, capLabel } from '../core/i18n.js';
import { api, showToast, closeModal, esc, inlineConfirm, closeMenus, openMenu, isFreeEntry, freeEntryLabel, providerTypeName } from '../core/utils.js';
import { syncAriaSort } from '../core/sort.js';
import { syncChipState } from '../core/chips.js';
import { initLogoSelect, setLogoSelectOptions } from '../components/logo-select.js';
import { mergeReasoningOverride } from '../core/reasoning-override.js';
import { providerRowsHtml, syncPickerBtn, openPickerMenu } from '../components/provider-picker.js';

// ── helpers (module-private) ──

let _loadedReasoningOverride = {};

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
  // Server-provided type field takes precedence
  if (info.type) return info.type;
  // Capability-based fallback for legacy entries
  const caps = info.capabilities || ['text'];
  if (caps.includes('embedding') && !caps.includes('text')) return 'embedding';
  if (caps.includes('rerank') && !caps.includes('text')) return 'rerank';
  return 'llm';
}

/** Look up the badge CSS class for a model type from registry metadata. */
function _typeBadgeClass(typeName) {
  const types = (S.configData && S.configData.model_types) || [];
  const desc = types.find(t => t.name === typeName);
  if (desc && desc.badge_class) return desc.badge_class;
  // Fallback for types not yet in metadata
  return 'cap-badge-' + typeName;
}

/** Return the list of registered model type names from server metadata. */
function _registeredTypeNames() {
  const types = (S.configData && S.configData.model_types) || [];
  return types.map(t => t.name);
}

/**
 * Build the filter seg-control and modal seg-control from model_types
 * metadata.  Called once after config loads and again on every model render.
 *
 * A rebuild replaces the option nodes, so it is only done when the option set
 * actually changed — a keyboard user arrowing through the group would
 * otherwise lose focus to <body> on every step.  The state-only path also
 * keeps the modal's selection intact when a render happens while it is open.
 */
function _buildTypeSegControls() {
  const typeNames = _registeredTypeNames();
  if (!typeNames.length) return;
  _syncFilterTypeSeg(document.getElementById('modelTypeSeg'), typeNames);
  _syncModalTypeSeg(document.getElementById('modelModalTypeSeg'), typeNames);
}

function _syncFilterTypeSeg(seg, typeNames) {
  if (!seg) return;
  const domains = ['all', ...typeNames];
  const items = Array.from(seg.querySelectorAll(':scope > *'));
  const sameSet = items.length === domains.length &&
    items.every((el, i) => el.dataset.domain === domains[i]);
  if (sameSet) {
    items.forEach((el, i) => {
      const domain = domains[i];
      const active = S._modelDomain === domain;
      el.classList.toggle('active', active);
      el.setAttribute('aria-checked', active ? 'true' : 'false');
      el.setAttribute('tabindex', active ? '0' : '-1');
      const span = el.querySelector('span');
      const labelKey = domain === 'all' ? 'filter.all' : 'label.' + domain;
      if (span) span.textContent = t(labelKey) || (domain === 'all' ? 'All' : domain.toUpperCase());
    });
    return;
  }
  seg.innerHTML = domains.map((domain) => {
    const active = S._modelDomain === domain;
    const labelKey = domain === 'all' ? 'filter.all' : 'label.' + domain;
    const label = t(labelKey) || (domain === 'all' ? 'All' : domain.toUpperCase());
    return `<div class="${active ? 'active' : ''}" role="radio" data-domain="${esc(domain)}" aria-checked="${active}" tabindex="${active ? '0' : '-1'}" onclick="switchModelDomain(this,'${esc(domain)}')"><span data-i18n="${labelKey}">${esc(label)}</span></div>`;
  }).join('');
}

/** The status facet: how much of the fleet can take traffic right now.
 *
 *  Counts are taken over the slice the other filters left, not the whole
 *  table, so the numbers answer "how many of the rows I am looking at".
 */
function _buildStatusSeg(counts) {
  const seg = document.getElementById('modelStatusSeg');
  if (!seg) return;
  const options = [
    ['all', 'modelFilter.all'],
    ['usable', 'modelFilter.usable'],
    ['partial', 'modelFilter.partial'],
    ['off', 'modelFilter.unusable'],
  ];
  seg.innerHTML = options.map(([value, key]) => {
    const active = (S._modelStatus || 'all') === value;
    return `<div class="${active ? 'active' : ''}" role="radio" data-status="${value}"`
      + ` aria-checked="${active}" tabindex="${active ? '0' : '-1'}"`
      + ` onclick="setModelStatus('${value}')"`
      + ` onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();setModelStatus('${value}')}">`
      + `<span data-i18n="${key}">${esc(t(key))}</span><span class="seg-count">${counts[value] ?? 0}</span></div>`;
  }).join('');
}

function setModelStatus(value) {
  S._modelStatus = value;
  renderModels();
}
window.setModelStatus = setModelStatus;

/** 'usable' (every provider live), 'partial' (some provider entry off) or
 *  'off' (nothing can serve it). */
function _modelStatus(info, disabledProviders) {
  if (!_entryUsable(info, disabledProviders)) return 'off';
  const details = _getProviderDetails(info);
  const live = details.filter(p => !_providerEntryOff(p, disabledProviders)).length;
  return live < details.length ? 'partial' : 'usable';
}

function _syncModalTypeSeg(seg, typeNames) {
  if (!seg) return;
  const items = Array.from(seg.querySelectorAll(':scope > *'));
  const sameSet = items.length === typeNames.length &&
    items.every((el, i) => el.querySelector('input')?.value === typeNames[i]);
  if (sameSet) {
    // The radios own the selection (openModelModal sets it), so only refresh
    // the labels, which is what a language switch needs.
    items.forEach((el, i) => {
      const span = el.querySelector('span');
      const labelKey = 'label.' + typeNames[i];
      if (span) span.textContent = t(labelKey) || typeNames[i].toUpperCase();
    });
    return;
  }
  seg.innerHTML = typeNames.map((name, i) => {
    const labelKey = 'label.' + name;
    return `<div class="${i === 0 ? 'active' : ''}" onclick="segModelType(this,'${esc(name)}')"><input type="radio" name="modelType" value="${esc(name)}"${i === 0 ? ' checked' : ''} onchange="segModelType(this)"><span data-i18n="${labelKey}">${esc(t(labelKey) || name.toUpperCase())}</span></div>`;
  }).join('');
}

// ── Model modal ──

function switchModelTab(clicked, showId) {
  // The form lives in the side panel now, which is not a .modal.
  const modal = clicked.closest('.modal') || clicked.closest('.model-form')
    || document.getElementById('modelModal');
  modal.querySelectorAll('.model-tab-content').forEach(el => el.classList.remove('active'));
  modal.querySelector('#' + showId).classList.add('active');
  clicked.parentElement.querySelectorAll('.model-tab').forEach(t => t.classList.remove('active'));
  clicked.classList.add('active');
}

// Called from the option div's click and from the radio's change event: the
// change path is what a keyboard arrow or Space produces, and it must not be
// mistaken for a second activation.  Resolve the option box first, then make
// the checked radio and the styled class agree.
function segModelType(el, value) {
  const box = el.closest('.seg-control > *') || el;
  box.parentElement.querySelectorAll(':scope > *').forEach(l => l.classList.remove('active'));
  box.classList.add('active');
  const radio = box.querySelector('input');
  if (radio) radio.checked = true;
  onModelTypeChange();
}

// The chip is a <label>, so a click anywhere on it (or Space on the focused
// checkbox) toggles the input natively before this runs.  Reading the input's
// state instead of flipping it is what makes the keyboard path work: flipping
// here cancelled the browser's own toggle and left the chip unchanged.
function syncCapChip(el, checkboxId) {
  const cb = document.getElementById(checkboxId);
  syncChipState(el, cb);
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

/** Show the chosen provider on the form's picker trigger. */
function _syncModelProviderBtn(name) {
  syncPickerBtn(document.getElementById('modelProviderBtn'), name, t('label.selectOne'));
}

/** Pick the provider from the form's roster menu. */
function selectModelProvider(name) {
  document.getElementById('modelProvider').value = name;
  _syncModelProviderBtn(name);
  closeMenus();
}
window.selectModelProvider = selectModelProvider;

function toggleModelProviderPicker(btn) {
  const menu = document.getElementById('modelProviderMenu');
  if (menu.classList.contains('open')) { closeMenus(); return; }
  openPickerMenu(menu, btn, openMenu, closeMenus);
}
window.toggleModelProviderPicker = toggleModelProviderPicker;

function openModelModal(model, provider, capabilities, upstreamModel, sourceModel) {
  // sourceModel: when cloning, the original model name to read reasoning from
  // (the visible name field is left blank so the user picks a new name).
  document.getElementById('modelModalTitle').textContent = model ? t('modal.editModel') : t('modal.addModel');
  document.getElementById('modelName').value = model || '';
  document.getElementById('modelName').dataset.originalName = model || '';
  document.getElementById('modelUpstream').value = upstreamModel || '';
  // Populate the provider picker from config.  Every provider is listed, the
  // disabled ones dimmed: a model may still point at one.
  const providers = S.configData ? Object.entries(S.configData.providers) : [];
  const muted = new Set(
    providers.filter(([, cfg]) => cfg.enabled === false).map(([name]) => name),
  );
  document.getElementById('modelProviderMenu').innerHTML = providerRowsHtml(
    providers.map(([name]) => name),
    n => `selectModelProvider('${esc(n)}')`,
    muted,
  );
  document.getElementById('modelProvider').value = provider || '';
  _syncModelProviderBtn(provider);
  // Multi-provider indicator
  const multiHint = document.getElementById('modelMultiProviderHint');
  if (multiHint) {
    const modelInfo = model && S.configData ? S.configData.models[model] : null;
    const isMulti = modelInfo && typeof modelInfo === 'object' && modelInfo.providers && modelInfo.providers.length > 1;
    if (isMulti) {
      multiHint.textContent = t('model.multipleProvidersHint', {n: modelInfo.providers.length});
      multiHint.style.display = '';
    } else {
      multiHint.style.display = 'none';
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

  // Set model type via seg-control (data-driven: find the matching radio)
  const caps = capabilities || ['text', 'tools'];
  const detectedType = _getModelType({capabilities: caps, type: null});
  const segLabels = document.getElementById('modelModalTypeSeg').querySelectorAll(':scope > div');
  segLabels.forEach(l => l.classList.remove('active'));
  // Find the seg-control option matching detectedType
  const matchingRadio = document.querySelector('input[name="modelType"][value="' + detectedType + '"]');
  if (matchingRadio) {
    matchingRadio.checked = true;
    matchingRadio.closest('div').classList.add('active');
  } else if (segLabels.length) {
    // Fallback to first option (LLM)
    segLabels[0].classList.add('active');
    const firstRadio = segLabels[0].querySelector('input');
    if (firstRadio) firstRadio.checked = true;
  }
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
  // What to merge the form's edits onto when saving: the stored override, which
  // may carry fields these four controls cannot express.
  _loadedReasoningOverride = configOverride;
  const srcBadge = document.getElementById('reasoningSourceBadge');
  srcBadge.textContent = reasoning.source || 'none';
  srcBadge.className = 'badge badge-sm' + (reasoning.source === 'config' ? ' badge-cap-tools' : reasoning.source === 'model_override' ? ' badge-cap-reasoning' : '');
  // Update inherit option labels: show "value (inherited)" or just "(inherit)"
  const inh = t('label.inherited');
  const ttInherit = document.querySelector('#reasoningThinkingType option[value=""]');
  const tmSummary = reasoning.thinking_modes ? Object.keys(reasoning.thinking_modes).join('/') : null;
  ttInherit.textContent = tmSummary ? tmSummary + ' (' + inh + ')' : '(' + inh + ')';
  const br = document.getElementById('reasoningBudgetRatio');
  br.placeholder = reasoning.budget_ratio != null ? reasoning.budget_ratio + ' (' + inh + ')' : '(' + inh + ')';
  const dsInherit = document.querySelector('#reasoningDisabled option[value=""]');
  const hasDisabled = reasoning.thinking_modes && reasoning.thinking_modes.disabled;
  dsInherit.textContent = hasDisabled ? 'thinking_disabled (' + inh + ')' : 'omit (' + inh + ')';
  // Set values — config override if present, otherwise blank (inheriting)
  document.getElementById('reasoningThinkingType').value = configOverride.thinking_type || '';
  document.getElementById('reasoningBudgetRatio').value = configOverride.budget_ratio != null ? configOverride.budget_ratio : '';
  document.getElementById('reasoningDisabled').value = configOverride.disabled || '';
  document.getElementById('reasoningReplayMissing').value = configOverride.replay_missing || '';
  document.getElementById('reasoningEffortField').value = configOverride.effort_field || '';
  const range = Array.isArray(configOverride.effort_range) ? configOverride.effort_range : [];
  document.getElementById('reasoningEffortMin').value = range[0] || '';
  document.getElementById('reasoningEffortMax').value = range[1] || '';
  document.getElementById('reasoningUnsignedBlocks').value = configOverride.unsigned_blocks || '';
  const vis = configOverride.visibility_modes || {};
  for (const mode of ['Auto', 'Concise', 'Detailed', 'None']) {
    document.getElementById('reasoningVis' + mode).value = vis[mode.toLowerCase()] || '';
  }
  _renderReasoningEffective(reasoning, configOverride);

  _setDrawerPane('form');
  _openDrawer();
}

/**
 * Show the resolved capability fields the form does not edit.
 *
 * The override the operator saves is merged onto the shim's capability, so the
 * values that actually reach the request come from two places at once.  Reading
 * them here answers "why did this model send `high`?" without opening the shim
 * YAML.  A field overridden on this model is marked, and a field the shim never
 * declares reads "—" rather than a plausible-looking default.
 */
function _renderReasoningEffective(reasoning, override) {
  const box = document.getElementById('reasoningEffective');
  if (!box) return;
  const fmt = (v) => {
    if (v == null || v === '') return '—';
    if (Array.isArray(v)) return v.length ? v.join(' / ') : '—';
    if (typeof v === 'object') {
      const parts = Object.entries(v).filter(([, x]) => x != null).map(([k, x]) => `${k}: ${x}`);
      return parts.length ? parts.join(' · ') : '—';
    }
    return String(v);
  };
  const rows = [
    ['effortField', 'effort_field', override.effort_field],
    ['effortRange', 'effort_range', override.effort_range],
    ['visibilityModes', 'visibility_modes', override.visibility_modes],
    ['unsignedBlocks', 'unsigned_blocks', override.unsigned_blocks],
    ['replayMissing', 'replay_missing', override.replay_missing],
  ];
  box.innerHTML = rows.map(([key, field, overridden]) => {
    const value = fmt(overridden != null ? overridden : reasoning[field]);
    const mark = overridden != null ? ' · ' + t('label.overriddenHere', 'overridden here') : '';
    return `<div><span style="min-width:140px;display:inline-block">${esc(t('label.' + key))}</span>` +
      `<code style="${overridden != null ? 'color:var(--accent)' : ''}">${esc(value + mark)}</code></div>`;
  }).join('');
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
  S._modelStatus = 'all';
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

// The narrow-screen layout hides the sortable header, so the toolbar offers
// the same three keys as a select plus a direction toggle.
function setModelSortKey(key) {
  S._modelSortKey = key;
  renderModels();
}

function toggleModelSortDir() {
  S._modelSortDir = S._modelSortDir === 'asc' ? 'desc' : 'asc';
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

// ── Test menu helpers ──

/** Return the default test kind for the primary test button. */
function _defaultTestKind(modelType) {
  if (modelType === 'embedding') return 'embedding';
  if (modelType === 'rerank') return 'rerank';
  if (modelType === 'llm') return 'text';
  // Unknown types: use the type name as the test kind
  return modelType;
}

/** Build test menu items HTML for a given model type. */
function _testMenuItems(modelType, modelName, {hasTools, hasVision, hasReasoning}, provider) {
  const n = esc(modelName);
  // Fourth argument of runTest: pins the request to one provider.  Omitted on
  // the row's own test menu, where routing picks the provider.
  const pArg = provider ? `,null,'${esc(provider)}'` : '';
  if (modelType === 'embedding') {
    return `
      <div class="test-menu-item" onclick="runTest('${n}','embedding'${pArg})">${t('test.embedding')}</div>
      <div class="test-menu-item" onclick="runTest('${n}','embed_batch'${pArg})">${t('test.embedBatch')}</div>
      <div class="test-menu-item" onclick="promptMatryoshka('${n}')">${t('test.matryoshka')}</div>
      <div class="test-menu-item" onclick="runTest('${n}','embed_multimodal'${pArg})">${t('test.embedMultimodal')}</div>`;
  }
  if (modelType === 'rerank') {
    return `
      <div class="test-menu-item" onclick="runTest('${n}','rerank'${pArg})">${t('test.rerank')}</div>
      <div class="test-menu-item" onclick="runTest('${n}','rerank_batch'${pArg})">${t('test.rerankBatch')}</div>`;
  }
  if (modelType === 'llm') {
    return `
      <div class="test-menu-item" onclick="runTest('${n}','text'${pArg})">${t('test.text')}</div>
      <div class="test-menu-item" onclick="runTest('${n}','stream'${pArg})">${t('test.stream')}</div>
      <div class="test-menu-item${hasTools ? '' : ' disabled'}" onclick="${hasTools ? `runTest('${n}','tools'${pArg})` : ''}">${t('test.tools')}</div>
      <div class="test-menu-item${hasVision ? '' : ' disabled'}" onclick="${hasVision ? `runTest('${n}','vision'${pArg})` : ''}">${t('test.vision')}</div>
      <div class="test-menu-item${hasReasoning ? '' : ' disabled'}" onclick="${hasReasoning ? `runTest('${n}','reasoning'${pArg})` : ''}">${t('test.reasoning')}</div>`;
  }
  // Fallback for unknown/new types: single test item using type name
  return `
    <div class="test-menu-item" onclick="runTest('${n}','${esc(modelType)}'${pArg})">${t('btn.test')} ${esc(modelType)}</div>`;
}

// ── Multi-provider helpers ──

function _getAllProviders(info) {
  if (typeof info === 'string') return [info];
  if (info.providers && Array.isArray(info.providers)) {
    return info.providers.map(p => typeof p === 'string' ? p : p.name || '');
  }
  return [info.provider || ''];
}

function _getProviderDetails(info) {
  if (typeof info === 'string') return [{name: info, enabled: true}];
  if (info.providers && Array.isArray(info.providers)) {
    return info.providers.map(p => {
      if (typeof p === 'string') return {name: p, enabled: true};
      return {
        name: p.name || '',
        weight: p.weight,
        upstream_model: p.upstream_model,
        enabled: p.enabled !== false,
      };
    });
  }
  return [{name: info.provider || '', upstream_model: info.upstream_model, enabled: true}];
}

/** Label for a provider cell — "<Brand> (Free)" for a free entry, else the name.
 *  The free-entry rule is shared with providers.js via core/utils.js. */
function _providerLabel(providerName) {
  const providers = S.configData?.providers;
  const shims = S.configData?.registered_shims;
  if (!isFreeEntry(providers, shims, providerName)) return providerName;
  return freeEntryLabel(providers, shims, providerName, t('free.suffix'));
}

/** True when this provider entry can take traffic right now.
 *
 *  Two switches can stop it: the provider itself, in the providers section,
 *  and this model's own entry for it. */
function _providerEntryOff(detail, disabledProviders) {
  return detail.enabled === false || disabledProviders.has(detail.name);
}

/** True when at least one of the model's providers can serve it right now. */
function _entryUsable(info, disabledProviders) {
  if (typeof info === 'object' && info && info.enabled === false) return false;
  return _getProviderDetails(info).some(
    p => !_providerEntryOff(p, disabledProviders),
  );
}

function _renderProviderCell(name, info, disabledProviders) {
  const details = _getProviderDetails(info);
  if (details.length <= 1) {
    const p = details[0] || {name: ''};
    // No "(Disabled)" word: the dimmed cell and the switch already say it, and
    // the word is what squeezes a long provider name into the switch.
    return `<span class="provider-link" title="${esc(p.name)}" onclick="goToProviderFromModel('${esc(p.name)}')">${esc(_providerLabel(p.name))}</span>`;
  }
  const live = details.filter(p => !_providerEntryOff(p, disabledProviders)).length;
  return `<button class="btn btn-sm provider-multi-btn" aria-haspopup="true" aria-expanded="false" onclick="toggleProviderMenu(this,'${esc(name)}')">
      <span>${t('model.multipleProviders')}</span><span class="provider-multi-count">${details.length}</span>${live < details.length ? `<span class="provider-multi-off">${details.length - live} ${t('label.off')}</span>` : ''}<span aria-hidden="true">&#9662;</span>
    </button>
    <div class="more-menu provider-menu" role="menu">${_providerMenuRows(name, info, disabledProviders)}</div>`;
}

/** One menu row per provider: its own on/off switch, weight, test and remove. */
function _providerMenuRows(name, info, disabledProviders) {
  const details = _getProviderDetails(info);
  const modelType = _getModelType(info);
  const kind = _defaultTestKind(modelType);
  const caps = typeof info === 'string' ? ['text'] : (info.capabilities || ['text']);
  const opts = {
    hasTools: caps.includes('tools'),
    hasVision: caps.includes('vision'),
    hasReasoning: caps.includes('reasoning'),
  };
  return details.map(p => {
    const off = _providerEntryOff(p, disabledProviders);
    const n = esc(name);
    const pn = esc(p.name);
    const w = p.weight || 1;
    return `<div class="provider-menu-row${off ? ' pm-off' : ''}" role="menuitem">
      <label class="toggle row-toggle" title="${off ? t('provider.disabled') : t('provider.enabled')}"><input type="checkbox" role="switch" ${off ? '' : 'checked'} aria-checked="${!off}" aria-label="${esc(p.name)}" onchange="toggleModelProvider('${n}','${pn}')"><span class="slider"></span></label>
      <div class="pm-id">
        <span class="provider-link pm-name" title="${esc(p.name)}" onclick="goToProviderFromModel('${pn}')">${esc(_providerLabel(p.name))}</span>
        <span class="pm-upstream">${p.upstream_model ? `→ ${esc(p.upstream_model)}` : ''}</span>
      </div>
      <input type="number" class="pd-weight" min="1" max="100" value="${w}" title="${t('label.weight')}" onchange="updateProviderWeight('${n}','${pn}',this)" onkeydown="if(event.key==='Enter'){this.blur()}">
      <div class="test-group"><button class="btn btn-sm btn-test" onclick="runTest('${n}','${kind}',null,'${pn}')">${t('btn.test')}</button><button class="btn btn-sm btn-caret" aria-haspopup="true" aria-expanded="false" aria-label="${t('btn.test')} ${pn}" onclick="toggleTestMenu(this)">&#9662;</button><div class="test-menu">${_testMenuItems(modelType, name, opts, p.name)}</div></div>
      <button class="btn btn-sm pd-remove" title="${t('btn.remove')}" onclick="removeProviderFromModel('${n}','${pn}')">✕</button>
    </div>`;
  }).join('');
}

/** The model shown in the side panel, and which of its two panes is up. */
let _drawerModel = null;
let _drawerPane = 'detail';

/** Where focus was before the panel took it, so closing puts it back. */
let _focusBeforeDrawer = null;

function _openDrawer() {
  const el = document.getElementById('modelDrawer');
  if (!el.classList.contains('open')) _focusBeforeDrawer = document.activeElement;
  el.classList.add('open');
  el.setAttribute('aria-hidden', 'false');
  document.body.classList.add('model-drawer-open');
  // The panel is the new context: focus it so Escape and the close button work
  // without a click, without stealing focus on every re-render.
  if (document.activeElement !== el && !el.contains(document.activeElement)) {
    el.querySelector('.md-close')?.focus();
  }
  _markActiveModelRow();
}

/** The row the panel is about, so the table and the panel read as one thing. */
function _markActiveModelRow() {
  document.querySelectorAll('#modelTable tr.model-row-active')
    .forEach(r => r.classList.remove('model-row-active'));
  if (!_drawerModel) return;
  const check = document.querySelector(
    `#modelTable .row-check[data-model="${CSS.escape(_drawerModel)}"]`,
  );
  check?.closest('tr')?.classList.add('model-row-active');
}

/** Swap the panel between the summary and the edit form. */
function _setDrawerPane(which) {
  _drawerPane = which;
  document.getElementById('drawerDetail').hidden = which !== 'detail';
  document.getElementById('modelModal').hidden = which !== 'form';
  document.getElementById('drawerTitle').textContent = which === 'detail'
    ? (_drawerModel || '')
    : document.getElementById('modelModalTitle').textContent;
  // The panel's own switch, hidden while adding (there is no summary yet).
  const tabs = document.getElementById('drawerTabs');
  if (tabs) {
    tabs.hidden = !_drawerModel;
    tabs.querySelectorAll('.md-tab').forEach((b, i) => {
      const on = (i === 0) === (which === 'detail');
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', String(on));
    });
  }
}

/** Open the form for the model the panel is showing, from the panel itself. */
function openModelEditor() {
  if (!_drawerModel) return;
  if (_drawerPane !== 'form') editModel(_drawerModel, '');
  else _setDrawerPane('form');
}
window.openModelEditor = openModelEditor;

/** Switch the panel's pane from its tab strip. */
function setModelDrawerPane(which) {
  if (which === 'form') { openModelEditor(); return; }
  _setDrawerPane('detail');
  _syncModelDrawer();
}
window.setModelDrawerPane = setModelDrawerPane;

function openModelDrawer(name) {
  _drawerModel = name;
  _setDrawerPane('detail');
  _syncModelDrawer();
  _openDrawer();
}
window.openModelDrawer = openModelDrawer;

function closeModelDrawer() {
  const el = document.getElementById('modelDrawer');
  el.classList.remove('open');
  el.setAttribute('aria-hidden', 'true');
  document.body.classList.remove('model-drawer-open');
  document.querySelectorAll('#modelTable tr.model-row-active')
    .forEach(r => r.classList.remove('model-row-active'));
  _drawerModel = null;
  _drawerPane = 'detail';
  if (_focusBeforeDrawer && document.contains(_focusBeforeDrawer)) {
    _focusBeforeDrawer.focus();
  }
  _focusBeforeDrawer = null;
}
window.closeModelDrawer = closeModelDrawer;

/** Provider rows: the dropdown's rows for a multi entry, a plain one otherwise.
 *
 *  A single-provider entry has no providers[] to switch or remove, so it gets
 *  the name, its upstream and a test button and nothing that would fail. */
function _detailProviderRows(name, info, disabledProviders) {
  if (Array.isArray(info && info.providers) && info.providers.length > 1) {
    return _providerMenuRows(name, info, disabledProviders);
  }
  const [p] = _getProviderDetails(info);
  if (!p) return '';
  const kind = _defaultTestKind(_getModelType(info));
  const off = _providerEntryOff(p, disabledProviders);
  return `<div class="provider-menu-row pd-single${off ? ' pm-off' : ''}">
      <span class="pm-id">
        <span class="provider-link pm-name" title="${esc(p.name)}" onclick="goToProviderFromModel('${esc(p.name)}')">${esc(_providerLabel(p.name))}</span>
        <span class="pm-upstream">${p.upstream_model ? `→ ${esc(p.upstream_model)}` : ''}</span>
      </span>
      <div class="test-group"><button class="btn btn-sm btn-test" onclick="runTest('${esc(name)}','${kind}',null,'${esc(p.name)}')">${t('btn.test')}</button></div>
    </div>`;
}

/** The detail block for one model. */
function _modelDetailHtml(name, info, disabledProviders) {
  const caps = typeof info === 'string' ? ['text'] : (info.capabilities || ['text']);
  const capList = caps.map(c =>
    `<span class="cap-badge cap-badge-${c}" title="${esc(capLabel(c))}">${_capIcons[c] || ''}<span>${esc(capLabel(c))}</span></span>`,
  ).join('');
  return `<div class="model-detail">
      <div class="md-row"><span class="md-label">${t('label.capabilities')}</span><span class="md-caps">${capList}</span></div>
      <div class="md-providers">${_detailProviderRows(name, info, disabledProviders)}</div>
      <div class="md-actions">
        <button class="btn btn-sm" onclick="editModel('${esc(name)}','${esc(_getProviderDetails(info)[0]?.name || '')}')">${t('btn.edit')}</button>
        <button class="btn btn-sm" onclick="cloneModel('${esc(name)}')">${t('btn.clone')}</button>
        <button class="btn btn-sm btn-danger" onclick="deleteModel('${esc(name)}', this)">${t('btn.delete')}</button>
      </div>
    </div>`;
}

/** Repaint the summary pane; a form in progress is left alone. */
function _syncModelDrawer() {
  if (_drawerPane !== 'detail') return;
  const el = document.getElementById('drawerDetail');
  if (!el) return;
  document.getElementById('drawerTitle').textContent = _drawerModel || '';
  const info = _drawerModel ? S.configData?.models?.[_drawerModel] : null;
  if (!info) {
    el.innerHTML = '';
    return;
  }
  const providers = S.configData.providers || {};
  const disabledProviders = new Set(
    Object.entries(providers).filter(([, cfg]) => cfg.enabled === false).map(([n]) => n),
  );
  el.innerHTML = _modelDetailHtml(_drawerModel, info, disabledProviders);
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && document.getElementById('modelDrawer')?.classList.contains('open')) {
    closeModelDrawer();
  }
});

// A click outside the panel closes it — except on another row, which switches
// the panel to that model instead of closing it.
document.addEventListener('click', (e) => {
  const el = document.getElementById('modelDrawer');
  if (!el || !el.classList.contains('open')) return;
  if (e.target.closest('#modelDrawer')) return;
  if (e.target.closest('#modelTable tr')) return;
  if (e.target.closest('.modal-overlay.open')) return;
  // A control that opens the panel is, by definition, outside it.
  if (e.target.closest('.drawer-trigger')) return;
  closeModelDrawer();
});

/** The model whose provider menu is open, so a re-render can reopen it.
 *
 *  Acting inside the menu — a weight, a switch, a removal — reloads the config
 *  and rebuilds the table.  Without this the menu closes under the pointer
 *  after the first change, and the operator cannot make a second one. */
let _openProviderMenuFor = null;

function toggleProviderMenu(btn, modelName) {
  const menu = btn.nextElementSibling;
  const wasOpen = menu.classList.contains('open');
  closeMenus();
  if (!wasOpen) {
    _openProviderMenuFor = modelName;
    openMenu(menu, btn);
  } else {
    _openProviderMenuFor = null;
  }
}

/** Reopen the provider menu after the table was rebuilt. */
function _restoreProviderMenu() {
  if (!_openProviderMenuFor) return;
  const check = document.querySelector(
    `#modelTable .row-check[data-model="${CSS.escape(_openProviderMenuFor)}"]`,
  );
  const btn = check && check.closest('tr').querySelector('.provider-multi-btn');
  if (!btn) {
    _openProviderMenuFor = null;
    return;
  }
  openMenu(btn.nextElementSibling, btn);
}

// A click outside the menu and its button closes it for good: the next render
// must not bring it back.
document.addEventListener('click', (e) => {
  if (e.target.closest('.provider-menu') || e.target.closest('.provider-multi-btn')) {
    return;
  }
  _openProviderMenuFor = null;
});
window.toggleProviderMenu = toggleProviderMenu;

/** Turn one provider's entry for a model on or off.
 *
 *  This is not the same as disabling the provider: it narrows one model to
 *  its other providers. */
async function toggleModelProvider(modelName, providerName) {
  const info = S.configData?.models?.[modelName];
  if (!info) return;
  const detail = _getProviderDetails(info).find(p => p.name === providerName);
  if (!detail) return;
  const next = detail.enabled === false;
  const res = await api.post(
    `/admin/api/config/models/${encodeURIComponent(modelName)}/update-provider`,
    {provider: providerName, enabled: next},
  );
  if (res.ok) {
    showToast(`${providerName}: ${t(next ? 'model.enabled' : 'model.disabled')}`);
    window.loadConfig();
  } else {
    showToast(res.error || t('toast.error'), 'error');
  }
}
window.toggleModelProvider = toggleModelProvider;


async function removeProviderFromModel(modelName, providerName) {
  if (!confirm(t('confirm.removeProviderFromModel', {provider: providerName, model: modelName}))) return;
  const res = await api.post(`/admin/api/config/models/${encodeURIComponent(modelName)}/remove-provider`, {provider: providerName});
  if (res.ok) {
    showToast(t('toast.providerRemovedFromModel', {provider: providerName, model: modelName}));
    window.loadConfig();
  } else {
    showToast(res.error || t('toast.error'), 'error');
  }
}

/** The mutable `providers[]` entry for one provider, or null when the model
 *  is a single-provider entry. */
function _providerEntry(info, providerName) {
  if (!info || typeof info !== 'object' || !Array.isArray(info.providers)) return null;
  return info.providers.find(
    p => (typeof p === 'string' ? p : p.name) === providerName,
  ) || null;
}

/** Save one provider's weight.
 *
 *  The weight shows nowhere but this menu, so the answer is written back into
 *  the loaded config instead of reloading it: a reload rebuilds the table and
 *  the menu would close under the pointer after every step. */
async function updateProviderWeight(modelName, providerName, input) {
  const detail = _getProviderDetails(S.configData.models[modelName])
    .find(p => p.name === providerName);
  const previous = detail ? (detail.weight || 1) : 1;
  const w = parseInt(input.value, 10);
  if (isNaN(w) || w < 1 || w > 100) {
    showToast(t('toast.weightRange'), 'error');
    input.value = previous;
    return;
  }
  const res = await api.post(
    `/admin/api/config/models/${encodeURIComponent(modelName)}/update-provider`,
    {provider: providerName, weight: w},
  );
  if (!res.ok) {
    showToast(res.error || t('toast.error'), 'error');
    input.value = previous;
    return;
  }
  const entry = _providerEntry(S.configData.models[modelName], providerName);
  if (entry && typeof entry === 'object') entry.weight = w;
  showToast(t('toast.weightSet', {provider: providerName, w}));
}

// ── Model rendering ──

/** The mark a provider card shows, resolved the same way (config > shim). */
function _providerLogo(name) {
  const cfg = (S.configData.providers || {})[name] || {};
  const shims = S.configData.registered_shims || [];
  const shimLogo = Object.fromEntries(shims.map(s => [s.name, s.logo]));
  return cfg.logo || shimLogo[name] || shimLogo[providerTypeName(cfg, name)] || '';
}

const _capIcons = {
  text: '<svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7V4h16v3"/><path d="M12 4v16"/><path d="M8 20h8"/></svg>',
  vision: '<svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>',
  tools: '<svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/></svg>',
  reasoning: '<svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18h6"/><path d="M10 22h4"/><path d="M12 2a7 7 0 00-4 12.7V17h8v-2.3A7 7 0 0012 2z"/></svg>',
};

function _checkRoutingLoops() {
  api.get('/admin/api/config/models/loop-check')
    .then(data => {
      const banner = document.getElementById('loopWarningBanner');
      if (!banner) return;
      if (data.warnings && data.warnings.length) {
        banner.innerHTML = `<strong>${t('error.routingLoop')}</strong> ` +
          data.warnings.map(w => esc(w)).join('<br>');
        banner.style.display = 'block';
      } else {
        banner.style.display = 'none';
      }
    })
    .catch(() => {});
}

function renderModels() {
  // Rebuild type seg-controls from server metadata on each render
  // (cheap DOM update, ensures newly registered types appear immediately)
  _buildTypeSegControls();

  _checkRoutingLoops();

  const tbody = document.getElementById('modelTable');
  const models = S.configData.models || {};
  const totalCount = Object.keys(models).length;

  // Populate provider filter (preserve selection).  A native <select> cannot
  // render the provider logo, so this is a custom listbox over a hidden input.
  initLogoSelect('modelProviderFilter', () => renderModels());
  const provFilter = document.getElementById('modelProviderFilter');
  const prevProv = provFilter.value;
  const provSet = new Set();
  for (const [, info] of Object.entries(models)) {
    _getAllProviders(info).forEach(p => provSet.add(p));
  }
  // Each provider carries how many models it serves, the same count its card
  // shows, so the roster can be read without opening the providers tab.
  const provCount = {};
  for (const [, info] of Object.entries(models)) {
    for (const p of _getAllProviders(info)) provCount[p] = (provCount[p] || 0) + 1;
  }
  setLogoSelectOptions('modelProviderFilter', [
    {value: '', label: t('filter.allProviders'), logo: ''},
    ...[...provSet].sort().map(p => ({
      value: p, label: `${p} (${provCount[p] || 0})`, logo: _providerLogo(p),
    })),
  ], prevProv);

  // Update sort header indicators
  const sortNameEl = document.getElementById('sortName');
  const sortProviderEl = document.getElementById('sortProvider');
  const sortEnabledEl = document.getElementById('sortEnabled');
  sortNameEl.className = 'sortable' + (S._modelSortKey === 'name' ? ` ${S._modelSortDir}` : '');
  sortProviderEl.className = 'sortable' + (S._modelSortKey === 'provider' ? ` ${S._modelSortDir}` : '');
  if (sortEnabledEl) sortEnabledEl.className = 'sortable' + (S._modelSortKey === 'enabled' ? ` ${S._modelSortDir}` : '');
  syncAriaSort([
    [sortNameEl, 'name'],
    [sortProviderEl, 'provider'],
    [sortEnabledEl, 'enabled'],
  ], S._modelSortKey, S._modelSortDir);
  const sortKeyEl = document.getElementById('modelSortKey');
  if (sortKeyEl) sortKeyEl.value = S._modelSortKey;
  const sortDirEl = document.getElementById('modelSortDirBtn');
  if (sortDirEl) {
    sortDirEl.textContent = S._modelSortDir === 'asc' ? '\u2191' : '\u2193';
    sortDirEl.setAttribute('aria-label', t('sort.direction') + ': ' + t(S._modelSortDir === 'asc' ? 'sort.asc' : 'sort.desc'));
  }

  if (totalCount === 0) {
    tbody.innerHTML = `<tr role="row"><td colspan="4" role="cell" class="model-empty">${t('empty.models')}</td></tr>`;
    document.getElementById('modelCount').textContent = '';
    return;
  }

  // Filter by provider dropdown and search query
  const selectedProv = provFilter.value;
  const query = (document.getElementById('modelSearch').value || '').trim().toLowerCase();
  let entries = Object.entries(models).map(([name, info]) => {
    const provs = _getAllProviders(info);
    const prov = provs[0] || '';
    return {name, prov, provs, info};
  });
  if (selectedProv) {
    entries = entries.filter(e => e.provs.includes(selectedProv));
  }
  if (query) {
    entries = entries.filter(e => e.name.toLowerCase().includes(query) || e.provs.some(p => p.toLowerCase().includes(query)));
  }

  // Filter by model type (domain)
  if (S._modelDomain !== 'all') {
    entries = entries.filter(e => _getModelType(e.info) === S._modelDomain);
  }

  // Providers that are off altogether, and the status facet over what is left.
  const providers = S.configData.providers || {};
  const disabledProviders = new Set(
    Object.entries(providers).filter(([, cfg]) => cfg.enabled === false).map(([n]) => n),
  );
  const statusCounts = {all: entries.length, usable: 0, partial: 0, off: 0};
  for (const e of entries) statusCounts[_modelStatus(e.info, disabledProviders)] += 1;
  _buildStatusSeg(statusCounts);
  if (S._modelStatus && S._modelStatus !== 'all') {
    entries = entries.filter(e => _modelStatus(e.info, disabledProviders) === S._modelStatus);
  }

  // Show/hide reset button
  const hasFilters = S._modelDomain !== 'all' || selectedProv || query
    || (S._modelStatus && S._modelStatus !== 'all');
  const resetBtn = document.getElementById('modelResetBtn');
  if (resetBtn) resetBtn.classList.toggle('visible', !!hasFilters);


  // Sort.  A model no provider can serve sinks below the usable ones whatever
  // the chosen key is: it cannot take traffic, so it is not what an operator
  // scanning the table is looking for.
  const dir = S._modelSortDir === 'asc' ? 1 : -1;
  entries.sort((a, b) => {
    const ua = _entryUsable(a.info, disabledProviders) ? 0 : 1;
    const ub = _entryUsable(b.info, disabledProviders) ? 0 : 1;
    if (ua !== ub) return ua - ub;
    const va = S._modelSortKey === 'provider' ? a.prov : a.name;
    const vb = S._modelSortKey === 'provider' ? b.prov : b.name;
    return va.localeCompare(vb) * dir;
  });

  // Update count
  const countEl = document.getElementById('modelCount');
  countEl.textContent = t('model.count', {shown: entries.length, total: totalCount});

  if (entries.length === 0) {
    tbody.innerHTML = `<tr role="row"><td colspan="6" role="cell" class="model-empty">${t('empty.models')}</td></tr>`;
    return;
  }



  tbody.innerHTML = entries.map(({name, prov, provs, info}) => {
    const allProvsDisabled = provs.every(p => disabledProviders.has(p));
    const anyProvEnabled = provs.some(p => !disabledProviders.has(p));
    const modelEnabled = typeof info === 'object' ? info.enabled !== false : true;
    const effectiveEnabled = modelEnabled && anyProvEnabled;
    const rowDimmed = !modelEnabled || allProvsDisabled;
    const caps = typeof info === 'string' ? ['text'] : (info.capabilities || ['text']);
    const modelType = _getModelType(info);
    // Icon only: the column is narrow and the four names repeat on every row.
    // The name stays available as the badge's tooltip and accessible name.
    const capBadges = modelType === 'llm' ? caps.map(c => {
      const cls = 'cap-badge-' + c;
      const icon = _capIcons[c] || '';
      const label = capLabel(c);
      return `<span class="cap-badge ${cls}" title="${esc(label)}" aria-label="${esc(label)}">${icon}</span>`;
    }).join('') : '';
    const hasVision = caps.includes('vision');
    const hasTools = caps.includes('tools');
    const hasReasoning = caps.includes('reasoning');
    const typeBadgeCls = _typeBadgeClass(modelType);
    const typeBadge = modelType === 'llm' ? '' : `<span class="cap-badge ${typeBadgeCls}">${_CAP_ICONS[modelType] || ''}${esc(capLabel(modelType))}</span>`;
    // The upstream name is the same string on nearly every row and the provider
    // menu already lists it per provider; it stays a tooltip here.
    const upstream = (typeof info === 'object' && info.upstream_model) ? info.upstream_model : '';
    const urlTpl = (typeof info === 'object' && info.url_template) ? info.url_template : '';
    const urlTplTag = urlTpl ? ` <span style="font-size:10px;color:var(--text-dim);background:var(--bg-card);padding:1px 4px;border-radius:3px;border:1px solid var(--border)" title="${t('label.urlTemplate')}: ${esc(urlTpl)}">⛓ url</span>` : '';
    return `<tr${rowDimmed ? ' class="model-disabled"' : ''} role="row"
      onclick="if(!event.target.closest('button,input,label,a,.more-menu,.provider-link'))openModelDrawer('${esc(name)}')">
      <td class="col-check" role="cell" aria-labelledby="thModelSelect"><input type="checkbox" class="row-check" data-model="${esc(name)}" onchange="updateModelBulk()"></td>
      <td class="model-name-cell" role="cell" aria-labelledby="sortName">
        <div class="mn-inner">
          <code class="copyable" role="button" tabindex="0"
                title="${esc(upstream ? name + ' → ' + upstream : name)}"
                onclick="openModelDrawer('${esc(name)}')"
                onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();openModelDrawer('${esc(name)}')}">${esc(name)}</code>
          <span class="mn-badges">${typeBadge}${capBadges}</span>
        </div>
      </td>
      <td class="model-provider-cell" role="cell" aria-labelledby="sortProvider">${_renderProviderCell(name, info, disabledProviders)}</td>
      <td class="model-actions-cell" role="cell" aria-labelledby="sortEnabled">
        <label class="toggle row-toggle" style="margin-right:6px" title="${allProvsDisabled ? t('provider.disabled') : modelEnabled ? t('model.enabled') : t('model.disabled')}"><input type="checkbox" role="switch" ${effectiveEnabled ? 'checked' : ''} aria-checked="${effectiveEnabled}" aria-label="${esc(name)}" onchange="toggleModel('${esc(name)}')"><span class="slider"></span></label>
        <div class="test-group" style="display:inline-block">
          <button class="btn btn-sm btn-test${modelType !== 'llm' ? ' btn-test-embed' : ''}" onclick="runTest('${esc(name)}','${_defaultTestKind(modelType)}')">${t('btn.test')}</button>
          <button class="btn btn-sm btn-caret" aria-haspopup="true" aria-expanded="false" aria-label="${t('btn.test')} ${esc(name)}" onclick="toggleTestMenu(this)">&#9662;</button>
          <div class="test-menu">
            ${_testMenuItems(modelType, name, {hasTools, hasVision, hasReasoning})}
          </div>
        </div>
        <button class="btn btn-sm more-menu-toggle" aria-haspopup="true" aria-expanded="false" aria-label="${t('btn.more')} ${esc(name)}" onclick="toggleMoreMenu(this)" style="padding:3px 6px">⋯</button>
        <div class="more-menu" role="menu">
          <div class="more-menu-item" role="menuitem" aria-label="${t('btn.edit')} ${esc(name)}" onclick="closeMenus();editModel('${esc(name)}','${esc(prov)}')">${t('btn.edit')}</div>
          <div class="more-menu-item" role="menuitem" aria-label="${t('btn.copy')} ${esc(name)}" onclick="closeMenus();copyText('${esc(name)}')">${t('btn.copy')}</div>
          <div class="more-menu-item" role="menuitem" aria-label="${t('btn.clone')} ${esc(name)}" onclick="closeMenus();cloneModel('${esc(name)}')">${t('btn.clone')}</div>
          <div class="more-menu-sep" role="separator"></div>
          <div class="more-menu-item danger" role="menuitem" aria-label="${t('btn.delete')} ${esc(name)}" onclick="deleteModel('${esc(name)}', this)">${t('btn.delete')}</div>
        </div>
      </td>
    </tr>`;
  }).join('');

  _syncModelDrawer();

  _restoreProviderMenu();

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
  if (modelType !== 'llm') {
    // Non-LLM types: capability is the type name itself
    capabilities = [modelType];
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
    const replayMissing = document.getElementById('reasoningReplayMissing').value || null;
    // Merge onto what was loaded rather than rebuilding: clearing a control
    // must delete its key (an omitted key means "leave it" to the backend), and
    // fields the card shows but cannot edit must survive an unrelated save.
    // Effort range is two boxes but one field: either box alone is an edit,
    // and both empty means "inherit".
    const effMin = document.getElementById('reasoningEffortMin').value.trim();
    const effMax = document.getElementById('reasoningEffortMax').value.trim();
    const visModes = {};
    for (const mode of ['Auto', 'Concise', 'Detailed', 'None']) {
      const v = document.getElementById('reasoningVis' + mode).value;
      if (v) visModes[mode.toLowerCase()] = v;
    }
    body.reasoning_override = mergeReasoningOverride(_loadedReasoningOverride, {
      thinking_type: thinkingType,
      budget_ratio: budgetRatioStr === '' ? null : parseFloat(budgetRatioStr),
      disabled: disabledStrategy,
      replay_missing: replayMissing,
      effort_field: document.getElementById('reasoningEffortField').value.trim() || null,
      // A range is a pair: half of one is not a range, so it is treated as
      // "inherit" rather than sent as a list with a hole in it.
      effort_range: (effMin && effMax) ? [effMin, effMax] : null,
      unsigned_blocks: document.getElementById('reasoningUnsignedBlocks').value || null,
      visibility_modes: Object.keys(visModes).length ? visModes : null,
    });
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
  if (res.ok) {
    const msg = res.merged ? t('toast.modelMerged', {name, provider: res.provider}) : t('toast.modelSaved', {name});
    showToast(msg);
    closeModelDrawer();
    window.loadConfig();
  } else if (res.merge_possible && res.existing_provider) {
    if (confirm(t('confirm.mergeModel', {name, provider: res.existing_provider}))) {
      body.merge = true;
      const res2 = await api.put(`/admin/api/config/models/${encodeURIComponent(name)}`, body);
      if (res2.ok) { showToast(t('toast.modelMerged', {name, provider: body.provider})); closeModelDrawer(); window.loadConfig(); }
      else { showToast(res2.error || t('toast.error'), 'error'); }
    }
  } else { showToast(res.error || t('toast.error'), 'error'); }
}

async function _doDeleteModel(name) {
  const res = await api.del(`/admin/api/config/models/${encodeURIComponent(name)}`);
  if (res.ok) { showToast(t('toast.modelDeleted',{name})); window.loadConfig(); }
  else { showToast(res.error || t('toast.error'), 'error'); }
}

/** Delete a model after an inline confirmation.
 *
 * `btn` is the "⋯" menu item.  The menu stays open while confirming — the
 * confirmation replaces the item's own label, so hiding the menu here would
 * hide the confirmation with it.
 */
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
  } else { showToast(res.error || t('toast.error'), 'error'); }
}

function editModel(name, provider) {
  const info = S.configData.models[name];
  const caps = (typeof info === 'object' && info.capabilities) ? info.capabilities : ['text'];
  const upstream = (typeof info === 'object' && info.upstream_model) ? info.upstream_model : '';
  const provs = _getAllProviders(info);
  // An omitted provider (the panel's own Edit tab passes none) falls back to
  // the model's own, so the form never opens with an empty roster choice.
  const prov = provs.length > 1 ? (provs.find(p => {
    const cfg = (S.configData.providers || {})[p];
    return cfg && cfg.enabled !== false;
  }) || provs[0]) : (provider || provs[0] || '');
  openModelModal(name, prov, caps, upstream);
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
    document.querySelectorAll('#modelTable .row-check:checked').forEach(cb => cb.checked = false);
    updateModelBulk();
    window.loadConfig();
  } else { showToast(res.error || t('toast.error'), 'error'); }
}

function toggleMoreMenu(btn) {
  const menu = btn.nextElementSibling;
  const wasOpen = menu.classList.contains('open');
  closeMenus();
  if (!wasOpen) openMenu(menu, btn);
}

// ── Register on window for cross-module and inline-handler access ──

Object.assign(window, {
  switchModelTab, segModelType, syncCapChip, toggleStreamUrl,
  openModelModal, onModelTypeChange, updateReasoningVisibility,
  switchModelDomain, resetModelFilters, sortModels,
  renderModels, saveModel, toggleModel, editModel, cloneModel,
  selectAllModels, updateModelBulk, bulkModels, toggleMoreMenu,
  deleteModel, goToModelsForProvider, goToProviderFromModel,
  removeProviderFromModel, updateProviderWeight, _buildTypeSegControls,
  toggleModelProvider,
  setModelSortKey, toggleModelSortDir,
});

export { renderModels, _providerLogo };
