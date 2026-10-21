// Side-effect-only module: exposes functions via window.* for inline HTML handlers.
import { S } from '../core/state.js';
import { t } from '../core/i18n.js';
import { api, showToast, closeModal, esc } from '../core/utils.js';

// ===================== Fetch Models from Provider =====================

/**
 * Build fetch-models type radios dynamically from model_types metadata.
 * Replaces the previously hardcoded radio buttons in admin.html.
 */
function _buildFetchTypeRadios() {
  const container = document.getElementById('fetchTypeRadios');
  if (!container || !S.configData?.model_types) return;
  container.innerHTML = '';
  const types = S.configData.model_types;
  for (let i = 0; i < types.length; i++) {
    const desc = types[i];
    const label = document.createElement('label');
    const checked = desc.is_llm ? ' checked' : '';
    const labelText = t('label.' + desc.name);
    label.innerHTML = `<input type="radio" name="fetchModelType" value="${esc(desc.name)}"${checked} onchange="onFetchTypeChange()"> <span data-i18n="label.${esc(desc.name)}">${esc(labelText)}</span>`;
    container.appendChild(label);
  }
}

function openFetchModelsModal() {
  // Populate provider dropdown
  const sel = document.getElementById('fetchProvider');
  sel.innerHTML = `<option value="">${t('label.selectOne')}</option>`;
  if (S.configData) {
    for (const [name, cfg] of Object.entries(S.configData.providers)) {
      if (cfg.enabled === false) continue;
      sel.innerHTML += `<option value="${esc(name)}">${esc(name)}</option>`;
    }
  }
  // Reset state
  S._fetchedModels = [];
  S._fetchUpstreamMap = {};
  S._fetchProvider = '';
  document.getElementById('fetchModelsContent').style.display = 'none';
  document.getElementById('fetchModelsLoading').style.display = 'none';
  document.getElementById('fetchModelsError').style.display = 'none';
  document.getElementById('fetchPrefix').value = '';
  document.getElementById('fetchModelSearch').value = '';
  document.getElementById('fetchAddBtn').disabled = true;
  // Build type radios from metadata and reset selectors
  _buildFetchTypeRadios();
  const llmRadio = document.querySelector('input[name="fetchModelType"][value="llm"]');
  if (llmRadio) llmRadio.checked = true;
  document.getElementById('fetchCapText').checked = true;
  document.getElementById('fetchCapVision').checked = false;
  document.getElementById('fetchCapTools').checked = true;
  document.getElementById('fetchCapReasoning').checked = false;
  document.getElementById('fetchCapsRow').style.display = '';
  onFetchTypeChange();
  openModal('fetchModelsModal');
}

function onFetchTypeChange() {
  const isLLM = document.querySelector('input[name="fetchModelType"]:checked').value === 'llm';
  document.getElementById('fetchCapsRow').style.display = isLLM ? '' : 'none';
}

function _getFetchCapabilities() {
  const type = document.querySelector('input[name="fetchModelType"]:checked').value;
  // LLM: capabilities come from the checkboxes
  if (type === 'llm') {
    const caps = [];
    if (document.getElementById('fetchCapText').checked) caps.push('text');
    if (document.getElementById('fetchCapVision').checked) caps.push('vision');
    if (document.getElementById('fetchCapTools').checked) caps.push('tools');
    if (document.getElementById('fetchCapReasoning').checked) caps.push('reasoning');
    return caps.length > 0 ? caps : ['text'];
  }
  // Non-LLM types: the type name is the single capability
  return [type];
}

function _getFetchModelType() {
  return document.querySelector('input[name="fetchModelType"]:checked').value;
}

async function doFetchModels() {
  const provider = document.getElementById('fetchProvider').value;
  if (!provider) {
    document.getElementById('fetchModelsContent').style.display = 'none';
    return;
  }
  S._fetchProvider = provider;
  document.getElementById('fetchModelsContent').style.display = 'none';
  document.getElementById('fetchModelsError').style.display = 'none';
  document.getElementById('fetchModelsLoading').style.display = 'block';

  try {
    const data = await api.get(`/admin/api/config/providers/${encodeURIComponent(provider)}/models`);
    document.getElementById('fetchModelsLoading').style.display = 'none';
    if (data.error) {
      document.getElementById('fetchModelsError').textContent = data.error;
      document.getElementById('fetchModelsError').style.display = 'block';
      return;
    }
    S._fetchedModels = data.models || [];
    S._fetchUpstreamMap = data.upstream_map || {};
    if (S._fetchedModels.length === 0) {
      document.getElementById('fetchModelsError').textContent = t('fetch.noModels');
      document.getElementById('fetchModelsError').style.display = 'block';
      return;
    }
    document.getElementById('fetchModelsContent').style.display = 'block';
    renderFetchedModels();
  } catch (e) {
    document.getElementById('fetchModelsLoading').style.display = 'none';
    document.getElementById('fetchModelsError').textContent = e.message || String(e);
    document.getElementById('fetchModelsError').style.display = 'block';
  }
}

function _getExistingProviders(entry) {
  if (!entry) return [];
  if (entry.providers) return entry.providers.map(p => typeof p === 'string' ? p : p.name);
  if (entry.provider) return [entry.provider];
  return [];
}

function renderFetchedModels() {
  const list = document.getElementById('fetchModelsList');
  const query = (document.getElementById('fetchModelSearch').value || '').trim().toLowerCase();
  const existingModels = S.configData.models || {};
  const prefix = document.getElementById('fetchPrefix').value || '';

  let models = S._fetchedModels;
  if (query) {
    models = models.filter(m => m.toLowerCase().includes(query));
  }

  list.innerHTML = models.map(m => {
    const displayName = prefix ? prefix + m : m;
    const entry = existingModels[displayName];
    const existingProviders = _getExistingProviders(entry);
    const providerAlreadyPresent = existingProviders.includes(S._fetchProvider);
    const canAddProvider = entry && !providerAlreadyPresent;
    const exists = entry && providerAlreadyPresent;
    const upstreamId = S._fetchUpstreamMap ? S._fetchUpstreamMap[m] : '';
    const upstreamHint = upstreamId ? ` <span style="font-size:11px;color:var(--text-dim)">→ ${esc(upstreamId)}</span>` : '';

    let attrs = '';
    let tag = '';
    let style = '';
    if (exists) {
      attrs = ' checked data-exists="true"';
      tag = ` <span class="exists-tag" style="font-size:11px;color:var(--text-dim)">(exists)</span>`;
      style = ' style="opacity:0.6"';
    } else if (canAddProvider) {
      attrs = ' data-can-add="true"';
      tag = ` <span class="add-provider-tag" style="font-size:11px;color:var(--blue)">${esc(t('fetch.addProvider'))}</span>`;
    }

    return `<label${style}>
      <input type="checkbox" value="${esc(m)}" onchange="updateFetchCount()"${attrs}>
      <span>${esc(m)}</span>${upstreamHint}${tag}
    </label>`;
  }).join('');

  updateFetchCount();
}

function filterFetchedModels() {
  renderFetchedModels();
}

function toggleAllFetched(checked) {
  const boxes = document.querySelectorAll('#fetchModelsList input[type="checkbox"]');
  boxes.forEach(cb => {
    if (!cb.dataset.exists) cb.checked = checked;
  });
  updateFetchCount();
}

function updateFetchCount() {
  const all = document.querySelectorAll('#fetchModelsList input[type="checkbox"]');
  const checked = document.querySelectorAll('#fetchModelsList input[type="checkbox"]:checked');
  const uncheckedExists = document.querySelectorAll('#fetchModelsList input[type="checkbox"][data-exists="true"]:not(:checked)');
  const checkedNew = document.querySelectorAll('#fetchModelsList input[type="checkbox"]:checked:not([data-exists="true"]):not([data-can-add="true"])');
  const checkedCanAdd = document.querySelectorAll('#fetchModelsList input[type="checkbox"][data-can-add="true"]:checked');
  const hasChanges = uncheckedExists.length > 0 || checkedNew.length > 0 || checkedCanAdd.length > 0;
  document.getElementById('fetchCount').textContent = t('fetch.count', {checked: checked.length, total: all.length});
  const btn = document.getElementById('fetchAddBtn');
  btn.disabled = !hasChanges;
  btn.textContent = t('btn.applyChanges');
}

async function bulkAddFetchedModels() {
  const prefix = document.getElementById('fetchPrefix').value || '';
  const btn = document.getElementById('fetchAddBtn');
  btn.disabled = true;
  btn.textContent = '...';

  try {
    const toAdd = [...document.querySelectorAll('#fetchModelsList input[type="checkbox"]:checked:not([data-exists="true"]):not([data-can-add="true"])')].map(cb => cb.value);
    const toAppend = [...document.querySelectorAll('#fetchModelsList input[type="checkbox"][data-can-add="true"]:checked')].map(cb => cb.value);
    const toRemove = [...document.querySelectorAll('#fetchModelsList input[type="checkbox"][data-exists="true"]:not(:checked)')].map(cb => {
      return prefix ? prefix + cb.value : cb.value;
    });

    let addedCount = 0;
    let appendedCount = 0;
    let removedCount = 0;

    if (toAdd.length > 0 || toAppend.length > 0) {
      const modelType = _getFetchModelType();
      const body = {
        provider: S._fetchProvider,
        models: toAdd,
        prefix: prefix,
        type: modelType,
        capabilities: _getFetchCapabilities(),
        upstream_map: S._fetchUpstreamMap,
      };
      if (toAppend.length > 0) {
        body.models_append = toAppend;
      }
      const res = await api.post('/admin/api/config/models', body);
      if (res.ok) {
        addedCount = (res.added || []).length;
        appendedCount = (res.appended || []).length;
      } else {
        showToast(res.error || 'Failed to add models', 'error');
      }
    }

    for (const name of toRemove) {
      const res = await api.del(`/admin/api/config/models/${encodeURIComponent(name)}`);
      if (res.ok) removedCount++;
    }

    const msgs = [];
    if (addedCount > 0) msgs.push(t('toast.modelsAdded', {count: addedCount}));
    if (appendedCount > 0) msgs.push(t('toast.providersAppended', {count: appendedCount}));
    if (removedCount > 0) msgs.push(t('toast.modelsRemoved', {count: removedCount}));
    if (msgs.length > 0) {
      showToast(msgs.join(', '));
    } else {
      showToast(t('toast.modelsAllExist'));
    }
    closeModal('fetchModelsModal');
    window.loadConfig();
  } catch (e) {
    showToast(String(e), 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = t('btn.applyChanges');
  }
}

Object.assign(window, {
  openFetchModelsModal, onFetchTypeChange, doFetchModels,
  filterFetchedModels, toggleAllFetched, updateFetchCount,
  bulkAddFetchedModels,
});
