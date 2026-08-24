// Side-effect-only module: exposes functions via window.* for inline HTML handlers.
import { S } from './state.js';
import { t } from './i18n.js';
import { api, showToast, closeModal, esc } from './core.js';

// ===================== Fetch Models from Provider =====================

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
  S._fetchProvider = '';
  document.getElementById('fetchModelsContent').style.display = 'none';
  document.getElementById('fetchModelsLoading').style.display = 'none';
  document.getElementById('fetchModelsError').style.display = 'none';
  document.getElementById('fetchPrefix').value = '';
  document.getElementById('fetchModelSearch').value = '';
  document.getElementById('fetchAddBtn').disabled = true;
  // Reset type/capability selectors
  document.querySelector('input[name="fetchModelType"][value="llm"]').checked = true;
  document.getElementById('fetchCapText').checked = true;
  document.getElementById('fetchCapVision').checked = false;
  document.getElementById('fetchCapTools').checked = true;
  document.getElementById('fetchCapReasoning').checked = false;
  document.getElementById('fetchCapsRow').style.display = '';
  document.getElementById('fetchModelsModal').classList.add('open');
}

function onFetchTypeChange() {
  const isLLM = document.querySelector('input[name="fetchModelType"]:checked').value === 'llm';
  document.getElementById('fetchCapsRow').style.display = isLLM ? '' : 'none';
}

function _getFetchCapabilities() {
  const type = document.querySelector('input[name="fetchModelType"]:checked').value;
  if (type === 'embedding') return ['embedding'];
  const caps = [];
  if (document.getElementById('fetchCapText').checked) caps.push('text');
  if (document.getElementById('fetchCapVision').checked) caps.push('vision');
  if (document.getElementById('fetchCapTools').checked) caps.push('tools');
  if (document.getElementById('fetchCapReasoning').checked) caps.push('reasoning');
  return caps.length > 0 ? caps : ['text'];
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
    const exists = displayName in existingModels;
    return `<label${exists ? ' style="opacity:0.6"' : ''}>
      <input type="checkbox" value="${esc(m)}" onchange="updateFetchCount()"${exists ? ' checked data-exists="true"' : ''}>
      <span>${esc(m)}</span>${exists ? ' <span class="exists-tag" style="font-size:11px;color:var(--text-dim)">(exists)</span>' : ''}
    </label>`;
  }).join('');

  updateFetchCount();
}

function filterFetchedModels() {
  renderFetchedModels();
}

function toggleAllFetched(checked) {
  const boxes = document.querySelectorAll('#fetchModelsList input[type="checkbox"]');
  boxes.forEach(cb => cb.checked = checked);
  updateFetchCount();
}

function updateFetchCount() {
  const all = document.querySelectorAll('#fetchModelsList input[type="checkbox"]');
  const checked = document.querySelectorAll('#fetchModelsList input[type="checkbox"]:checked');
  // Check if there are any changes: unchecked exists or checked non-exists
  const uncheckedExists = document.querySelectorAll('#fetchModelsList input[type="checkbox"][data-exists="true"]:not(:checked)');
  const checkedNew = document.querySelectorAll('#fetchModelsList input[type="checkbox"]:checked:not([data-exists="true"])');
  const hasChanges = uncheckedExists.length > 0 || checkedNew.length > 0;
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
    // Models to add: checked and not already existing
    const toAdd = [...document.querySelectorAll('#fetchModelsList input[type="checkbox"]:checked:not([data-exists="true"])')].map(cb => cb.value);
    // Models to remove: unchecked but marked as existing
    const toRemove = [...document.querySelectorAll('#fetchModelsList input[type="checkbox"][data-exists="true"]:not(:checked)')].map(cb => {
      return prefix ? prefix + cb.value : cb.value;
    });

    let addedCount = 0;
    let removedCount = 0;

    // Add new models
    if (toAdd.length > 0) {
      const res = await api.post('/admin/api/config/models', {
        provider: S._fetchProvider,
        models: toAdd,
        prefix: prefix,
        capabilities: _getFetchCapabilities(),
      });
      if (res.ok) {
        addedCount = (res.added || []).length;
      } else {
        showToast(res.error || 'Failed to add models', 'error');
      }
    }

    // Remove deselected existing models
    for (const name of toRemove) {
      const res = await api.del(`/admin/api/config/models/${encodeURIComponent(name)}`);
      if (res.ok) removedCount++;
    }

    // Show result
    const msgs = [];
    if (addedCount > 0) msgs.push(t('toast.modelsAdded', {count: addedCount}));
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
