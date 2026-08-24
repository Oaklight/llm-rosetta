import { S } from './state.js';
import { t } from './i18n.js';
import { api, showToast, closeModal, esc, inlineConfirm } from './core.js';

// ===================== API Keys Tab =====================

async function loadKeys() {
  try {
    S.keysData = await api.get('/admin/api/keys');
    renderKeys();
    window.updateKeyFilterOptions();
  } catch(e) {
    console.error('Failed to load keys:', e);
  }
}

async function loadLogKeyLabels() {
  try {
    const data = await api.get('/admin/api/requests/key-labels');
    S.logKeyLabels = data.labels || [];
    window.updateKeyFilterOptions();
  } catch(e) {
    console.error('Failed to load log key labels:', e);
  }
}

function renderKeys() {
  const tbody = document.getElementById('keysTable');
  const keys = (S.keysData && S.keysData.keys) || [];
  if (keys.length === 0) {
    tbody.innerHTML = `<tr><td colspan="4" style="color:var(--text-dim)">${t('keys.noKeys')}</td></tr>`;
    return;
  }
  tbody.innerHTML = keys.map(k => {
    const created = k.created ? new Date(k.created).toLocaleDateString() : '—';
    return `<tr>
      <td><span class="key-label-text" id="label-${k.id}">${esc(k.label || '—')}</span>
        <button class="key-btn" style="margin-left:4px" onclick="editKeyLabel('${k.id}','${esc(k.label || '')}')" title="Edit"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/></svg></button></td>
      <td><code>${esc((k.allowed_shims || ['*']).join(', '))}</code></td>
      <td>${created}</td>
      <td style="white-space:nowrap"><button class="btn btn-sm" onclick="rotateKey('${k.id}','${esc(k.label || k.id)}', this)">${t('btn.rotate')}</button> <button class="btn btn-sm btn-danger" onclick="deleteKey('${k.id}','${esc(k.label || k.id)}', this)">${t('btn.delete')}</button></td>
    </tr>`;
  }).join('');
}

function openKeyModal() {
  document.getElementById('keyLabel').value = '';
  document.getElementById('keyManual').value = '';
  document.getElementById('keyAllowedShims').value = '*';
  document.getElementById('keyModal').classList.add('open');
}

async function generateKey() {
  const label = document.getElementById('keyLabel').value.trim();
  const manual = document.getElementById('keyManual').value.trim();
  const shimsRaw = document.getElementById('keyAllowedShims').value.trim();
  const allowed_shims = shimsRaw ? shimsRaw.split(',').map(s => s.trim()).filter(Boolean) : ['*'];
  const body = {label, allowed_shims};
  if (manual) body.key = manual;
  const res = await api.post('/admin/api/keys', body);
  if (res.ok) {
    closeModal('keyModal');
    showToast(t('toast.keySaved'));
    // Show the created key for one-time copy
    document.getElementById('createdKeyValue').value = res.key.key;
    document.getElementById('keyCreatedModal').classList.add('open');
    loadKeys();
  } else {
    showToast(res.error || 'Failed', 'error');
  }
}

async function copyCreatedKey() {
  const input = document.getElementById('createdKeyValue');
  try {
    await navigator.clipboard.writeText(input.value);
    showToast(t('toast.keyCopied'));
  } catch(e) {
    showToast('Copy failed', 'error');
  }
}

async function _doDeleteKey(id, label) {
  const res = await api.del(`/admin/api/keys/${encodeURIComponent(id)}`);
  if (res.ok) { showToast(t('toast.keyDeleted')); loadKeys(); }
  else { showToast(res.error || 'Failed', 'error'); }
}

async function editKeyLabel(id, currentLabel) {
  const newLabel = prompt('Label:', currentLabel);
  if (newLabel === null) return;
  const res = await api.put(`/admin/api/keys/${encodeURIComponent(id)}`, {label: newLabel});
  if (res.ok) { showToast(t('toast.keyLabelUpdated')); loadKeys(); }
  else { showToast(res.error || 'Failed', 'error'); }
}

function deleteKey(id, label, btn) {
  if (!btn) { _doDeleteKey(id, label); return; }
  inlineConfirm(btn, () => _doDeleteKey(id, label));
}

function rotateKey(id, label, btn) {
  inlineConfirm(btn, async () => {
    const res = await api.post(`/admin/api/keys/${encodeURIComponent(id)}/rotate`);
    if (res.ok) {
      showToast(t('toast.keyRotated', {label}));
      // Show the new key for one-time copy
      document.getElementById('createdKeyValue').value = res.key;
      document.getElementById('keyCreatedModal').classList.add('open');
      loadKeys();
    } else {
      showToast(res.error || 'Failed', 'error');
    }
  });
}

Object.assign(window, {
  loadKeys, loadLogKeyLabels, renderKeys, openKeyModal,
  generateKey, copyCreatedKey, deleteKey, rotateKey, editKeyLabel,
});

export { loadKeys, loadLogKeyLabels, renderKeys };
