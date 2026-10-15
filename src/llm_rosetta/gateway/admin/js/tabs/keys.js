import { S } from '../core/state.js';
import { t } from '../core/i18n.js';
import { api, showToast, closeModal, esc, fmtTokens, inlineConfirm } from '../core/utils.js';

// ===================== API Keys Tab =====================

async function loadKeys() {
  try {
    // Usage is auxiliary — a failure there must not blank the key list.
    const [keysData, usageData] = await Promise.all([
      api.get('/admin/api/keys'),
      api.get('/admin/api/keys/usage').catch(e => {
        console.error('Failed to load key usage:', e);
        return null;
      }),
    ]);
    S.keysData = keysData;
    S.keyUsage = usageData;
    // Keep the selected tier valid if the server's window list changed.
    const windows = usageWindows();
    if (!windows.includes(S.keyUsageWindow)) S.keyUsageWindow = windows[0];
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

// Tiers come from the endpoint's window list, so adding a window server-side
// needs no edit here.
function usageWindows() {
  const w = S.keyUsage && S.keyUsage.windows;
  return (w && w.length) ? w : ['5h', '24h', '7d', '30d'];
}

function windowLabel(w) {
  const key = 'keys.range' + w;
  const label = t(key);
  return label === key ? w : label;  // unknown window → show the raw name
}

function renderKeyUsageControl() {
  const ctrl = document.getElementById('keyUsageRange');
  if (!ctrl) return;
  const win = S.keyUsageWindow || usageWindows()[0];
  ctrl.innerHTML = usageWindows().map(w =>
    `<label class="${w === win ? 'active' : ''}" onclick="setKeyUsageWindow('${esc(w)}')">` +
    `<input type="radio" name="kuRange"${w === win ? ' checked' : ''}>` +
    `<span>${esc(windowLabel(w))}</span></label>`
  ).join('');
}

// Switch the usage tier without a refetch — the endpoint returns every
// window, so this is a pure client-side re-render.
function setKeyUsageWindow(win) {
  if (!usageWindows().includes(win)) return;
  S.keyUsageWindow = win;
  renderKeyUsageControl();
  renderKeys();
}

// A read-only row for usage no live key owns.  It has to be visible, or the
// panel's figures would silently exclude traffic the dashboard counts.
function attributedUsageRow(label, win, tip, counts) {
  const w = counts || {};
  const title = esc(tip);
  return `<tr style="color:var(--text-dim)">
    <td title="${title}">${esc(label)}</td>
    <td>—</td>
    <td>—</td>
    <td style="text-align:right" title="${title}">${(w.request_count || 0).toLocaleString()}</td>
    <td style="text-align:right" title="${title}">${fmtTokens(w.input_tokens || 0)}</td>
    <td style="text-align:right" title="${title}">${fmtTokens(w.output_tokens || 0)}</td>
    <td style="text-align:right" title="${title}"><strong>${fmtTokens(w.total_tokens || 0)}</strong></td>
    <td></td>
  </tr>`;
}

// The rows that account for everything the live keys do not.
function renderUnattributedRows(win) {
  const usage = S.keyUsage || {};
  const rows = [];
  if (usage.internal) {
    rows.push(attributedUsageRow(
      t('keys.internalRow'), win, t('keys.internalHint'),
      usage.internal.windows && usage.internal.windows[win]
    ));
  }
  if (usage.unattributed) {
    const detail = Object.entries(usage.unattributed.by_label || {})
      .map(([label, w]) => `${label}: ${fmtTokens(((w || {})[win] || {}).total_tokens || 0)}`)
      .join('\n');
    rows.push(attributedUsageRow(
      t('keys.unattributedRow'), win,
      t('keys.unattributedHint') + (detail ? '\n\n' + detail : ''),
      usage.unattributed.windows && usage.unattributed.windows[win]
    ));
  }
  return rows;
}

function renderKeys() {
  // Re-rendered here too so a language switch relabels the tier control.
  renderKeyUsageControl();
  const tbody = document.getElementById('keysTable');
  const keys = (S.keysData && S.keysData.keys) || [];
  const win = S.keyUsageWindow || usageWindows()[0];
  const extraRows = renderUnattributedRows(win);
  if (keys.length === 0 && extraRows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="color:var(--text-dim)">${t('keys.noKeys')}</td></tr>`;
    return;
  }
  const usageKeys = (S.keyUsage && S.keyUsage.keys) || {};
  const usageTip = esc(t('keys.rotateBucketHint'));
  tbody.innerHTML = keys.map(k => {
    const created = k.created ? new Date(k.created).toLocaleDateString() : '—';
    const createdFull = k.created ? new Date(k.created).toLocaleString(undefined, {year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',timeZoneName:'shortOffset'}) : '';
    const lastUsed = k.last_used ? new Date(k.last_used).toLocaleDateString() : '—';
    const lastUsedFull = k.last_used ? new Date(k.last_used).toLocaleString(undefined, {year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',timeZoneName:'shortOffset'}) : '';
    const entry = usageKeys[k.id];
    const w = (entry && entry.windows && entry.windows[win]) || {};
    const reqs = (w.request_count || 0).toLocaleString();
    // Two keys behind one label cannot be told apart in the log, so this row
    // reads zero while its traffic shows up under Unattributed.
    const sharedMark = (entry && entry.shared_label)
      ? `<span title="${esc(t('keys.sharedLabelHint'))}" style="color:var(--orange);margin-left:4px;cursor:help">⚠</span>`
      : '';
    return `<tr>
      <td><span class="key-label-text" id="label-${k.id}">${esc(k.label || '—')}</span>
        <button class="key-btn" style="margin-left:4px" onclick="editKeyLabel('${k.id}','${esc(k.label || '')}')" title="Edit"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/></svg></button>${sharedMark}</td>
      <td title="${createdFull}">${created}</td>
      <td title="${lastUsedFull}">${lastUsed}</td>
      <td style="text-align:right" title="${usageTip}">${reqs}</td>
      <td style="text-align:right" title="${usageTip}">${fmtTokens(w.input_tokens || 0)}</td>
      <td style="text-align:right" title="${usageTip}">${fmtTokens(w.output_tokens || 0)}</td>
      <td style="text-align:right" title="${usageTip}"><strong>${fmtTokens(w.total_tokens || 0)}</strong></td>
      <td style="white-space:nowrap;text-align:right"><button class="btn btn-sm" onclick="rotateKey('${k.id}','${esc(k.label || k.id)}', this)">${t('btn.rotate')}</button> <button class="btn btn-sm btn-danger" onclick="deleteKey('${k.id}','${esc(k.label || k.id)}', this)">${t('btn.delete')}</button></td>
    </tr>`;
  }).join('') + extraRows.join('');
}

function openKeyModal() {
  document.getElementById('keyLabel').value = '';
  document.getElementById('keyManual').value = '';
  openModal('keyModal');
}

async function generateKey() {
  const label = document.getElementById('keyLabel').value.trim();
  const manual = document.getElementById('keyManual').value.trim();
  const body = {label};
  if (manual) body.key = manual;
  const res = await api.post('/admin/api/keys', body);
  if (res.ok) {
    closeModal('keyModal');
    showToast(t('toast.keySaved'));
    // Show the created key for one-time copy
    document.getElementById('createdKeyValue').value = res.key.key;
    openModal('keyCreatedModal');
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
      openModal('keyCreatedModal');
      loadKeys();
    } else {
      showToast(res.error || 'Failed', 'error');
    }
  });
}

async function backfillKeysLastUsed() {
  try {
    const res = await api.post('/admin/api/keys/backfill-last-used');
    if (res && res.error) { showToast(res.error, 'error'); return; }
    showToast(t('toast.keysBackfilled').replace('{n}', res.updated || 0));
    if (res.updated > 0) loadKeys();
  } catch(e) { showToast('Backfill failed', 'error'); }
}

Object.assign(window, {
  loadKeys, loadLogKeyLabels, renderKeys, openKeyModal,
  generateKey, copyCreatedKey, deleteKey, rotateKey, editKeyLabel,
  backfillKeysLastUsed, setKeyUsageWindow,
});

export { loadKeys, loadLogKeyLabels, renderKeys, setKeyUsageWindow };
