import { S } from '../core/state.js';
import { t } from '../core/i18n.js';
import { api, showToast, closeModal, openModal, esc, fmtTokens, inlineConfirm } from '../core/utils.js';
import { syncAriaSort, sortByKey } from '../core/sort.js';

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
  return `<tr style="color:var(--text-dim)" role="row">
    <td class="key-label-cell" role="cell" aria-labelledby="keySortLabel" title="${title}">${esc(label)}</td>
    <td class="key-created-cell" role="cell" aria-labelledby="keySortCreated" data-label="${esc(t('col.created'))}">—</td>
    <td class="key-lastused-cell" role="cell" aria-labelledby="keySortLastUsed" data-label="${esc(t('col.lastUsed'))}">—</td>
    <td class="key-requests-cell" role="cell" aria-labelledby="keySortRequests" data-label="${esc(t('col.requests'))}" title="${title}">${(w.request_count || 0).toLocaleString()}</td>
    <td class="key-input-cell" role="cell" aria-labelledby="keySortInput" data-label="${esc(t('tokenUsage.col.input'))}" title="${title}">${fmtTokens(w.input_tokens || 0)}</td>
    <td class="key-output-cell" role="cell" aria-labelledby="keySortOutput" data-label="${esc(t('tokenUsage.col.output'))}" title="${title}">${fmtTokens(w.output_tokens || 0)}</td>
    <td class="key-total-cell" role="cell" aria-labelledby="keySortTotal" data-label="${esc(t('tokenUsage.col.total'))}" title="${title}"><strong>${fmtTokens(w.total_tokens || 0)}</strong></td>
    <td class="key-actions-cell" role="cell"></td>
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

// Column sorting mirrors the Models table: clicking the active column
// flips the direction, clicking another starts ascending.  Until a column is
// chosen the server's order is kept.
function sortKeys(key) {
  if (S._keysSortKey === key) {
    S._keysSortDir = S._keysSortDir === 'asc' ? 'desc' : 'asc';
  } else {
    S._keysSortKey = key;
    S._keysSortDir = 'asc';
  }
  renderKeys();
}

// The card layout hides the sortable header row, so the toolbar offers the
// same keys as a select plus a direction toggle at phone width.  The empty
// value restores the server's order.
function setKeySortKey(key) {
  S._keysSortKey = key;
  renderKeys();
}

function toggleKeySortDir() {
  S._keysSortDir = S._keysSortDir === 'asc' ? 'desc' : 'asc';
  renderKeys();
}

// A single unparseable timestamp would turn every comparison into NaN and
// freeze the whole column, so a bad value reads as "oldest" instead.
function _parseDate(value) {
  if (!value) return 0;
  const ms = Date.parse(value);
  return Number.isFinite(ms) ? ms : 0;
}

function _keySortValue(k, win) {
  const entry = S.keyUsage && S.keyUsage.keys && S.keyUsage.keys[k.id];
  const w = (entry && entry.windows && entry.windows[win]) || {};
  switch (S._keysSortKey) {
    case 'label': return (k.label || '').toLowerCase();
    case 'created': return _parseDate(k.created);
    case 'last_used': return _parseDate(k.last_used);
    case 'requests': return w.request_count || 0;
    case 'input': return w.input_tokens || 0;
    case 'output': return w.output_tokens || 0;
    case 'total': return w.total_tokens || 0;
    default: return 0;
  }
}

function _syncKeySortHeaders() {
  const headers = [];
  for (const key of ['label', 'created', 'last_used', 'requests', 'input', 'output', 'total']) {
    const el = document.getElementById('keySort' + key.split('_').map(s => s[0].toUpperCase() + s.slice(1)).join(''));
    if (!el) continue;
    el.className = 'sortable' + (S._keysSortKey === key ? ' ' + S._keysSortDir : '');
    headers.push([el, key]);
  }
  syncAriaSort(headers, S._keysSortKey, S._keysSortDir);
}

function renderKeys() {
  // Re-rendered here too so a language switch relabels the tier control.
  renderKeyUsageControl();
  const tbody = document.getElementById('keysTable');
  const allKeys = (S.keysData && S.keysData.keys) || [];
  const query = (document.getElementById('keySearch')?.value || '').trim().toLowerCase();
  const win = S.keyUsageWindow || usageWindows()[0];
  // sortByKey returns a copy, so the server's order survives until a column
  // is chosen and the mobile "Default" option can restore it.
  const keys = sortByKey(
    query ? allKeys.filter(k => (k.label || '').toLowerCase().includes(query)) : allKeys,
    {
      key: S._keysSortKey,
      dir: S._keysSortDir,
      valueOf: k => _keySortValue(k, win),
      labelOf: k => k.label || '',
    },
  );
  _syncKeySortHeaders();
  const sortSel = document.getElementById('keySortKey');
  if (sortSel) sortSel.value = S._keysSortKey;
  const sortDirBtn = document.getElementById('keySortDirBtn');
  if (sortDirBtn) {
    // With the server's order restored there is no direction to flip.
    sortDirBtn.disabled = !S._keysSortKey;
    sortDirBtn.textContent = S._keysSortDir === 'asc' ? '↑' : '↓';
    sortDirBtn.setAttribute('aria-label', t('sort.direction') + ': ' + t(S._keysSortDir === 'asc' ? 'sort.asc' : 'sort.desc'));
  }
  // Unattributed rows carry no label, so they cannot match a search.
  const extraRows = query ? [] : renderUnattributedRows(win);
  const countEl = document.getElementById('keyCount');
  if (countEl) {
    countEl.textContent = query
      ? t('keys.countFiltered', {shown: keys.length, total: allKeys.length})
      : t('keys.count', {count: allKeys.length});
  }
  if (allKeys.length === 0 && extraRows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="color:var(--text-dim)">${t('keys.noKeys')}</td></tr>`;
    return;
  }
  if (keys.length === 0 && extraRows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="color:var(--text-dim)">${t('keys.noMatch')}</td></tr>`;
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
    return `<tr role="row">
      <td class="key-label-cell" role="cell" aria-labelledby="keySortLabel"><span class="key-label-text" id="label-${k.id}">${esc(k.label || '—')}</span>
        <button class="key-btn" style="margin-left:4px" onclick="editKeyLabel('${k.id}')" title="Edit"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/></svg></button>${sharedMark}</td>
      <td class="key-created-cell" role="cell" aria-labelledby="keySortCreated" data-label="${esc(t('col.created'))}" title="${createdFull}">${created}</td>
      <td class="key-lastused-cell" role="cell" aria-labelledby="keySortLastUsed" data-label="${esc(t('col.lastUsed'))}" title="${lastUsedFull}">${lastUsed}</td>
      <td class="key-requests-cell" role="cell" aria-labelledby="keySortRequests" data-label="${esc(t('col.requests'))}" title="${usageTip}">${reqs}</td>
      <td class="key-input-cell" role="cell" aria-labelledby="keySortInput" data-label="${esc(t('tokenUsage.col.input'))}" title="${usageTip}">${fmtTokens(w.input_tokens || 0)}</td>
      <td class="key-output-cell" role="cell" aria-labelledby="keySortOutput" data-label="${esc(t('tokenUsage.col.output'))}" title="${usageTip}">${fmtTokens(w.output_tokens || 0)}</td>
      <td class="key-total-cell" role="cell" aria-labelledby="keySortTotal" data-label="${esc(t('tokenUsage.col.total'))}" title="${usageTip}"><strong>${fmtTokens(w.total_tokens || 0)}</strong></td>
      <td class="key-actions-cell" role="cell" style="white-space:nowrap"><button class="btn btn-sm" onclick="rotateKey('${k.id}','${esc(k.label || k.id)}', this)">${t('btn.rotate')}</button> <button class="btn btn-sm btn-danger" onclick="deleteKey('${k.id}','${esc(k.label || k.id)}', this)">${t('btn.delete')}</button></td>
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

// The row's label span is the source of truth: embedding the label in the
// onclick attribute would need JS-string escaping on top of HTML escaping.
let _editingKeyId = null;

function editKeyLabel(id) {
  _editingKeyId = id;
  const span = document.getElementById('label-' + id);
  const current = span ? span.textContent.trim() : '';
  const input = document.getElementById('keyLabelInput');
  input.value = current === '—' ? '' : current;
  openModal('keyLabelModal');
  setTimeout(() => { input.focus(); input.select(); }, 0);
}

function onKeyLabelInputKeydown(e) {
  if (e.key === 'Enter') {
    e.preventDefault();
    saveKeyLabel();
  } else if (e.key === 'Escape') {
    e.preventDefault();
    closeModal('keyLabelModal');
  }
}

async function saveKeyLabel() {
  if (!_editingKeyId) return;
  const label = document.getElementById('keyLabelInput').value.trim();
  const res = await api.put(`/admin/api/keys/${encodeURIComponent(_editingKeyId)}`, {label});
  if (res.ok) {
    closeModal('keyLabelModal');
    _editingKeyId = null;
    showToast(t('toast.keyLabelUpdated'));
    loadKeys();
  } else {
    showToast(res.error || 'Failed', 'error');
  }
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
  onKeyLabelInputKeydown, saveKeyLabel,
  backfillKeysLastUsed, setKeyUsageWindow, sortKeys, setKeySortKey, toggleKeySortDir,
});

export { loadKeys, loadLogKeyLabels, renderKeys, setKeyUsageWindow };
