/**
 * logo-picker.js — Compact icon picker for provider logos.
 *
 * Renders as a clickable icon button. When clicked, opens a dropdown
 * with a search input and grid of lobehub icons from jsdelivr CDN.
 * Empty state shows an SVG placeholder.
 */

const _ICON_PKG = '@lobehub/icons-static-svg';
const _CDN_BASE = 'https://cdn.jsdelivr.net/npm';
const _ICON_URL = (name) => `${_CDN_BASE}/${_ICON_PKG}@latest/icons/${name}.svg`;

const _POPULAR_ICONS = [
  'openai', 'anthropic', 'claude', 'google', 'gemini',
  'deepseek', 'meta', 'mistral', 'cohere', 'perplexity',
  'qwen', 'zhipu', 'moonshot', 'minimax', 'groq', 'together',
  'fireworks', 'huggingface', 'cloudflare',
  'aws', 'azure', 'bedrock', 'vertexai',
  'ollama', 'lmstudio', 'nvidia', 'sambanova', 'cerebras', 'xai', 'grok',
];

let _allIcons = null;
let _fetchPromise = null;
let _pickerOpen = false;
let _selectedName = '';
let _customUrl = '';
let _highlightIdx = -1;

export function initLogoPicker() {
  const trigger = document.getElementById('logoPickerTrigger');
  const input = document.getElementById('logoPickerInput');
  const clear = document.getElementById('logoPickerClear');
  if (!trigger || !input) return;

  trigger.addEventListener('click', (e) => {
    e.stopPropagation();
    if (_pickerOpen) _closeDropdown();
    else _openDropdown();
  });
  input.addEventListener('input', _onInput);
  input.addEventListener('keydown', _onKeydown);
  clear.addEventListener('click', _onClear);

  document.addEventListener('click', (e) => {
    const wrapper = document.getElementById('logoPickerWrapper');
    if (wrapper && !wrapper.contains(e.target)) _closeDropdown();
  }, true);

}

export function setLogoPickerValue(url) {
  _customUrl = '';
  _selectedName = '';
  _highlightIdx = -1;
  if (url) {
    const m = url.match(/icons-static-svg[@/][^/]*\/icons\/([^/.]+)\.svg/);
    if (m) _selectedName = m[1];
    else _customUrl = url;
  }
  _syncUI();
}

export function getLogoPickerValue() {
  if (_selectedName) return _ICON_URL(_selectedName);
  return _customUrl;
}

async function _fetchAllIcons() {
  if ((_allIcons && _allIcons.length > 0) || _fetchPromise) return _allIcons || _fetchPromise;
  _fetchPromise = (async () => {
    try {
      const vr = await fetch(`https://data.jsdelivr.com/v1/packages/npm/${_ICON_PKG}/resolved?specifier=latest`);
      if (!vr.ok) throw new Error(vr.status);
      const { version } = await vr.json();
      if (!version) throw new Error('no version');
      const tr = await fetch(`https://data.jsdelivr.com/v1/packages/npm/${_ICON_PKG}@${version}?structure=tree`);
      if (!tr.ok) throw new Error(tr.status);
      const tree = await tr.json();
      const dir = _findDir(tree, 'icons');
      if (!dir?.files) throw new Error('no icons dir');
      _allIcons = dir.files.filter(f => f.name?.endsWith('.svg')).map(f => f.name.replace(/\.svg$/, '')).sort();
      return _allIcons;
    } catch (e) {
      console.warn('Logo picker: fetch failed:', e.message);
      _allIcons = [];
      return [];
    } finally { _fetchPromise = null; }
  })();
  return _fetchPromise;
}

function _findDir(node, name) {
  for (const e of (node.files || node.directories || [])) {
    if (e.name === name && e.files) return e;
    const s = _findDir(e, name);
    if (s) return s;
  }
  return null;
}

function _openDropdown() {
  if (_pickerOpen) return;
  _pickerOpen = true;
  _highlightIdx = -1;
  const dd = document.getElementById('logoPickerDropdown');
  dd.classList.add('open');
  _renderDropdown();
  setTimeout(() => document.getElementById('logoPickerInput')?.focus(), 50);
  if (!_allIcons) _fetchAllIcons().then(() => { if (_pickerOpen) _renderDropdown(); });
}

function _closeDropdown() {
  _pickerOpen = false;
  _highlightIdx = -1;
  document.getElementById('logoPickerDropdown')?.classList.remove('open');
}

function _renderDropdown() {
  const dd = document.getElementById('logoPickerDropdown');
  if (!dd || !_pickerOpen) return;
  const query = (document.getElementById('logoPickerInput')?.value || '').trim().toLowerCase();
  const popSet = new Set(_POPULAR_ICONS);
  const pop = [], other = [];
  for (const n of _POPULAR_ICONS) if (!query || n.includes(query)) pop.push(n);
  if (_allIcons?.length)
    for (const n of _allIcons) if (!popSet.has(n) && (!query || n.includes(query))) other.push(n);
  const MAX = 80, capped = other.slice(0, MAX);

  let html = '';
  let idx = 0;
  if (pop.length) {
    html += '<div class="lp-section-label">Popular</div><div class="lp-grid">';
    for (const n of pop) {
      const cls = (n === _selectedName ? ' lp-active' : '') + (idx === _highlightIdx ? ' lp-highlight' : '');
      html += `<div class="lp-item${cls}" data-icon="${n}" data-idx="${idx}"><img class="lp-thumb" src="${_ICON_URL(n)}" alt="" loading="lazy"><span class="lp-label">${_esc(n)}</span></div>`;
      idx++;
    }
    html += '</div>';
  }
  if (capped.length) {
    html += '<div class="lp-section-label">Other</div><div class="lp-grid">';
    for (const n of capped) {
      const cls = (n === _selectedName ? ' lp-active' : '') + (idx === _highlightIdx ? ' lp-highlight' : '');
      html += `<div class="lp-item${cls}" data-icon="${n}" data-idx="${idx}"><img class="lp-thumb" src="${_ICON_URL(n)}" alt="" loading="lazy"><span class="lp-label">${_esc(n)}</span></div>`;
      idx++;
    }
    html += '</div>';
    if (other.length > MAX) html += `<div class="lp-more">${other.length - MAX} more — type to filter</div>`;
  }
  if (!pop.length && !capped.length) {
    html += query
      ? `<div class="lp-empty">No icons matching "${_esc(query)}"</div><div class="lp-custom-hint">Press Enter to use as custom URL</div>`
      : '<div class="lp-empty">Loading icons…</div>';
  }

  // Keep search input, replace content after it
  const searchRow = dd.querySelector('.lp-search-row');
  const frag = document.createElement('div');
  frag.innerHTML = html;
  // Remove old content (everything after search row)
  while (searchRow.nextSibling) dd.removeChild(searchRow.nextSibling);
  while (frag.firstChild) dd.appendChild(frag.firstChild);

  for (const el of dd.querySelectorAll('.lp-item'))
    el.addEventListener('click', () => _selectIcon(el.dataset.icon));
}

function _selectIcon(name) {
  _selectedName = name;
  _customUrl = '';
  _closeDropdown();
  _syncUI();
}

function _syncUI() {
  const placeholder = document.getElementById('logoPickerPlaceholder');
  const img = document.getElementById('logoPickerPreviewImg');
  const clear = document.getElementById('logoPickerClear');
  const hidden = document.getElementById('provLogo');
  const input = document.getElementById('logoPickerInput');

  const url = _selectedName ? _ICON_URL(_selectedName) : _customUrl;
  if (url) {
    img.src = url;
    img.style.display = '';
    if (placeholder) placeholder.style.display = 'none';
    clear.style.display = '';
    if (hidden) hidden.value = url;
    if (input && !_pickerOpen) input.value = _selectedName || _customUrl;
  } else {
    img.src = '';
    img.style.display = 'none';
    if (placeholder) placeholder.style.display = '';
    clear.style.display = 'none';
    if (hidden) hidden.value = '';
    if (input && !_pickerOpen) input.value = '';
  }
}

function _onInput() {
  const val = (document.getElementById('logoPickerInput')?.value || '').trim();
  if (val.startsWith('http://') || val.startsWith('https://') || val.startsWith('data:')) {
    _selectedName = '';
    _customUrl = val;
    _syncUI();
    _closeDropdown();
    return;
  }
  _selectedName = '';
  _highlightIdx = -1;
  if (!_pickerOpen) _openDropdown();
  else {
    _renderDropdown();
    if (!_allIcons || !_allIcons.length) _fetchAllIcons().then(() => { if (_pickerOpen) _renderDropdown(); });
  }
}

function _onClear(e) {
  e.stopPropagation();
  _selectedName = '';
  _customUrl = '';
  _syncUI();
}

function _onKeydown(e) {
  if (!_pickerOpen) {
    if (e.key === 'ArrowDown' || e.key === 'Enter') { e.preventDefault(); _openDropdown(); }
    return;
  }
  const items = document.querySelectorAll('#logoPickerDropdown .lp-item');
  const count = items.length;
  if (e.key === 'ArrowDown') { e.preventDefault(); _highlightIdx = Math.min(_highlightIdx + 1, count - 1); _renderDropdown(); _scroll(); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); _highlightIdx = Math.max(_highlightIdx - 1, 0); _renderDropdown(); _scroll(); }
  else if (e.key === 'Enter') {
    e.preventDefault();
    if (_highlightIdx >= 0 && _highlightIdx < count) _selectIcon(items[_highlightIdx].dataset.icon);
    else {
      const val = document.getElementById('logoPickerInput')?.value?.trim();
      if (val) {
        const all = (_allIcons || []).concat(_POPULAR_ICONS);
        if (all.includes(val)) { _selectIcon(val); return; }
        _customUrl = val; _selectedName = ''; _closeDropdown(); _syncUI();
      }
    }
  } else if (e.key === 'Escape') { e.preventDefault(); _closeDropdown(); }
}

function _scroll() {
  document.getElementById('logoPickerDropdown')?.querySelector('.lp-highlight')?.scrollIntoView({ block: 'nearest' });
}
function _esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
