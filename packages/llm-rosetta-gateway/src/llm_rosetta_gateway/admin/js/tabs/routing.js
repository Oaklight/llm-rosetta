// routing.js — The Routing panel: who serves a model, and why.
//
// Every number on this page comes from `GET /admin/api/routing/<model>`; the
// browser does not re-derive a share, a state or a block end.  The panel
// writes the model's routing block through the same endpoint family
// (`PUT /admin/api/routing/<model>`), which rebuilds the config — so a value
// the parser refuses comes back as the parser's own message.
import { S } from '../core/state.js';
import { t } from '../core/i18n.js';
import { api, esc, showToast } from '../core/utils.js';
import { ROUTING_PRESETS, matchPreset, presetBlock } from '../core/routing-presets.js';
import { laneGripHtml, laneSwitchHtml } from '../components/provider-row.js';
import { loadConfig } from './providers.js';

let _data = null;
let _timer = null;
// The model the panel is showing.  Empty means nothing is picked yet: the
// list on the left is the selector, and the detail pane says so.
let _model = '';
// The routable models, from the gateway rather than the config document: a
// model whose provider was skipped at load has nothing to route to.
let _modelList = null;
// One line per model for the list: lane count, blocked lanes, their relative
// shares and the requests the window counted.
let _summary = {};
// Whether the log answered.  When it did not, every row's count is zero
// because nothing was read, not because nothing ran.
let _activityKnown = true;
// The window the list counts traffic over.  Five hours is the default: the
// tab answers "what is being routed now", and a model that served last week
// is not what an operator opens it to look at.
let _window = '5h';
// Which quota window each lane shows.  The cell shows the tightest by
// default; a click walks the vendor's windows so a 0.00 can be traced to the
// window that is actually spent.
const _windowChoice = {};
// The lane a drag started on, and the row it is over.  A drag never writes
// until it is dropped, so an abandoned drag changes nothing.
let _dragLane = null;
// The lane a switch just flipped, and the row index it held when clicked.
// The payload sorts candidates first, so without this the row the operator
// switched off jumps to the tail — off screen when the route is long, which
// is the same "where did it go" the provider list had.
let _pinnedLane = null;
// Which pane the detail column shows, and which decision row is open.
let _pane = 'policy';
let _openDecision = null;

/** The models the list offers. */
function _models() {
  if (_modelList) return _modelList;
  return Object.keys((S.configData && S.configData.models) || {}).sort();
}

function _currentModel() {
  return _model || '';
}

/** Show a model in the list without loading it. */
function _setModel(name) {
  _model = name || '';
}

/* ── The model list: the selector is the list itself ───────────────────── */

/** The models the search box currently lets through. */
function _visibleModels() {
  const input = document.getElementById('routingModelSearch');
  const needle = String((input && input.value) || '').trim().toLowerCase();
  return _models().filter((name) => !needle || name.toLowerCase().includes(needle));
}

function _renderModelList() {
  const el = document.getElementById('routingModelList');
  if (!el) return;
  const names = _visibleModels();
  if (!names.length) {
    el.innerHTML = '<div class="routing-model-empty">' + esc(t('routing.noModels')) + '</div>';
    return;
  }
  el.innerHTML = names.map((name) => {
    const row = _summary[name] || {};
    const shares = Array.isArray(row.shares) ? row.shares : [];
    const total = shares.reduce((sum, value) => sum + Math.max(0, Number(value) || 0), 0) || 1;
    const lanes = Number(row.providers) || 0;
    const blocked = Number(row.blocked) || 0;
    const requests = Number(row.requests) || 0;
    const percentages = shares.map((value) =>
      Math.round((Math.max(0, Number(value) || 0) / total) * 100)
    );
    // The stack is the board's share column in miniature: a lane that is not
    // serving has no width and is drawn muted, the way the bar does it.  A
    // zero-share lane keeps a 2 px mark so its existence is still visible —
    // a 0 % slice leaves no trace, and the bar then reads as one lane's.
    // The bar alone cannot say which lane the width belongs to, so the title
    // names every lane and its share.
    const lanesInfo = Array.isArray(row.lanes) ? row.lanes : [];
    const stack = shares.map((value, index) => {
      const share = Math.max(0, Number(value) || 0);
      const zero = share <= 0;
      const width = zero ? '2px' : ((share / total) * 100).toFixed(1) + '%';
      const lane = lanesInfo[index];
      // The slice takes the lane's state, the way the board's dot does: amber
      // for a lane that is only sunk, red for one that cannot serve at all.
      const tone = !lane || lane.state === 'serving'
        ? ''
        : (lane.state === 'sunk' ? 'is-warn' : 'is-bad');
      const classes = [tone, zero ? 'mute is-zero' : ''].filter(Boolean).join(' ');
      // Each slice carries its own lane, so hovering the 2 px mark of a
      // blocked lane says which lane it is and what is wrong with it.
      const label = lane
        ? ' title="' + esc(
          lane.name + ' ' + Math.round((share / total) * 100) + '%' +
          (tone ? ' · ' + _stateLabel({ state: lane.state }) : '')
        ) + '"'
        : '';
      return '<i class="' + classes + '"' + label +
        ' style="width:' + width + '"></i>';
    }).join('');
    const stackTitle = lanesInfo
      .map((lane) =>
        lane.name + ' ' +
        Math.round((Math.max(0, Number(lane.share) || 0) / total) * 100) + '%'
      )
      .join(' · ');
    // The count is where an operator asks "how many, and are they well?" —
    // the answer belongs on it, one lane per line, rather than on the slices
    // they would have to hunt through one at a time.
    // A lane a ladder holds out says when it comes back: "rate limited" alone
    // makes the reader open the model to find out how long.
    const laneTitle = lanesInfo
      .map((lane) =>
        lane.name + ' · ' +
        Math.round((Math.max(0, Number(lane.share) || 0) / total) * 100) + '% · ' +
        _stateLabel({ state: lane.state }) +
        (lane.until && lane.state === 'rate_limited'
          ? ' · ' + new Date(lane.until).toLocaleTimeString()
          : '')
      )
      .join('\n');
    return (
      '<div class="routing-model-item' + (name === _model ? ' is-selected' : '') +
      '" role="option" aria-selected="' + (name === _model) + '" tabindex="0"' +
      ' data-model="' + esc(name) + '">' +
      '<div class="routing-model-row1"><b title="' + esc(name) + '">' + esc(name) + '</b>' +
      '<span class="routing-model-count">' +
      esc(_activityKnown
        ? t('routing.modelRequests', { count: requests })
        : t('routing.activityUnknown')) + '</span></div>' +
      (lanes > 1 ? '<div class="routing-model-stack" title="' + esc(stackTitle) + '">' + stack + '</div>' : '') +
      '<div class="routing-model-row1 is-meta"><span class="routing-model-count' +
      (lanesInfo.length ? ' is-laned' : '') +
      '"' + (lanesInfo.length ? ' title="' + esc(laneTitle) + '"' : '') + '>' +
      esc(t(lanes === 1 ? 'routing.modelLane' : 'routing.modelLanes', { count: lanes })) +
      (lanes > 1 ? ' · ' + percentages.map((pct) => pct + '%').join(' · ') : '') +
      '</span>' +
      (blocked
        ? '<span class="routing-model-count is-blocked">' +
          esc(t('routing.modelBlocked', { count: blocked })) + '</span>'
        : '') +
      '</div></div>'
    );
  }).join('');
  el.querySelectorAll('.routing-model-item').forEach((item) => {
    const pick = () => loadRouting(item.dataset.model);
    item.addEventListener('click', pick);
    item.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        pick();
      }
    });
  });
}

/** Wire the search box and the window switch once; both survive a render. */
function _wireList() {
  const input = document.getElementById('routingModelSearch');
  if (input && !input.dataset.wired) {
    input.dataset.wired = '1';
    input.addEventListener('input', _renderModelList);
    input.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter') return;
      const pick = _visibleModels()[0];
      if (pick) loadRouting(pick);
    });
  }
  const chips = document.getElementById('routingWindows');
  if (chips && !chips.dataset.wired) {
    chips.dataset.wired = '1';
    chips.querySelectorAll('[data-window]').forEach((chip) => {
      const choose = () => _setWindow(chip.dataset.window);
      chip.addEventListener('click', choose);
      chip.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        choose();
      });
    });
  }
}

/** Count one window's traffic in the list, and remember the choice. */
async function _setWindow(window) {
  if (!window || window === _window) return;
  _window = window;
  document.querySelectorAll('#routingWindows [data-window]').forEach((chip) => {
    const active = chip.dataset.window === window;
    chip.classList.toggle('active', active);
    chip.setAttribute('aria-pressed', String(active));
  });
  await _loadRoster();
}

/** Fetch the roster and its per-model summary for the current window. */
async function _loadRoster() {
  try {
    const answer = await api.get(
      '/admin/api/routing?window=' + encodeURIComponent(_window)
    );
    _modelList = answer.models || null;
    _summary = answer.summary || {};
    _activityKnown = answer.activity_known !== false;
  } catch {
    /* the config document's keys remain as a fallback */
  }
  _renderModelList();
}

/* ── The board ─────────────────────────────────────────────────────────── */

/** Seconds until an ISO timestamp, never negative. */
function _secondsLeft(iso) {
  if (!iso) return null;
  const ms = new Date(iso).getTime() - Date.now();
  return ms > 0 ? Math.ceil(ms / 1000) : 0;
}

function _stateLabel(provider) {
  if (provider.state === 'serving') return t('routing.state.serving');
  if (provider.state === 'sunk') return t('routing.state.sunk');
  if (provider.state === 'off') return t('routing.state.off');
  const key = 'routing.state.' + provider.state;
  const label = t(key);
  return label === key ? provider.state : label;
}

function _stateTone(provider) {
  if (provider.state === 'serving') return 'ok';
  if (provider.state === 'sunk') return 'warn';
  return 'fail';
}

/** Where a lane actually goes: its wire format and the shim supplying it.
 *  A lane the route does not carry says why instead: it is not going anywhere.
 *  The server names the switch that is off; a gateway-disabled lane carries
 *  its own diagnostic, which is the sentence worth showing. */
function _laneSub(provider) {
  if (provider.excluded) {
    if (provider.reason_kind === 'auto' && provider.reason) return provider.reason;
    return t(provider.reason_kind === 'entry'
      ? 'routing.reason.entry'
      : 'routing.reason.off');
  }
  const parts = [provider.format, provider.shim || provider.upstream_model].filter(Boolean);
  return parts.join(' · ');
}

/** A bar whose fill is `fraction` of its width (0..1). */
function _bar(fraction, tone) {
  const pct = Math.max(0, Math.min(1, Number(fraction) || 0)) * 100;
  return (
    '<span class="routing-bar' + (tone ? ' is-' + tone : '') + '">' +
    '<i style="width:' + pct.toFixed(1) + '%"></i></span>'
  );
}

/** The model's total share, for the percentage column. */
function _shareTotal() {
  return Math.max(
    1e-9,
    ((_data && _data.providers) || []).reduce(
      (sum, entry) => sum + Math.max(0, Number(entry.share) || 0),
      0
    )
  );
}

function _sharePct(provider) {
  return Math.round((Math.max(0, Number(provider.share) || 0) / _shareTotal()) * 100);
}

/** The badge that says why a lane is not simply serving. */
function _statePill(provider) {
  if (provider.state === 'blocked' || provider.state === 'rate_limited') {
    const hits = (provider.block && provider.block.refusals) || null;
    return '<span class="badge badge-error">' +
      esc(hits ? t('routing.pill.blocked', { count: hits }) : t('routing.state.blocked')) +
      '</span>';
  }
  if (provider.state === 'sunk') {
    return '<span class="badge badge-warning">' + esc(t('routing.pill.sunk')) + '</span>';
  }
  if (provider.state === 'spent') {
    return '<span class="badge">' + esc(t('routing.pill.spent')) + '</span>';
  }
  return '';
}

/** The payload order, with the lane just toggled back where it was.
 *
 *  The payload puts candidates first and an off lane last, which is the right
 *  reading order but the wrong thing to do under the pointer: the row the
 *  operator flipped would leave the viewport.  One lane is pinned, not the
 *  whole list, so every other change still sorts normally. */
function _withPinnedLane(lanes) {
  const items = lanes.slice();
  if (!_pinnedLane) return items;
  const from = items.findIndex((entry) => entry.name === _pinnedLane.name);
  if (from === -1) return items;
  const [row] = items.splice(from, 1);
  items.splice(Math.min(_pinnedLane.index, items.length), 0, row);
  return items;
}

/** The board: one row per candidate, in the model's declared order. */
function _renderBoard() {
  const el = document.getElementById('routingBoard');
  if (!el || !_data) return;
  if (!_data.providers.length) {
    el.innerHTML = '<div class="conn-row"><span class="conn-warn"></span><span class="conn-detail">' +
      esc(t('routing.noProviders')) + '</span></div>';
    return;
  }
  const head =
    '<div class="routing-route-row is-head">' +
    '<span>' + esc(t('routing.col.pos')) + '</span>' +
    '<span>' + esc(t('routing.col.provider')) + '</span>' +
    '<span>' + esc(t('routing.col.weight')) + '</span>' +
    '<span>' + esc(t('routing.col.allowance')) + '</span>' +
    '<span>' + esc(t('routing.col.share')) + '</span>' +
    '<span>' + esc(t('routing.col.state')) + '</span>' +
    '</div>';
  // Drag and weights act on the lanes the route carries: an excluded lane has
  // no share to reorder or weight, and its row stays where it is.
  const included = _data.providers.filter((provider) => !provider.excluded);
  const editable = included.length > 1;
  const rows = _withPinnedLane(_data.providers).map((provider, index) => {
    const excluded = !!provider.excluded;
    const left = _secondsLeft(provider.until);
    const until = left === null
      ? ''
      : ' · <span class="routing-countdown" data-until="' + esc(provider.until) + '">' +
        esc(t('routing.secondsLeft', { seconds: left })) + '</span>';
    const headroom = provider.headroom === null || provider.headroom === undefined
      ? null
      : Number(provider.headroom);
    const windows = Array.isArray(provider.windows) ? provider.windows : [];
    const choice = Math.min(
      Math.max(0, Number(_windowChoice[provider.name]) || 0),
      Math.max(0, windows.length - 1)
    );
    const shown = windows.length ? windows[choice] : null;
    const windowTitle = windows.map((entry) =>
      t('routing.windowLine', {
        name: entry.name,
        percent: Math.round(Number(entry.remaining) * 100),
        reset: entry.resets_at ? new Date(entry.resets_at).toLocaleTimeString() : '—',
      })
    ).join('\n');
    // A lane with no probe, and one whose probe has not answered yet, both
    // have no headroom; the cell says which, in as few words as it can.
    // Nothing left is drawn as a hollow track: a 0 % fill on a solid track
    // reads as a full bar, which is the opposite of what it says.
    const remaining = headroom === null
      ? null
      : (shown ? Number(shown.remaining) : headroom);
    const allowance = remaining === null
      ? '<small class="routing-sub routing-sub-muted">' +
        esc(t(provider.probe ? 'routing.allowance.waiting' : 'routing.allowance.noProbe')) +
        '</small>'
      : _bar(
          remaining,
          remaining <= 0 ? 'empty' : (provider.state === 'serving' ? '' : 'dim')
        ) +
        '<small class="routing-sub">' +
        esc(t('routing.headroom', { value: remaining.toFixed(2) })) +
        (shown ? ' · ' + esc(shown.name) : '') + '</small>';
    // The tooltip sits on the cell, not on the sub-line, so hovering the bar
    // itself says which windows the figure came from.
    const allowanceCell = excluded
      ? '<small class="routing-sub routing-sub-muted">—</small>'
      : allowance;
    const allowanceTitle = excluded
      ? t('routing.reason.off')
      : (headroom === null
        ? t('routing.allowance.hint') + (windowTitle ? '\n' + windowTitle : '')
        : (windowTitle ? windowTitle + '\n' : '') + t('routing.allowance.cycleHint'));
    return (
      '<div class="routing-route-row' + (excluded ? ' is-excluded' : '') +
      (!excluded && editable ? ' is-draggable' : '') + '"' +
      (!excluded && editable ? ' draggable="true" data-lane="' + esc(provider.name) + '"' : '') + '>' +
      '<span class="routing-pos">' +
      (!excluded && editable
        ? laneGripHtml({ title: t('routing.gripHint'), cls: 'routing-grip' })
        : '') +
      (excluded ? '—' : index + 1) + '</span>' +
      '<span class="routing-who">' +
      laneSwitchHtml({
        name: provider.name,
        on: !excluded,
        title: excluded ? t('routing.enableLane') : t('routing.disableLane'),
        data: 'data-lane="' + esc(provider.name) + '"',
        extraClass: 'routing-switch',
      }) +
      '<span class="routing-lane-id"><b>' + esc(provider.name) + '</b>' +
      '<small title="' + esc(_laneSub(provider)) + '">' + esc(_laneSub(provider)) + '</small></span></span>' +
      '<span class="routing-stepper">' +
      // A weight only means something between lanes: a model with one
      // provider always serves 100%, and the config refuses the edit.
      (!excluded && editable
        ? '<button type="button" aria-label="' + esc(t('routing.weightLower')) + '"' +
          ' data-weight="-1" data-provider="' + esc(provider.name) + '">−</button>' +
          '<span>' + esc(String(provider.weight)) + '</span>' +
          '<button type="button" aria-label="' + esc(t('routing.weightRaise')) + '"' +
          ' data-weight="1" data-provider="' + esc(provider.name) + '">+</button>'
        : '<span class="routing-weight-static">' + esc(String(provider.weight)) + '</span>') +
      '</span>' +
      '<span class="routing-allowance' + (headroom === null ? ' is-empty' : '') +
      (windows.length > 1 ? ' is-cyclable' : '') + '"' +
      ' title="' + esc(allowanceTitle) + '"' +
      (windows.length > 1 ? ' data-windows="' + esc(provider.name) + '"' : '') + '>' +
      allowanceCell + '</span>' +
      '<span class="routing-share">' + (excluded ? '—' : _sharePct(provider) + '%') + '</span>' +
      '<span class="routing-state">' +
      '<span class="conn-' + _stateTone(provider) + '"></span>' +
      esc(_stateLabel(provider)) + until +
      ' ' + _statePill(provider) + '</span>' +
      '</div>'
    );
  });
  // The board's rows say which lane is held out; this says the route is, so a
  // single-lane model does not need the reader to notice a red dot.
  const laneBlocks = _data.lane_blocks || {};
  const limited = Number(laneBlocks.count) || 0;
  const banner = limited
    ? '<div class="routing-lane-blocks badge badge-error">' +
      esc(laneBlocks.until
        ? t(limited === 1 ? 'routing.laneBlockUntil' : 'routing.laneBlocksUntil', {
          count: limited,
          time: new Date(laneBlocks.until).toLocaleTimeString(),
        })
        : t(limited === 1 ? 'routing.laneBlock' : 'routing.laneBlocks', {
          count: limited,
        })) +
      '</div>'
    : '';
  el.innerHTML = banner + '<div class="routing-route">' + head + rows.join('') + '</div>';
  // The switch writes the provider's own entry; a lane switched off here
  // leaves the route the same way it would from the Providers page.
  el.querySelectorAll('.routing-switch input').forEach((input) => {
    input.addEventListener('change', () => _toggleLane(input.dataset.lane));
  });
  // A model name is arbitrary text, so the weight buttons carry their
  // provider in a data attribute and a listener reads it.
  el.querySelectorAll('.routing-allowance.is-cyclable').forEach((cell) => {
    cell.addEventListener('click', () => {
      const name = cell.dataset.windows;
      const lane = (_data.providers || []).find((entry) => entry.name === name);
      const count = lane && Array.isArray(lane.windows) ? lane.windows.length : 0;
      if (count < 2) return;
      _windowChoice[name] = ((Number(_windowChoice[name]) || 0) + 1) % count;
      _renderBoard();
    });
  });
  el.querySelectorAll('.routing-stepper button').forEach((button) => {
    button.addEventListener('click', () =>
      _stepWeight(button.dataset.provider, Number(button.dataset.weight))
    );
  });
  _wireDrag(el);
  _renderLegend();
}

/** Wire the rows of a multi-provider board for reordering.

    The order is the route: an ``ordered`` model serves from the first
    eligible lane, and every other strategy breaks a tie the same way.  A drag
    writes the config list, then the board reloads from what the config
    accepted, so the row a lane lands on is the row the selector reads.
 */
function _wireDrag(el) {
  const rows = [...el.querySelectorAll('.routing-route-row[draggable="true"]')];
  if (!rows.length) return;
  rows.forEach((row) => {
    row.addEventListener('dragstart', (event) => {
      _dragLane = row.dataset.lane;
      row.classList.add('is-dragging');
      event.dataTransfer.effectAllowed = 'move';
      // Firefox starts a drag only when some data is set.
      event.dataTransfer.setData('text/plain', _dragLane);
    });
    row.addEventListener('dragend', () => {
      _dragLane = null;
      rows.forEach((other) => {
        other.classList.remove('is-dragging', 'is-drop-before', 'is-drop-after');
      });
    });
    row.addEventListener('dragover', (event) => {
      if (!_dragLane || row.dataset.lane === _dragLane) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = 'move';
      const box = row.getBoundingClientRect();
      const after = event.clientY > box.top + box.height / 2;
      rows.forEach((other) => {
        other.classList.toggle('is-drop-before', other === row && !after);
        other.classList.toggle('is-drop-after', other === row && after);
      });
    });
    row.addEventListener('dragleave', () => {
      row.classList.remove('is-drop-before', 'is-drop-after');
    });
    row.addEventListener('drop', (event) => {
      event.preventDefault();
      const after = row.classList.contains('is-drop-after');
      const from = _dragLane;
      rows.forEach((other) => {
        other.classList.remove('is-dragging', 'is-drop-before', 'is-drop-after');
      });
      _dragLane = null;
      if (!from || from === row.dataset.lane) return;
      _reorderLane(from, row.dataset.lane, after);
    });
  });
}

/** Move *lane* beside *target* and write the order the panel now shows. */
async function _reorderLane(lane, target, after) {
  // Only the lanes the route carries: an excluded lane keeps the slot it
  // holds in the document, which is what the endpoint's short list means.
  const names = (_data.providers || [])
    .filter((entry) => !entry.excluded)
    .map((entry) => entry.name);
  if (!names.includes(lane) || !names.includes(target)) return;
  names.splice(names.indexOf(lane), 1);
  const at = names.indexOf(target) + (after ? 1 : 0);
  names.splice(at, 0, lane);
  const model = _currentModel();
  const res = await api.post(
    '/admin/api/config/models/' + encodeURIComponent(model) + '/reorder-providers',
    { providers: names }
  );
  if (!res.ok) {
    showToast(res.error || t('toast.error'), 'error');
    return;
  }
  await loadConfig();
  await loadRouting(model);
}

/** How this model's Share column is made: the strategy decides the formula. */
function _shareFormula() {
  const strategy = (_data && _data.routing && _data.routing.strategy) || 'weighted';
  const key = 'routing.legend.formula.' + strategy;
  const text = t(key);
  return text === key ? t('routing.legend.formula') : text;
}

/** The legend under the board: what the dots mean and how a share is made. */
function _renderLegend() {
  const el = document.getElementById('routingLegend');
  if (!el || !_data) return;
  el.innerHTML =
    '<span class="routing-legend-item"><span class="conn-ok"></span>' +
    esc(t('routing.legend.serving')) + '</span>' +
    '<span class="routing-legend-item"><span class="conn-warn"></span>' +
    esc(t('routing.legend.sunk')) + '</span>' +
    '<span class="routing-legend-item"><span class="conn-fail"></span>' +
    esc(t('routing.legend.blocked')) + '</span>' +
    '<span class="routing-legend-formula">' + esc(_shareFormula()) + '</span>';
}

/** Flip one provider's switch and repaint the board from the new config.
 *
 *  The write goes to the provider's entry, the same call the Providers page
 *  makes, so the two pages cannot disagree about what "off" means. */
async function _toggleLane(name) {
  // The row the switch sits on, read before the write: the payload that comes
  // back sorts an off lane after the candidates, and the row must not move.
  const index = _data
    ? _withPinnedLane(_data.providers).findIndex((entry) => entry.name === name)
    : -1;
  const lane = _data ? (_data.providers || []).find((entry) => entry.name === name) : null;
  // A lane can be off through the model's own entry, which the provider
  // switch does not touch: write the entry when that is the switch that is off.
  const entryOff = !!(lane && lane.excluded && lane.reason_kind === 'entry');
  const res = entryOff
    ? await api.post(
        '/admin/api/config/models/' + encodeURIComponent(_model) + '/update-provider',
        { provider: name, enabled: true }
      )
    : await api.post(
        '/admin/api/config/providers/' + encodeURIComponent(name) + '/toggle'
      );
  if (!res.ok) {
    showToast(res.error || t('toast.error'), 'error');
    return;
  }
  const on = entryOff || !!res.enabled;
  showToast(name + ': ' + t(on ? 'provider.enabled' : 'provider.disabled'));
  await loadConfig();
  _pinnedLane = index === -1 ? null : { name, index };
  if (_model) await loadRouting(_model);
}

/** Raise or lower one lane's weight, then show what the config accepted. */
async function _stepWeight(provider, delta) {
  const model = _currentModel();
  if (!model || !provider) return;
  const lane = (_data.providers || []).find((entry) => entry.name === provider);
  const next = Math.max(1, Math.min(100, Number(lane ? lane.weight : 1) + delta));
  if (lane && next === lane.weight) return;
  try {
    await api.post(
      '/admin/api/config/models/' + encodeURIComponent(model) + '/update-provider',
      { provider, weight: next }
    );
    await loadRouting(model);
  } catch (e) {
    showToast(String(e.message || e), 'error');
  }
}

/** Shorten an id the way the board does: first 4 … last 3. */
function _shortId(value) {
  const text = String(value || '');
  if (text.length <= 10) return text || '—';
  return text.slice(0, 4) + '…' + text.slice(-3);
}

/** The first product token of a User-Agent: enough to name the agent. */
function _shortUa(value) {
  const text = String(value || '').trim();
  const token = text.split(/[\s(]/)[0];
  return token.length > 28 ? token.slice(0, 27) + '…' : token;
}

function _latency(ms) {
  if (ms === null || ms === undefined) return '—';
  const value = Number(ms);
  return value >= 1000 ? (value / 1000).toFixed(1) + ' s' : Math.round(value) + ' ms';
}

/** The decision stream: the request log's last rows for this model. */
function _renderDecisions() {
  const el = document.getElementById('routingDecisions');
  const count = document.getElementById('routingDecisionsCount');
  if (!el || !_data) return;
  if (count) {
    count.textContent = _data.decisions.length
      ? t('routing.decisionsCount', { count: _data.decisions.length })
      : '';
  }
  if (!_data.decisions.length) {
    el.innerHTML = '<div class="conn-row"><span class="conn-warn"></span><span class="conn-detail">' +
      esc(t('routing.noDecisions')) + '</span></div>';
    return;
  }
  const head =
    '<div class="routing-dec-row is-head">' +
    '<span>' + esc(t('routing.col.time')) + '</span>' +
    '<span>' + esc(t('routing.col.request')) + '</span>' +
    '<span>' + esc(t('routing.col.client')) + '</span>' +
    '<span>' + esc(t('routing.col.chosen')) + '</span>' +
    '<span>' + esc(t('routing.col.reason')) + '</span>' +
    '<span>' + esc(t('routing.col.status')) + '</span>' +
    '<span>' + esc(t('routing.col.latency')) + '</span>' +
    '</div>';
  const rows = _data.decisions.map((decision, index) => {
    const when = decision.time ? new Date(decision.time).toLocaleTimeString() : '—';
    const status = decision.status === null || decision.status === undefined
      ? '—'
      : String(decision.status);
    const tone = Number(decision.status) >= 400 ? 'fail' : 'ok';
    const reason = decision.failover
      ? '<span class="badge badge-warning">' +
        esc(t('routing.failover', { from: decision.failover.from || '?' })) + '</span>'
      : '<span class="routing-muted">—</span>';
    // Which lane served, when one provider is several upstream models (#1430):
    // the provider name alone cannot tell the cheap lane from the fast one.
    const lane = decision.upstream_model
      ? '<small class="routing-muted">' + esc(decision.upstream_model) + '</small>'
      : '';
    const key = String(index);
    // A click opens the chain under the row: who asked, what the gateway did,
    // and why the lane was the one chosen.
    const open = _openDecision === key;
    const detail = open
      ? '<div class="routing-dec-detail">' +
        '<div class="routing-dec-chain">' +
        esc(decision.client || '—') + ' → gateway → ' + esc(decision.provider || '—') +
        (decision.upstream_model ? '/' + esc(decision.upstream_model) : '') +
        ' · ' + esc(_strategyName()) + '</div>' +
        '<div class="routing-dec-why">' +
        esc(t('routing.decisionWhy', {
          reason: decision.failover
            ? t('routing.failover', { from: decision.failover.from || '?' })
            : t('routing.decisionAnchor'),
          status,
          latency: _latency(decision.duration_ms),
        })) + '</div></div>'
      : '';
    return (
      '<div class="routing-dec-row is-openable' + (open ? ' is-open' : '') +
      '" role="button" tabindex="0" data-decision="' + key + '">' +
      '<span class="routing-mono">' + esc(when) + '</span>' +
      '<span class="routing-mono">' + esc(_shortId(decision.request_id)) + '</span>' +
      '<span class="routing-mono routing-client" title="' + esc(decision.user_agent || '') + '">' +
      esc(decision.client || '—') +
      (decision.user_agent ? '<small>' + esc(_shortUa(decision.user_agent)) + '</small>' : '') +
      '</span>' +
      '<span class="routing-lane">' + esc(decision.provider || '—') + lane + '</span>' +
      '<span>' + reason + '</span>' +
      '<span class="routing-status is-' + tone + '">' + esc(status) + '</span>' +
      '<span class="routing-mono">' + esc(_latency(decision.duration_ms)) + '</span>' +
      '</div>' + detail
    );
  });
  el.innerHTML = '<div class="routing-dec">' + head + rows.join('') + '</div>';
  el.querySelectorAll('[data-decision]').forEach((row) => {
    const toggle = () => {
      _openDecision = _openDecision === row.dataset.decision ? null : row.dataset.decision;
      _renderDecisions();
    };
    row.addEventListener('click', toggle);
    row.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        toggle();
      }
    });
  });
}

/** The model's strategy word, for the decision chain line. */
function _strategyName() {
  return (_data && _data.routing && _data.routing.strategy) || 'weighted';
}

/** The sessions table: one line per conversation and the lane it last reached.
 *
 *  A conversation is identified by the id the gateway derives from the client
 *  and the request's first message, so concurrent sessions behind one key
 *  stay apart.  A line with no id — an older row, or a request with nothing to
 *  derive one from — is the client identity, marked as such.  Two
 *  conversations that share a client and a first message derive the same id;
 *  the note states the count rather than claiming it exact.
 */
function _renderSessions() {
  const el = document.getElementById('routingSessions');
  const note = document.getElementById('routingSessionsNote');
  if (!el || !_data) return;
  const sessions = Array.isArray(_data.sessions) ? _data.sessions : [];
  const count = Number.isInteger(_data.session_count) ? _data.session_count : sessions.length;
  const sample = Number.isInteger(_data.session_sample) ? _data.session_sample : 0;
  if (note) {
    const affinity = t(
      _strategyName() && _data.routing && _data.routing.affinity === 'session'
        ? 'routing.sessionsSticky'
        : 'routing.sessionsFree'
    );
    note.textContent = sessions.length
      ? affinity + ' · ' + t('routing.sessionsCount', { count, sample })
      : affinity;
  }
  if (!sessions.length) {
    el.innerHTML = '<div class="conn-row"><span class="conn-warn"></span><span class="conn-detail">' +
      esc(t('routing.noSessions')) + '</span></div>';
    return;
  }
  const head =
    '<div class="routing-dec-row is-head routing-session-row">' +
    '<span>' + esc(t('routing.col.client')) + '</span>' +
    '<span>' + esc(t('routing.col.chosen')) + '</span>' +
    '<span>' + esc(t('routing.col.requests')) + '</span>' +
    '<span>' + esc(t('routing.col.lastSeen')) + '</span>' +
    '</div>';
  const rows = sessions.map((session) => (
    '<div class="routing-dec-row routing-session-row">' +
    '<span class="routing-mono routing-client" title="' + esc(session.user_agent || '') + '">' +
    esc(session.client || '—') +
    (session.user_agent ? '<small>' + esc(_shortUa(session.user_agent)) + '</small>' : '') +
    (session.derived === false
      ? '<small class="routing-session-group">' + esc(t('routing.sessionsGroup')) + '</small>'
      : '') +
    '</span>' +
    '<span>' + esc(session.provider || '—') + '</span>' +
    '<span class="routing-mono">' + esc(String(session.requests || 0)) + '</span>' +
    '<span class="routing-mono">' +
    esc(session.last ? new Date(session.last).toLocaleTimeString() : '—') + '</span>' +
    '</div>'
  ));
  el.innerHTML = '<div class="routing-dec">' + head + rows.join('') + '</div>';
}

/** One rule as a line: its conditions, then what it does. */
function _ruleText(rule) {
  const when = Object.entries(rule.when || {})
    .map(([key, value]) => key + '=' + (Array.isArray(value) ? value.join(',') : value))
    .join(' ');
  const action = rule.use
    ? t('routing.rule.use', { provider: rule.use })
    : t('routing.rule.pause', { provider: rule.pause || '?' });
  return (when ? when + ' → ' : '') + action;
}

/** The policy editor, pre-filled from the block the gateway reported. */
function _renderPolicy() {
  const el = document.getElementById('routingPolicy');
  if (!el || !_data) return;
  const block = _data.routing || {};
  const strategyOptions = ['weighted', 'ordered', 'rotate', 'least_used', 'pace']
    .map((name) => '<option value="' + name + '"' +
      (block.strategy === name ? ' selected' : '') + '>' + name + '</option>')
    .join('');
  const affinityOptions = ['off', 'session']
    .map((name) => '<option value="' + name + '"' +
      (block.affinity === name ? ' selected' : '') + '>' + name + '</option>')
    .join('');
  const firstToken = block.first_token_ms === null || block.first_token_ms === undefined
    ? ''
    : String(block.first_token_ms);
  const firstTokenChoices = [
    ['', t('routing.firstToken.off')],
    ['10000', t('routing.firstToken.next', { seconds: 10 })],
    ['20000', t('routing.firstToken.next', { seconds: 20 })],
    ['30000', t('routing.firstToken.next', { seconds: 30 })],
  ];
  // A value the config carries but the list does not offer still needs an
  // option: otherwise the browser selects the first one and the panel shows
  // "off" for a model that gates on five seconds.
  if (firstToken && !firstTokenChoices.some(([value]) => value === firstToken)) {
    firstTokenChoices.push([firstToken, t('routing.firstToken.custom', { ms: firstToken })]);
  }
  const firstTokenOptions = firstTokenChoices.map(([value, label]) =>
    '<option value="' + value + '"' +
    (firstToken === value ? ' selected' : '') + '>' + esc(label) + '</option>').join('');
  const preset = matchPreset(block);
  const presets = ROUTING_PRESETS.map((entry) =>
    '<div class="routing-preset' + (preset === entry.id ? ' active' : '') +
    '" role="button" tabindex="0" aria-pressed="' + (preset === entry.id) + '"' +
    ' title="' + esc(t(entry.hint)) + '"' +
    ' data-preset="' + esc(entry.id) + '">' + esc(t(entry.label)) + '</div>'
  ).join('');
  const rules = (block.rules || []).length
    ? '<ul class="routing-rules">' + block.rules.map((rule) =>
        '<li>' + esc(_ruleText(rule)) + '</li>').join('') + '</ul>'
    : '<ul class="routing-rules"><li class="is-none">' +
      esc(t('routing.rules.none')) + '</li></ul>';
  el.innerHTML =
    '<h3 class="routing-policy-title">' +
    esc(t('routing.policy.title', { model: _data.model })) + '</h3>' +
    '<div class="facet-chips routing-presets">' + presets + '</div>' +
    '<div class="routing-field-row">' +
    '<label>' + esc(t('routing.strategy')) +
    '<select id="routingStrategy">' + strategyOptions + '</select></label>' +
    '<label>' + esc(t('routing.affinity')) +
    '<select id="routingAffinity">' + affinityOptions + '</select></label>' +
    '</div>' +
    '<div class="routing-field-row">' +
    '<label class="routing-check"><input type="checkbox" id="routingSink"' +
    (block.sink === false ? '' : ' checked') + '> ' + esc(t('routing.sink')) + '</label>' +
    '<label>' + esc(t('routing.firstToken')) +
    '<select id="routingFirstToken">' + firstTokenOptions + '</select></label>' +
    '</div>' +
    '<div class="routing-rules-head">' + esc(t('routing.rules')) + '</div>' +
    '<div class="field-hint">' + esc(t('routing.rules.hint')) + '</div>' + rules +
    '<button type="button" class="btn btn-primary btn-sm routing-save" onclick="saveRouting()">' +
    esc(t('routing.save')) + '</button>' +
    '<div class="routing-rules-head">' + esc(t('routing.json')) + '</div>' +
    '<pre class="routing-json">' + esc(JSON.stringify(block, null, 2)) + '</pre>';
  el.querySelectorAll('.routing-preset').forEach((chip) => {
    chip.addEventListener('click', () => applyRoutingPreset(chip.dataset.preset));
    chip.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        applyRoutingPreset(chip.dataset.preset);
      }
    });
  });
}

export function renderRouting() {
  if (!_data) return;
  _renderBoard();
  _renderDecisions();
  _renderSessions();
  _renderPolicy();
  _showPane();
  _startTicking();
}

/** Show one of the detail column's panes, and mark its tab. */
function _showPane() {
  ['policy', 'decisions', 'sessions'].forEach((name) => {
    const pane = document.getElementById(
      'routingPane' + name.charAt(0).toUpperCase() + name.slice(1)
    );
    if (pane) pane.hidden = name !== _pane;
  });
  document.querySelectorAll('#routingPanes [data-pane]').forEach((button) => {
    const active = button.dataset.pane === _pane;
    button.setAttribute('aria-selected', String(active));
    button.classList.toggle('active', active);
  });
}

/** Wire the pane tabs once; they survive every render. */
function _wirePanes() {
  const tabs = document.getElementById('routingPanes');
  if (!tabs || tabs.dataset.wired) return;
  tabs.dataset.wired = '1';
  tabs.querySelectorAll('[data-pane]').forEach((button) => {
    button.addEventListener('click', () => {
      _pane = button.dataset.pane;
      _showPane();
    });
  });
}

/** Update the countdowns; refetch once when a block ends. */
function _startTicking() {
  if (_timer) clearInterval(_timer);
  _timer = setInterval(() => {
    let expired = false;
    document.querySelectorAll('#tab-routing [data-until]').forEach((node) => {
      const left = _secondsLeft(node.dataset.until);
      if (left === null) return;
      if (left === 0) expired = true;
      node.textContent = t('routing.secondsLeft', { seconds: left });
    });
    if (expired && _model) loadRouting(_model);
  }, 1000);
}

export function stopRoutingTimer() {
  if (_timer) { clearInterval(_timer); _timer = null; }
}

export async function loadRouting(model) {
  // The tab can be the first one a deep link opens, before initApp's own
  // config fetch has landed; the list has no options until then.
  if (!_models().length) {
    try { await loadConfig(); } catch { /* the models tab reports it */ }
  }
  // A tab activation (no explicit model) refreshes the roster: the config may
  // have gained or lost a model, and the window's counts have moved on.
  if (!model) {
    await _loadRoster();
  }
  _wireList();
  _wirePanes();
  const models = _models();
  // The first open selects the top of the list, which is the model requested
  // most recently; a later visit keeps the model already on screen.
  const asked = model || _model || _visibleModels()[0] || '';
  // Entering the tab (no explicit model) or switching models re-sorts the
  // board; a pin only belongs to the toggle that was just clicked.
  if (!model || asked !== _model) _pinnedLane = null;
  _model = models.includes(asked) ? asked : (_visibleModels()[0] || '');
  _setModel(_model);
  _renderModelList();
  if (!_model) {
    _data = null;
    _renderEmpty();
    return;
  }
  const target = _model;
  try {
    const answer = await api.get('/admin/api/routing/' + encodeURIComponent(target));
    // Two clicks in quick succession, or a second tab on the same model,
    // can resolve out of order; the slower answer belongs to a selection
    // that is no longer on screen.
    if (_model !== target) return;
    _data = answer;
    renderRouting();
  } catch (e) {
    if (_model !== target) return;
    showToast(String(e.message || e), 'error');
  }
}

/** Nothing to route yet: say so where the board would be. */
function _renderEmpty() {
  const board = document.getElementById('routingBoard');
  if (board) {
    const message = _models().length ? t('routing.pickModel') : t('routing.noModels');
    board.innerHTML = '<div class="conn-row"><span class="conn-warn"></span>' +
      '<span class="conn-detail">' + esc(message) + '</span></div>';
  }
  ['routingPolicy', 'routingDecisions'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = '';
  });
  const count = document.getElementById('routingDecisionsCount');
  if (count) count.textContent = '';
}

/** Apply a preset: it writes the same block the editor writes. */
export async function applyRoutingPreset(id) {
  const block = presetBlock(id);
  if (!block) return;
  await _save(block);
}

export async function saveRouting() {
  const strategy = document.getElementById('routingStrategy');
  const affinity = document.getElementById('routingAffinity');
  const sink = document.getElementById('routingSink');
  const firstToken = document.getElementById('routingFirstToken');
  const block = {
    strategy: strategy ? strategy.value : 'weighted',
    affinity: affinity ? affinity.value : 'off',
    sink: sink ? sink.checked : true,
  };
  // "off" is a value, not an omission: the save merges over the stored
  // block, so leaving the key out would keep the threshold being removed.
  block.first_token_ms = firstToken && firstToken.value
    ? Number(firstToken.value)
    : null;
  await _save(block);
}

async function _save(block) {
  const name = _currentModel();
  if (!name) return;
  // The editor owns a few keys; a block that also carries rules keeps them,
  // because the save replaces the block whole.
  const merged = Object.assign({}, (_data && _data.routing) || {}, block);
  try {
    await api.put('/admin/api/routing/' + encodeURIComponent(name), { routing: merged });
    showToast(t('routing.saved'), 'success');
    await loadRouting(name);
  } catch (e) {
    showToast(String(e.message || e), 'error');
  }
}

// The panel's controls are inline `onclick` handlers, which resolve names on
// `window`; this is the one place they are published.
Object.assign(window, {
  loadRouting,
  applyRoutingPreset,
  saveRouting,
});
