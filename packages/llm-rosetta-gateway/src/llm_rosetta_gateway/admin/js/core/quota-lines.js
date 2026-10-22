/**
 * quota-lines.js — The router's verdict on a provider, as card rows.
 *
 * Kept free of DOM and module imports (the translator is passed in) so the
 * precedence can be pinned by `node --test tests/js/quota-lines.test.mjs`.
 * The precedence is the whole point: a provider can have a healthy reading
 * and still be out of rotation, and a row that says "90% left" about a lane
 * the gateway is skipping tells the operator the opposite of the truth.
 */

/**
 * The weight the router applies for a remaining share.
 *
 * Mirrors `quota_weight()` on the server: `0.25 + 0.75 × headroom`, never
 * zero — a provider is removed by the quota state, not by this factor.
 */
export function routingWeight(headroom) {
  const share = Math.max(0, Math.min(1, Number(headroom)));
  return 0.25 + 0.75 * share;
}

/**
 * The percentage to print for a share, without claiming a certainty it lacks.
 *
 * Anything short of a full window is clamped below 100 — 0.996 is not "100%
 * left" — and a spent share still shows the quarter it serves at.
 */
export function sharePercent(share) {
  const pct = Math.round(Math.max(0, Math.min(1, Number(share))) * 100);
  return share >= 1 ? 100 : Math.min(99, Math.max(0, pct));
}

/**
 * The weight to print: the server's when it sent one, otherwise derived.
 *
 * The server owns the rule (`quota_weight()`); deriving it here is only for a
 * payload from an older gateway, so the formula stays in one authoritative
 * place.
 */
export function appliedWeight(routing) {
  if (routing && routing.weight !== null && routing.weight !== undefined) {
    return routing.weight;
  }
  return routingWeight(routing ? routing.headroom : 1);
}

/**
 * Gateway-owned block reasons, in the operator's language.
 *
 * A probe's own refusal message is vendor text and passes through verbatim;
 * only reasons this gateway writes itself have a key.
 */
const BLOCK_REASON_KEYS = {
  'keyless lane rate limited': 'quota.blockReason.keyless',
};

function blockReasonLabel(reason, t) {
  const key = BLOCK_REASON_KEYS[reason];
  if (!key) return reason;
  const label = t(key);
  return label === key ? reason : label;
}

/**
 * Rows for the router's verdict, in precedence order.
 *
 * Returns an empty list when the provider is blocked: the card renders the
 * block row itself (it carries the cooldown's own reason), and a share row
 * next to it would contradict it.
 */
export function routingRows(routing, t) {
  if (!routing || routing.blocked) return [];
  if (routing.excluded) {
    // The server sends a code; an unknown one falls back to the code itself
    // rather than rendering a missing i18n key.
    let reason = '—';
    if (routing.reason) {
      const key = 'quota.reason.' + routing.reason;
      const label = t(key);
      reason = label === key ? routing.reason : label;
    }
    return [{ tone: 'fail', text: t('quota.routingExcluded', { reason }) }];
  }
  if (routing.headroom === null || routing.headroom === undefined) {
    return [{ tone: 'warn', text: t('quota.routingUnmeasured') }];
  }
  return [{
    tone: 'ok',
    text: t('quota.routingShare', {
      share: sharePercent(appliedWeight(routing)),
      pct: sharePercent(routing.headroom),
    }),
  }];
}

/** The row for a live provider-wide cooldown, or `null`. */
export function blockRow(block, t) {
  if (!block) return null;
  const until = block.until ? new Date(block.until).toLocaleString() : '—';
  const reason = blockReasonLabel(block.reason || '', t) || '—';
  return { tone: 'warn', text: t('quota.blocked', { until, reason }) };
}
