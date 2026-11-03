/**
 * routing-presets.js — Named routing blocks for the Routing panel.
 *
 * A preset is not a mode the gateway knows about: it is a literal block in the
 * same vocabulary as `models.<name>.routing`, so choosing one writes the block
 * the operator could have typed, and the panel then shows what it wrote.
 * Kept free of DOM so `node --test tests/js/` can pin it.
 */

/** The presets, in the order the panel offers them. */
export const ROUTING_PRESETS = [
  {
    id: 'balanced',
    label: 'routing.preset.balanced',
    hint: 'routing.preset.balancedHint',
    block: { strategy: 'weighted', affinity: 'off', sink: true },
  },
  {
    id: 'sticky',
    label: 'routing.preset.sticky',
    hint: 'routing.preset.stickyHint',
    block: { strategy: 'weighted', affinity: 'session', sink: true },
  },
  {
    // The list order is the policy: the first eligible provider serves every
    // request.  Not "cheap first" — the gateway holds no pricing, so the
    // operator decides what "first" means by ordering the lanes.
    id: 'priority',
    label: 'routing.preset.priority',
    hint: 'routing.preset.priorityHint',
    block: { strategy: 'ordered', affinity: 'off', sink: true },
  },
  {
    id: 'spread',
    label: 'routing.preset.spread',
    hint: 'routing.preset.spreadHint',
    block: { strategy: 'rotate', affinity: 'off', sink: true },
  },
  {
    id: 'quota_saver',
    label: 'routing.preset.quotaSaver',
    hint: 'routing.preset.quotaSaverHint',
    block: { strategy: 'pace', affinity: 'off', sink: true },
  },
];

/** The block a preset writes, or `null` for an unknown id. */
export function presetBlock(id) {
  const preset = ROUTING_PRESETS.find((entry) => entry.id === id);
  return preset ? { ...preset.block } : null;
}

/**
 * Which preset, if any, a block is exactly.
 *
 * A block the operator edited by hand is normally no preset at all, and the
 * panel says so rather than claiming the nearest one.
 */
export function matchPreset(block) {
  if (!block) return null;
  for (const preset of ROUTING_PRESETS) {
    const keys = Object.keys(preset.block);
    const same = keys.every((key) => (block[key] ?? preset.block[key]) === preset.block[key]);
    if (same) return preset.id;
  }
  return null;
}
