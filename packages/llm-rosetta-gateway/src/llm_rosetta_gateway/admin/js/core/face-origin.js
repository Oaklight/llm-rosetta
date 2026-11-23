/**
 * face-origin.js — who chose a capability's shim: the preset or the operator.
 *
 * The provider drawer fills each capability from the entry's preset, and a
 * capability the operator has set by hand must not be moved by it.  The rule is
 * kept here, with no DOM, so a test can state it once and cover every path
 * (open, preset change, reset) instead of asserting on the drawer's source.
 *
 * A face is *manual* when it holds a shim the preset does not offer for it.  A
 * preset offers a shim for a capability only when the preset's vendor registers
 * one of that model type, so an LLM-only vendor offers nothing for embeddings:
 * whatever the face holds there is the operator's.
 */
import { shimGroupKey } from './shim-tree.js';

/** The shim a preset offers for one capability, or null.
 *
 *  The LLM capability *is* the preset unless the operator picks another.  A
 *  typed capability takes the first registered shim of its model type from the
 *  preset's vendor group.
 *
 *  @param {Array<{name: string, model_type?: string}>} shims registered shims
 *  @param {{name: string, is_llm?: boolean}} desc the capability's descriptor
 *  @param {string} preset the entry's preset (a shim or bare-format name)
 */
export function presetShimFor(shims, desc, preset) {
  if (desc.is_llm) return preset || null;
  const base = shims.find((s) => s.name === preset);
  if (!base) return null;
  const vendor = shimGroupKey(base);
  const hit = shims.find((s) => s.model_type === desc.name && shimGroupKey(s) === vendor);
  return hit ? hit.name : null;
}

/** Whether a stored face opens as manual.  Derived from the entry, never
 *  stored, so an old entry displays exactly as it was written. */
export function initialManual(value, target) {
  return !!value && (!target || value !== target);
}

/** What the drawer says about a face: 'preset', 'manual', or '' for nothing. */
export function faceOrigin({ preset, value, target, manual }) {
  if (!preset || !value) return '';
  if (!target) return 'manual';
  return manual ? 'manual' : 'preset';
}

/** What a preset change does to one face.
 *
 *  @returns {{move: boolean, manual: boolean}} whether to move the face to the
 *  new offer, and the flag it keeps afterwards.  A face the new preset offers
 *  nothing for keeps its value and, from here on, is the operator's.
 */
export function afterPresetChange({ manual, value, target }) {
  if (manual) return { move: false, manual: true };
  if (target) return { move: true, manual: false };
  return { move: false, manual: !!value };
}
