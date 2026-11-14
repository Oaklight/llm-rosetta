/**
 * Import shims for `route.test.mjs`.
 *
 * `core/topbar.js` reads `document` when `syncTopbarTitle()` runs, but the
 * module itself is safe to import in node; `core/route.js` is DOM-free.  The
 * re-export keeps the test's import list about the grammar rather than about
 * where the helpers happen to live.
 */

globalThis.window = globalThis;

export { buildHash, parseHash, sameRoute } from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/route.js';
export { documentTitle as documentTitleOf } from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/topbar.js';
