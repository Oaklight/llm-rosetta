// provider-row.js — the lane controls every surface draws.
//
// The row *layout* belongs to the surface: the Models page draws a menu row,
// the drawer a single-provider row, the Routing board a grid row.  The
// controls do not.  The same switch was assembled three times — provider
// toggle, model-entry toggle, board toggle — and each copy drifted: one lost
// its provider pin, another blamed the wrong switch for being off (#1374,
// #1375, #1381).  These builders are the parts that must stay identical.
//
// Not every switch in the admin is a lane: the model-level enable
// (`models.js::renderModels`, which needs an inline style) and the
// provider-level enable (`providers.js`, which sits in the provider table)
// still draw their own.  They switch a model and a provider, not a lane, and
// carry no weight or reason — if a third lane-shaped control appears, it
// belongs here.
import { esc, jsArg } from '../core/html.js';

/** The lane's on/off switch.
 *
 *  `onchange` is JS source, not text: build its arguments with `jsArg`, not
 *  `esc` — the attribute is parsed as HTML before the handler is compiled, so
 *  `esc`'s `&#39;` becomes a real quote and a name can close the literal.  A
 *  surface that listens for the event instead (the Routing board) passes
 *  `data` and no `onchange`.  `extraClass` is how a surface keeps its own
 *  hook (`routing-switch`). */
export function laneSwitchHtml({
  name,
  on,
  title,
  onchange = '',
  data = '',
  extraClass = '',
}) {
  const attrs = [
    on ? 'checked' : '',
    `aria-checked="${on}"`,
    `aria-label="${esc(name)}"`,
    onchange ? `onchange="${onchange}"` : '',
    data,
  ]
    .filter(Boolean)
    .join(' ');
  return (
    `<label class="toggle row-toggle${extraClass ? ' ' + extraClass : ''}" title="${esc(title)}">` +
    `<input type="checkbox" role="switch" ${attrs}>` +
    '<span class="slider"></span></label>'
  );
}

/** The drag handle.  `data` carries the surface's identifier (the Models page
 *  keys on the pair, the board on the lane); `cls` keeps the styling hook. */
export function laneGripHtml({ title, cls = 'pm-grip', data = '' }) {
  return (
    `<span class="${cls}" title="${esc(title)}" aria-hidden="true"` +
    (data ? ` ${data}` : '') +
    '>⠿</span>'
  );
}

/** The weight control, for the surfaces whose weight is a text field.
 *  The board's stepper is not this control: it writes on every step instead
 *  of on blur, so it stays the board's own. */
export function laneWeightInputHtml({ model, provider, weight, title }) {
  return (
    `<input type="number" class="pd-weight" min="1" max="100" value="${esc(weight)}"` +
    ` title="${esc(title)}"` +
    ` onchange="updateProviderWeight(${jsArg(model)},${jsArg(provider)},this)"` +
    ` onkeydown="if(event.key==='Enter'){this.blur()}">`
  );
}
