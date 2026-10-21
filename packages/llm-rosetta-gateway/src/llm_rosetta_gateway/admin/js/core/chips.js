/**
 * chips.js — state mirroring for a chip that wraps a checkbox.
 *
 * Kept import-free so the JS suite can pin the property that matters: the
 * handler reads the checkbox and never writes it.  A label dispatches its
 * click once for the label and once for the synthesised click on the nested
 * input, so a handler that toggled `checked` here would run twice and cancel
 * the browser's own toggle — the bug this helper exists to prevent.
 */

/**
 * Mirror *checkbox* into the chip's `active` class.
 *
 * @param {{classList: {toggle: function}}} el - the chip element
 * @param {{checked: boolean}} checkbox - the chip's checkbox
 */
export function syncChipState(el, checkbox) {
  el.classList.toggle('active', checkbox.checked);
}
