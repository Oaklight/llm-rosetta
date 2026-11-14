/**
 * disclosure.js — the open/close of one card, shared by both shim lists.
 *
 * The Installed view's vendor card and the Available view's package card are
 * the same widget: a head that is the whole click target, a body that stays
 * hidden until the card is the one being worked on, and a chevron that turns.
 * Keeping the mechanics here is what stops the two lists from drifting — one
 * of them stretched its neighbours open, the other did not.
 */

/**
 * Toggle one card and remember the choice for the next repaint.
 *
 * The DOM is edited in place rather than re-rendered: a full repaint would
 * close every card under the cursor and throw away the scroll position.
 *
 * @param {object} opts
 * @param {string} opts.root - Selector of the list that holds the cards.
 * @param {string} opts.card - Selector of one card, within `root`.
 * @param {string} opts.attr - The card's data attribute that carries its key.
 * @param {string} opts.key - This card's key.
 * @param {Set<string>} opts.open - The keys currently open.
 * @param {string} opts.head - Selector of the head inside the card.
 * @param {string} opts.body - Selector of the body inside the card.
 * @param {string} [opts.openClass] - Class that marks the open state; the
 *   chevron turns from the class, so no inline transform is set here.
 * @returns {boolean} Whether the card is open after the toggle.
 */
export function toggleCard({
  root, card, attr, key, open, head, body, openClass = 'is-open',
}) {
  const el = [...document.querySelectorAll(`${root} ${card}`)]
    .find((c) => c.dataset[attr] === key);
  const next = !open.has(key);
  if (next) open.add(key); else open.delete(key);
  if (!el) return next;
  el.classList.toggle(openClass, next);
  const headEl = el.querySelector(head);
  const bodyEl = el.querySelector(body);
  if (headEl) headEl.setAttribute('aria-expanded', String(next));
  if (bodyEl) bodyEl.hidden = !next;
  return next;
}
