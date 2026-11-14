/**
 * The order a form reports its problems in.
 *
 * The provider form used to answer "Select at least one capability" whatever
 * was wrong — an empty name, a name already in the config — because the
 * capability gate ran first and the other checks were never reached (#1519).
 * Which field to report is a rule: the first one the operator filled, so the
 * message matches the field they are looking at, and the mark goes on that
 * field rather than on the submit button.
 *
 * Pure on purpose: `node --test` can pin the order without a DOM.
 */

/** The smallest description of the form the rule needs. */
export const PROVIDER_FIELD_ORDER = ['name', 'base_url', 'type', 'capability'];

/**
 * The first problem the form has, or `null`.
 *
 * Order follows the form: the name is typed first, then the base URL decides
 * the type, and the capability checkboxes come last.  A name already in the
 * config is refused before anything else so a duplicate cannot hide behind
 * another message.
 *
 * @param {object} form
 * @param {string} form.name Trimmed provider id.
 * @param {string} form.baseUrl Trimmed base URL.
 * @param {string} form.type The top-level type the LLM section holds.
 * @param {boolean} form.isLlm Whether the LLM capability is ticked.
 * @param {boolean} form.anyCap Whether any capability is ticked.
 * @param {string[]} [form.existing] Provider ids already configured.
 * @param {string|null} [form.editing] The id being edited, or null for create.
 * @returns {{field: string, key: string}|null}
 */
export function providerFormError(form) {
  const name = String((form && form.name) || '').trim();
  const baseUrl = String((form && form.baseUrl) || '').trim();
  const type = String((form && form.type) || '').trim();
  const editing = form && form.editing ? String(form.editing) : null;
  const existing = ((form && form.existing) || []).map((id) => String(id).toLowerCase());
  if (!name) return { field: 'name', key: 'error.nameRequired' };
  // A rename onto another entry's id would overwrite that entry; creating a
  // second entry under an existing id does the same silently.
  if (existing.includes(name.toLowerCase()) && name.toLowerCase() !== (editing || '').toLowerCase()) {
    return { field: 'name', key: 'error.nameTaken' };
  }
  if (!baseUrl) return { field: 'base_url', key: 'error.baseUrlRequired' };
  if (form && form.isLlm && !type) return { field: 'type', key: 'error.typeRequired' };
  if (!(form && form.anyCap)) return { field: 'capability', key: 'error.noCapability' };
  return null;
}
