/**
 * shim-tree.js — pure builder for the admin provider-type picker tree.
 *
 * A shim *declares* where it belongs instead of having it inferred from its
 * name: `group` names the top level, `group_label` titles it, and
 * `display_name` labels the shim's own row.  Each of them falls back to the
 * historical `vendor--base` name split, so a shim that declares none of them
 * renders exactly as it did before those fields existed.
 *
 * That promise covers two populations, not one.  A shim written against the
 * schema that had no `group` / `group_label` may still declare `display_name`,
 * where it meant the *group* title; such a shim is read as the older schema
 * here — its `display_name` still titles the group, and its row still comes
 * from the name.  What separates the two readings is the presence of a
 * grouping field: see :func:`_declaresGrouping`.  Without that separation the
 * vendor's name would be printed twice, once as the group title and again on
 * the shim's only row.
 *
 * Every value the builder resolves carries the branch that produced it
 * (``declared`` / ``derived`` / the legacy ``display_name``), because those are
 * not interchangeable: a declared group is the vendor stating where it
 * belongs, while a derived one is an inference from the name that reproduces
 * the old rendering for the names we have seen and says nothing about the ones
 * we have not.  Recording it keeps "undeclared shims are unaffected by design"
 * separable from "they happen to be unaffected today".
 *
 * No DOM, no state, no imports — which is what lets a test assert the whole
 * registry against it.
 */

/** Where a resolved value came from. */
export const DECLARED = 'declared';
export const DERIVED = 'derived';
/** Legacy branch: `display_name` used to title the group before `group_label`. */
export const LEGACY_DISPLAY_NAME = 'display_name';

/** Capitalize the first letter of a string. */
function _capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

/** Index of the `vendor--base` separator, or -1 for a bare shim name. */
function _sep(name) { return name.indexOf('--'); }

/** Preferred display order for base types within a top-level group. */
const _BASE_TYPE_ORDER = [
  'chat_completions', 'open_responses', 'messages',
  'google_generate', 'google_interactions', 'decision',
];

/**
 * Top-level group key for a shim: the declared `group`, else the segment
 * before `--`.  A bare name (``moonshot``) is a group of its own.
 */
export function shimGroupKey(s) {
  if (s.group) return s.group;
  const sep = _sep(s.name);
  return sep === -1 ? s.name : s.name.slice(0, sep);
}

/** Which branch produced a shim's group key (see :func:`shimGroupKey`). */
export function shimGroupSource(s) {
  return s.group ? DECLARED : DERIVED;
}

/**
 * Whether a shim speaks the schema in which `display_name` labels its own row.
 *
 * `group_label` is the group title and `group` the group key, so a shim that
 * declares either has opted into the new fields and its `display_name` can only
 * mean the row label.  A shim that declares neither is read as the older
 * schema, where `display_name` titled the group — and is rendered exactly as it
 * was before these fields existed.
 */
function _declaresGrouping(s) {
  return Boolean(s.group || s.group_label);
}

/**
 * The title a shim declares for its top-level group, or ``null``.
 *
 * ``group_label`` is the declared title.  A shim that declares neither
 * ``group_label`` nor ``group`` keeps the legacy meaning of ``display_name`` —
 * before those fields existed it titled the group — so shims written against
 * the older schema, including external ones, render unchanged: the same title
 * *and* the same name-derived row label.
 */
function _declaredGroupTitle(s) {
  if (s.group_label) return s.group_label;
  if (!s.group && s.display_name) return s.display_name;
  return null;
}

/** Which branch produced a group's title (see :func:`_declaredGroupTitle`). */
function _groupTitleSource(s) {
  if (s.group_label) return DECLARED;
  if (!s.group && s.display_name) return LEGACY_DISPLAY_NAME;
  return DERIVED;
}

/**
 * Label for a shim's own row inside its group: the declared `display_name`,
 * else the segment after `--` (usually the base format, but a vendor may name
 * a deployment there), and a bare name shows its base.
 *
 * `display_name` labels the row only for a shim that declared a grouping field;
 * for one that did not, it is the legacy group title and the row stays
 * name-derived (see :func:`_declaresGrouping`).
 */
export function shimLabel(s) {
  if (s.display_name && _declaresGrouping(s)) return s.display_name;
  const sep = _sep(s.name);
  return sep === -1 ? s.base : s.name.slice(sep + 2);
}

/** Which branch produced a shim's row label (see :func:`shimLabel`). */
export function shimLabelSource(s) {
  return s.display_name && _declaresGrouping(s) ? DECLARED : DERIVED;
}

/**
 * Brand for free-source affordances (``"<Brand> (Free)"``): the vendor's name,
 * not the individual shim's.
 *
 * That is the group title, so a grouped vendor that wants more than its
 * capitalized group key on the button declares `group_label`.
 */
export function shimBrand(s) {
  return _declaredGroupTitle(s) || _capitalize(shimGroupKey(s));
}

/**
 * Build the top-level → base tree for the provider-type picker.
 *
 * Vendor-neutral base formats (``chat_completions``, …) become childless roots
 * — they are their own top level.  Returns ``{roots, rootOf(value)}``; each
 * root carries ``groupSource`` / ``titleSource`` and each child carries
 * ``labelSource``, so the branch behind every rendered string is inspectable.
 */
export function buildTypeModel(registeredShims, baseFormats) {
  const rootMap = new Map();
  const valueRoot = new Map();
  const _root = (rootName) => {
    if (!rootMap.has(rootName))
      rootMap.set(rootName, { name: rootName, label: rootName, logo: '', defaultValue: rootName, children: [] });
    return rootMap.get(rootName);
  };

  for (const s of registeredShims) {
    const root = _root(shimGroupKey(s));
    root.children.push({
      value: s.name,
      label: shimLabel(s),
      base: s.base,
      labelSource: shimLabelSource(s),
    });
    if (s.logo) root.logo = s.logo;
    const title = _declaredGroupTitle(s);
    if (title) {
      root.label = title;
      root.titleSource = _groupTitleSource(s);
    } else if (!root.titleSource) {
      root.titleSource = _groupTitleSource(s);
    }
    // A member that declared its group settles the root's provenance, even if
    // another member of the same root only derived the same key.
    if (root.groupSource !== DECLARED) root.groupSource = shimGroupSource(s);
    valueRoot.set(s.name, root);
  }
  for (const f of baseFormats) {
    valueRoot.set(f.name, _root(f.name));
  }

  const roots = [...rootMap.values()];
  for (const root of roots) {
    // Unknown base types (a base added upstream, or a non-chat one like
    // `decision`) rank last instead of first.
    const rank = (b) => {
      const i = _BASE_TYPE_ORDER.indexOf(b);
      return i === -1 ? _BASE_TYPE_ORDER.length : i;
    };
    root.children.sort((a, b) => {
      const d = rank(a.base) - rank(b.base);
      return d !== 0 ? d : a.label.localeCompare(b.label);
    });
    if (root.children.length) root.defaultValue = root.children[0].value;
  }
  roots.sort((a, b) => a.name.localeCompare(b.name));
  return { roots, rootOf: (v) => valueRoot.get(v) };
}
