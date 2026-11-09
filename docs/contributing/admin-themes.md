---
title: Admin Themes
---

# Admin themes: schemes as values, components as owners

The admin panel's look is split in two. A **scheme** declares values — colours,
radii, textures, how a table header is set. A **component** decides where those
values land — its box model, its layout, its pointer behaviour. The boundary
between the two is a contract, and it is enforced by tests rather than left to
review.

The contract exists because of a bug: emerald's card texture was a pseudo-element
positioned over the card with `pointer-events: auto`, so the Token card's
"details" link could not be clicked in that scheme —
[#1149](https://github.com/Oaklight/llm-rosetta/pull/1149). The link worked in
minimal, so it looked like a browser difference. Nothing forbade the theme from
reaching into the card's hit-testing; now something does
([epic #1154](https://github.com/Oaklight/llm-rosetta/issues/1154)).

## The vocabulary

| Piece | Lives in | Owns |
|-------|----------|------|
| Scheme | `admin/css/themes/<name>.css` | token values, per mode |
| Registry | `admin/themes.py` | the list of schemes: name, label, file |
| Component | `admin/css/components.css` and the structural rules in `base.css` | box model, layout, pointer behaviour |
| Variant | a component's own rule keyed on `data-variant` | a structural difference between schemes |

The selected scheme and mode are attributes on the document element
(`<html data-scheme="emerald" data-mode="dark">`), mirrored to `localStorage`
under `llm-rosetta-scheme` and `llm-rosetta-mode`. The panel reads them before
its first paint, so a stored scheme does not flash the default. The registry
crosses the wire as injected JSON (`#adminThemes`), which is the only list the
page, the Settings picker and `tools/theme_matrix` consult.

## The four rules

1. **A scheme selector appears only in a theme file, and names nothing but the
   scheme and mode.** `[data-scheme="emerald"]` and
   `[data-scheme="emerald"][data-mode="dark"]` are the only legal shapes. A
   selector such as `[data-scheme="emerald"] .stat-card` belongs to no one: it
   is a theme reaching into a component.
2. **A scheme block declares custom properties only.** Declarations such as
   `color`, `padding` or `position` in a scheme block are the same leak by
   another route.
3. **A decorative pseudo-element positioned over its parent declares
   `pointer-events: none`.** This is rule 3 of the incident above, stated as a
   rule. A decoration on a control is exempt — a switch knob's originating
   element is itself the click target — and every other exemption is listed by
   hand in the test, with its reason.
4. **A difference a value cannot express is a component variant, not a wider
   scheme selector.** Minimal and emerald differ in more than colour: minimal
   merges the KPI cards into one frame, emerald separates them and textures
   each. Those differences are expressed as tokens (`--stats-gap`,
   `--card-texture`, …). When a future scheme needs something the token set
   cannot say, the answer is a `data-variant` attribute set alongside the
   scheme and a rule in the component's own stylesheet — never
   `[data-scheme="…"] .component`. No component needs a variant today; this is
   the boundary for when one does.

Rules 1–3 are enforced by `tests/js/theme-contract.test.mjs` (`make test-js`),
which parses the stylesheets as text, including inside `@media` blocks. Rule 4
is a review rule until a variant exists.

## Scheme tokens

A scheme declares two kinds of value. **Structural tokens** are shared by light
and dark; **colour tokens** are declared once per mode.

### Structural tokens

| Group | Tokens | What they control |
|-------|--------|-------------------|
| Shape | `--radius`, `--badge-radius` | corner radius of surfaces and badges |
| Page texture | `--dot`, `--grid`, `--page-texture`, `--page-texture-size`, `--page-texture-display` | the fixed `body::before` backdrop |
| Cards | `--card-bg`, `--card-border`, `--card-radius`, `--card-texture`, `--card-texture-opacity` | the card surface and its texture |
| KPI grid | `--stats-gap`, `--stats-frame-bg`, `--stats-frame-border`, `--stats-frame-radius` | merged frame (minimal) vs separated cards (emerald) |
| Bars | `--bar-color`, `--bar-opacity`, `--bar-hover` | the small usage bars on cards |
| Table headers | `--th-transform`, `--th-spacing`, `--th-weight`, `--th-size`, `--th-color` | how column headers are set |
| Badges and buttons | `--badge-weight`, `--btn-primary-weight` | emphasis weights |

### Colour tokens

The names the panel's stylesheets converge on (issue
[#1472](https://github.com/Oaklight/llm-rosetta/issues/1472), P1):

| Role | Tokens | Notes |
|------|--------|-------|
| Surfaces | `--bg`, `--surface`, `--surface-2` | page, panel, hover/secondary |
| Lines | `--hairline`, `--hairline-strong` | rows and separators; the strong variant is derived |
| Text | `--text`, `--text-dim`, `--text-muted` | |
| Accent | `--accent`, `--accent-hover`, `--accent-subtle`, `--accent-on` | |
| Status | `--pos`, `--neg`, `--warn`, `--info` | each with `-subtle` (translucent wash), `-bg` (opaque tint over `--surface`) and `-border` (derived) |
| Other | `--purple`, `--magenta`, `--cyan`, `--lime`, `--green-deep` | capability and type colours |

`--hairline-strong`, the four `-border` variants, `--neg-bg`, `--warn-bg` and
`--info-bg` are `color-mix()` expressions over the role colours — the same construct
`components.css` already uses for its tints — so a theme can add them without
adding a colour value. They are provisional: no component paints them yet, and
the phase that first does (`P3`/`P4`) tunes the ratios against the theme matrix.

The type scale, the spacing steps and the shell's geometry (`--fs-*`,
`--tracking-label`, `--space-1..6`, `--side-w`, `--topbar-h`) do not vary by
theme; they live in `base.css` `:root`.

### Compatibility aliases

P1 moved the values under the names above and left the previous names
(`--bg-card`, `--bg-hover`, `--border`, `--green`, `--green-subtle`,
`--green-bg`, `--red`, `--red-subtle`, `--orange`, `--blue`, `--badge-radius`)
as aliases in a block at the end of each theme file, so no component changes
behaviour. That block is temporary: **P5 deletes it**, and the rule that keeps
it from becoming a second API is a grep — outside that block, no old name may
remain anywhere in `admin/`.

A dark block may re-declare `--dot` and `--grid` when the texture needs a
different intensity on a dark background (emerald does). It may not re-declare
anything else from the structural block unless the difference is genuinely a
difference between modes rather than schemes.

## Adding a scheme

The matrix's default scheme axis is the registry's list, so a new theme is
covered the next time it runs; `--schemes` narrows the matrix while you are
iterating on one. The checklist below is the whole surface.

1. **Write `admin/css/themes/<name>.css`.** One shared block
   (`[data-scheme="<name>"]`) and one block per mode
   (`[data-scheme="<name>"][data-mode="light"]`,
   `[data-scheme="<name>"][data-mode="dark"]`). Declare the same token names as
   the default theme — `tests/gateway/test_admin_themes.py` compares the two
   themes' available sets per mode, so a token left out fails there — and
   declare values only: the contract test rejects a real property in a scheme
   block. A mode may re-declare a shared token when the value really differs
   between modes (emerald's `--dot`/`--grid`).
2. **Register it in `admin/themes.py`.**

   ```python
   Theme("<name>", "<Label>", "css/themes/<name>.css"),
   ```

   That one entry is what the boot script receives, what the Settings picker
   lists, what the matrix enumerates, and what the wheel packaging ships
   (`css/**/*.css`). There is no second list to update.
3. **Optionally add the label to both dictionaries.** `scheme.<name>` goes in
   the English and Chinese dictionaries in `admin/js/core/i18n.js`; without it
   the picker shows the registry's own label. The Chinese dictionary stores
   non-ASCII characters as literal `\uXXXX` escapes, so edit it with a tool
   that preserves backslashes.
4. **Run the checks and attach the screenshots.**

   ```bash
   make test-js      # the contract: what a scheme may say
   make theme-matrix # the render: scheme × mode × width, with screenshots
   ```

   `make theme-matrix` writes `theme-matrix-out/report.md`, which embeds the
   screenshots. Add the report or its images to the PR.

## Verifying a theme change

| Check | Command | Covers | Does not cover |
|-------|---------|--------|----------------|
| Contract test | `make test-js` | which selectors and declarations a scheme may use; decorative pseudo-elements taking clicks; at-rule nesting | whether a value renders legibly, or whether a token is missing at runtime |
| Theme matrix | `make theme-matrix` | every scheme × mode × width against a live panel: scheme applied, every scheme token resolved, each tab does not scroll sideways, the Token card's details link clickable, and schemes/modes render differently; screenshots of the Dashboard and Providers tabs | regressions outside the captured tabs; pixel-level changes |

The matrix runs in CI (`theme-matrix` job) whenever the gateway package, the
JS tests or the harness itself change. It has no pixel baseline, so it is not
sensitive to fonts or the runner image; a failing probe is a behavioural or
structural change, and the screenshots are for the human reviewing it.

To run the matrix against a real gateway with realistic numbers in the
dashboard:

```bash
python -m tools.theme_matrix --base-url http://127.0.0.1:8765 \
    --admin-password "$THEME_MATRIX_ADMIN_PASSWORD"
```

The harness lives in `tools/theme_matrix/`; its README documents the probes and
the options in full.
