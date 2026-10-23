# Theme matrix

A standing Playwright harness for the admin panel's theme layer — the second
half of the check the CSS contract test cannot make.  The contract test
(`tests/js/theme-contract.test.mjs`) reads the stylesheets as text and says what
a scheme is *allowed* to declare.  This harness renders the panel and checks
what came out.

## What it does

For every cell of **scheme × mode × width** (default: every scheme `base.css`
declares — today Minimal and Emerald — × every declared mode × 390 / 768 /
1280 = 12) it:

1. **`scheme-applied`** — the scheme and mode planted in `localStorage`
   reached `document.documentElement`.
2. **`tokens-resolve`** — every custom property any scheme block in `base.css`
   declares resolves to a non-empty computed value.  A scheme that forgot a
   token falls back to nothing; a text-level test cannot see that.
3. **`scheme-values`** — samples `--bg`, `--accent` and `--card-bg` so the run
   can compare cells afterwards.
4. **`token-card-clickable`** — the Token card's "details" link is hit at its
   centre and its click opens the modal.  This is the #1149 regression as a
   render-level check: emerald's card texture sat above the card's contents,
   so the click landed on the card.
5. **`no-horizontal-overflow/dashboard`** and
   **`no-horizontal-overflow/providers`** — each tab's `scrollWidth` does not
   exceed the viewport at that width.  Providers carries wider content, so it
   is measured rather than only screenshotted.
6. Screenshots the **Dashboard** and **Providers** tabs, full page.

After the cells, two matrix-level checks compare the sampled values:

- **`schemes-render-differently`** — cells that differ only in scheme must
  disagree on at least one sampled token;
- **`modes-render-differently`** — the same for mode.

A block copy-pasted from another scheme keeps every per-cell probe green while
changing nothing on screen; these checks are what make a green run mean the
scheme rendered, not just that a value resolved.

`--schemes` / `--modes` narrow the axes while iterating. When a narrowed run
leaves a cross-cell check with nothing to compare, it reports **SKIP** rather
than a pass — the command still exits zero for a legitimate narrowing, but the
report says the comparison did not happen.

The screenshots are artifacts for review, not golden images, so the harness is
platform-independent and needs no pixel baseline.  The probes are the gate;
the images are what a human looks at when a probe fails, or when a scheme is
being added.

## Requirements

```bash
pip install "playwright>=1.40,<2"
python -m playwright install chromium    # add --with-deps on a bare CI image
```

The gateway package in this working tree is what gets rendered, so run it from
the repository root with the project's environment active.

## Usage

```bash
# A disposable gateway, a fresh data directory, the default matrix.
python -m tools.theme_matrix

# One axis, for iterating on a single scheme.
python -m tools.theme_matrix --schemes emerald --modes dark --widths 390,1280

# A real gateway with a real database (realistic numbers in the dashboard).
python -m tools.theme_matrix --base-url http://127.0.0.1:8765 \
    --admin-password "$THEME_MATRIX_ADMIN_PASSWORD"
```

Output goes to `theme-matrix-out/` by default:

- `report.json` — every cell, probe and artifact path.
- `report.md` — the same as a table with inline screenshots.
- `<scheme>-<mode>-<width>-dashboard.png`
- `<scheme>-<mode>-<width>-providers.png`

The command exits non-zero if any probe fails.  `make theme-matrix` runs the
default matrix.

By default the harness starts its own gateway: a synthetic config with two
disabled providers, no admin password, a temporary data directory, and an
ephemeral port.  `--keep-gateway` leaves that directory (config, data, log) on
disk for debugging.

## When it runs

- **Locally**: before and after any change to `admin/css/**` or
  `admin/admin.html`, and whenever a scheme is added.  Screenshots go in the
  PR description.
- **CI**: the `theme-matrix` job runs when the gateway package, `tests/js/` or
  this harness changes, and uploads `theme-matrix-out/` as an artifact.

## Adding a scheme

The guide is in the docs worktrees (`docs-en`/`docs-zh`,
`contributing/admin-themes.md`).  In short: declare the values in `base.css`,
list the scheme in `VALID_SCHEMES` (`admin/js/core/utils.js`) and the Settings
select, then run `make test-js` (the contract) and `make theme-matrix` (the
render).
