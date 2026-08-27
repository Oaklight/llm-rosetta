# UI Audit Track B — Handoff

> This document is a self-contained briefing for an independent session working
> on the **structure/behavior** track of the admin UI audit.

## Context

A full UI/UX audit was performed on the llm-rosetta gateway admin panel
(epic: [#557](https://github.com/Oaklight/llm-rosetta/issues/557)). Work is
split into two parallel tracks:

- **Track A** (visual layer, serial): #558 → #559 → #560 — CSS bugs, color
  system redesign, typography. Being handled in another session.
- **Track B** (structure/behavior, parallel): #561, #562, #563 — this track.

## Your issues

| Issue | Title | Priority | Scope |
|-------|-------|----------|-------|
| [#562](https://github.com/Oaklight/llm-rosetta/issues/562) | Responsive layout — tab overflow, table scroll, content width | P2 | CSS layout only |
| [#563](https://github.com/Oaklight/llm-rosetta/issues/563) | UI polish — favicon, empty states, toast, info leak | P3 | Additive, low conflict |
| [#561](https://github.com/Oaklight/llm-rosetta/issues/561) | Accessibility — ARIA roles, keyboard navigation | P2 | HTML attrs + JS |

**Recommended order: #562 → #563 → #561**

- #562 and #563 have near-zero conflict with Track A — safe to do right away.
- #561 touches `admin.html` structure heavily (adding `role`, `aria-*`,
  `tabindex` to tabs, segmented controls, modals, cards). Track A's #558 also
  edits `admin.html` (fixing hardcoded colors in inline styles). Do #561 last
  to minimize merge conflicts, or coordinate timing.

## Key files

```
src/llm_rosetta/gateway/admin/
├── admin.html              # 996 lines — the single-page HTML
├── css/
│   ├── base.css            # 225 lines — layout, header, tabs, settings popup
│   └── components.css      # 407 lines — buttons, cards, tables, modals, responsive
└── js/
    ├── core.js             # Theme defs (THEMES object), utility functions
    ├── init.js             # App init, branding, module loader
    ├── dashboard.js        # Dashboard tab logic
    ├── providers.js        # Provider tab logic
    ├── models.js           # Model tab logic
    ├── keys.js             # API keys tab logic
    ├── logs.js             # Request log tab logic
    ├── auth.js             # Login/auth flow
    ├── state.js            # Shared app state
    ├── i18n.js             # Internationalization
    ├── test.js             # Model test functionality
    └── fetch-models.js     # Fetch upstream models
```

## Running locally

```bash
ADMIN_PASSWORD=test123 conda run -n llm-rosetta python -m llm_rosetta.gateway --port 18899 --no-banner
```

Login at `http://localhost:18899/admin/` with password `test123`.

## Issue details (quick reference)

### #562 — Responsive layout

1. **Tab bar overflow**: `.tabs` needs `overflow-x: auto`. Tabs have
   `flex-shrink: 0` + `white-space: nowrap` but no scroll handling. 5 tabs
   don't fit at 375px.
2. **Tables not scrollable**: CSS defines `.table-scroll` in the 768px media
   query, but no `<table>` in HTML is wrapped in it. Dashboard, Models, API
   Keys, Request Log tables all clip on mobile.
3. **Content area too narrow**: `max-width: 1200px` wastes space on 1440px+.
   Consider 1400px or wider.
4. **Provider card density**: `minmax(320px, 1fr)` → only 3 cols at 1200px.
   Reduce to `minmax(280px, 1fr)` for 4 cols on wide screens.
5. **Settings popup padding**: 28px padding excessive on mobile. Reduce at
   600px breakpoint.

### #563 — UI polish

1. **Favicon**: `/favicon.ico` returns 401. Add one to the static handler,
   served unauthenticated.
2. **Empty states**: "No error dumps recorded." etc. — add icon + styled text.
3. **Toast position**: `bottom: 30px; right: 10px` — center horizontally
   instead.
4. **Config path leaks**: Shows full server path like
   `/home/user/.config/llm-rosetta-gateway/config.jsonc`. Abbreviate to
   filename, full path in tooltip.
5. **Clock timezone**: "6:37:33 PM" with no TZ. Append timezone abbreviation.
6. **Loading indicators**: Data sections pop in with no skeleton/spinner.

### #561 — Accessibility

1. **Tab bar**: Change `<div class="tab">` to proper `role="tablist"` /
   `role="tab"` / `role="tabpanel"` with `aria-selected` + arrow-key nav.
2. **Segmented controls**: Add `role="radiogroup"` / `role="radio"` to
   capability filters (All/LLM/Embedding/Rerank).
3. **Toggle switches**: Add `role="switch"` + `aria-checked` to enable/disable
   toggles.
4. **Modal focus trap**: Modals don't trap focus or restore it on close.
5. **Action button labels**: "Edit"/"Delete" buttons need
   `aria-label="Edit <provider-name>"` for context.

## Conflict avoidance rules

- **Don't touch color values** — Track A is rewriting the entire token system.
  If you need a new color (e.g., for an empty-state icon), use existing
  `var(--text-dim)` or `var(--border)`.
- **CSS layout changes** (overflow, grid-template, max-width, padding) are
  safe — Track A doesn't touch these.
- **HTML structural changes** (wrapping tables in divs, adding ARIA attrs) are
  mostly safe, but coordinate on `admin.html` if doing #561 — Track A's #558
  may edit the same lines to fix inline hardcoded colors.
- **JS changes** for keyboard handlers (#561) won't conflict — Track A doesn't
  touch JS behavior.

## Workflow reminder

Per project conventions: work in a worktree, open PR, send Matrix notification
with mentions before reporting done. Merge via rebase. See `CLAUDE.md` and
`submit-workflow.md` for full details.
