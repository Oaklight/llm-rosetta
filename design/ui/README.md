# UI Design Prototypes

Standalone HTML mockups used to explore admin panel layouts before implementing
them in `src/llm_rosetta/gateway/admin/`. Open directly in a browser — no build
step, no dependencies.

## provider-list-responsive.html

Four-stage responsive layout for the provider list view.

Measured natural content widths (from the live admin panel with 21 providers):

| Column  | Natural width |
|---------|---------------|
| name (logo + text + toggle) | 240px |
| badges (LLM/EMBEDDING/RERANK) | 202px |
| type | 219px |
| base_url | 352px |
| actions | 322px |
| **total** (incl. gaps + padding) | **~1463px** |

### Stages

| Stage | Breakpoint | Layout |
|-------|------------|--------|
| 1 — Full row | ≥ 1500px | 5 columns, no compression |
| 2 — Compact row | 1100–1500px | 5 columns, type/url truncate, gap halved |
| 3 — Two-line row | 700–1100px | 3 bands: title (name + badges + toggle) / meta (type + url, 1fr each) / actions |
| 4 — Card | < 700px | card layout — see below |

Stage 3 is required because the badges column cannot compress below 202px —
three tags at their intrinsic width. Once name + badges + actions exceed the
available width, the row must wrap.

Stage 4 has **no CSS of its own**. Below 700px `js/providers.js` removes the
`.list-view` class entirely (via `matchMedia`), so the plain card rules apply
untouched. The view preference is still remembered in `localStorage`; only the
class is dropped. This is why list view and grid view are byte-identical at
narrow widths rather than merely similar — verified by comparing every
element's bounding box in both modes at 640px and 375px.

### Conventions

- **One markup shape** for both views (issue #611). Layout lives entirely in
  CSS; `renderProviders()` has no view conditional.
- **`data-label` + `::before`** for field labels, so `Type:` / `Base URL:` /
  `API Key:` are shown or hidden per layout rather than being present in one
  markup variant and absent in another.
- **Subgrid** for stages 1–2 so column widths are defined once on the container
  and every row inherits them. Avoids hand-synced pixel values.
- **Fade-out mask** instead of `text-overflow: ellipsis` for truncated text.
  Only applied in stages 1–2, where columns are fixed; stage 3 sizes to content
  and stage 4 wraps, so neither needs it.
- **Badges left-aligned** in card layout — keeps the same visual baseline as the
  Type / Base URL / action rows below it.
