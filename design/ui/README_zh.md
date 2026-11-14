# UI 设计原型

在实现到 `packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/` 之前，用来探索后台面板布局的独立 HTML 原型。直接在浏览器打开即可，没有构建步骤，也没有依赖。

## provider-list-responsive.html

provider 列表视图的四段式响应式布局。

实测的自然内容宽度（来自有 21 个 provider 的实际后台面板）：

| 列 | 自然宽度 |
|---------|---------------|
| name（logo + 文本 + 开关） | 240px |
| badges（LLM/EMBEDDING/RERANK） | 202px |
| type | 219px |
| base_url | 352px |
| actions | 322px |
| **合计**（含间距与内边距） | **约 1463px** |

### 阶段

| 阶段 | 断点 | 布局 |
|-------|------------|--------|
| 1 — 整行 | ≥ 1500px | 5 列，不压缩 |
| 2 — 紧凑行 | 1100–1500px | 5 列，type/url 截断，间距减半 |
| 3 — 两行 | 700–1100px | 3 段：标题（name + badges + 开关）/ 元信息（type + url，各 1fr）/ 操作 |
| 4 — 卡片 | < 700px | 卡片布局，见下 |

阶段 3 是必需的，因为徽章列压不到 202px 以下：三个标签各有固有宽度。一旦 name + badges + actions 超过可用宽度，整行就必须换行。

阶段 4 没有自己的 CSS。低于 700px 时，`js/providers.js` 通过 `matchMedia` 直接移除 `.list-view` 类，于是普通卡片规则原样生效。视图偏好仍记在 `localStorage` 里，只是类被去掉。所以窄宽度下列表视图与网格视图是逐字节一致的，而不只是“相似”；这一点是在 640px 和 375px 下逐个元素比较包围盒验证的。

### 约定

- 两种视图共用一套标记（issue #611）。布局全部写在 CSS 里，`renderProviders()` 没有视图分支。
- 字段标签用 `data-label` + `::before`，这样 `Type:` / `Base URL:` / `API Key:` 可以按布局显示或隐藏，而不是一种标记里有、另一种里没有。
- 阶段 1–2 用 subgrid，列宽只在容器上定义一次，每行继承。避免手工同步像素值。
- 截断文本用淡出遮罩，而不是 `text-overflow: ellipsis`。只在列宽固定的阶段 1–2 使用；阶段 3 按内容定宽、阶段 4 换行，都不需要。
- 卡片布局里徽章左对齐，与下方 Type / Base URL / 操作行保持同一条视觉基线。
