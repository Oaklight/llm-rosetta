---
title: 管理面板主题
---

# 管理面板主题：方案只声明值，组件决定归属

管理面板的外观分成两层。**配色方案**（scheme）声明值——颜色、圆角、纹理、表头的排法；**组件**决定这些值落在哪里——它自己的盒模型、布局和点击行为。两层之间的边界是一份契约，由测试而不只是评审来保证。

这份契约来自一个真实的 bug：emerald 方案的卡片纹理是一个覆盖在卡片之上的伪元素，带着 `pointer-events: auto`，于是在该方案下 Token 卡片的「详情」链接点不动——[#1149](https://github.com/Oaklight/llm-rosetta/pull/1149)。链接在 minimal 下正常，所以它看起来像浏览器差异。当时没有任何规则禁止主题伸手进卡片的命中测试；现在有了（[epic #1154](https://github.com/Oaklight/llm-rosetta/issues/1154)）。

## 概念

| 概念 | 位置 | 负责 |
|------|------|------|
| 配色方案 | `admin/css/themes/<name>.css` | token 值，按模式声明 |
| 注册表 | `admin/themes.py` | 方案清单：名字、标签、文件 |
| 组件 | `admin/css/components.css` 与 `base.css` 中的结构规则 | 盒模型、布局、点击行为 |
| 变体 | 组件自己的规则，以 `data-variant` 为键 | 方案之间的结构性差异 |

选中的方案与模式是文档元素上的属性
（`<html data-scheme="emerald" data-mode="dark">`），并同步到 `localStorage` 的
`llm-rosetta-scheme` 与 `llm-rosetta-mode`。面板在首次绘制前读取它们，因此已保存的方案不会先闪一下默认方案。注册表以注入 JSON（`#adminThemes`）的形式送到页面，这是页面、设置下拉框与 `tools/theme_matrix` 唯一的清单来源。

## 四条规则

1. **方案选择器只能出现在主题文件里，且只能点名方案和模式。** 合法形式只有
   `[data-scheme="emerald"]` 与 `[data-scheme="emerald"][data-mode="dark"]`。
   像 `[data-scheme="emerald"] .stat-card` 这样的选择器不属于任何一方：它是主题伸手进了组件。
2. **方案块只声明自定义属性。** 在方案块里写 `color`、`padding` 或 `position`，是同一个泄漏的另一种写法。
3. **覆盖在父元素之上的装饰性伪元素必须声明 `pointer-events: none`。** 这是上面那次事故的规则化。控件上的装饰例外——开关旋钮的起始元素本身就是点击目标——其余每一处豁免都在测试里逐条列出，并写明理由。
4. **值表达不了的差异用组件变体，而不是更宽的选择器。** minimal 与 emerald 的差异不止颜色：minimal 把 KPI 卡片合并成一个框，emerald 把它们分开并给每张卡片加纹理。这些差异用 token 表达（`--stats-gap`、`--card-texture` 等）。将来若某个方案需要 token 集说不出的东西，答案是随方案一起设置的 `data-variant` 属性，加上组件自己样式表里的规则——绝不是 `[data-scheme="…"] .component`。目前没有组件需要变体；这是需要时的边界。

规则 1–3 由 `tests/js/theme-contract.test.mjs`（`make test-js`）强制执行，它把样式表当文本解析，`@media` 块里的规则同样检查。规则 4 在有变体之前是评审规则。

## 方案 token

一个方案声明两类值。**结构 token** 由浅色与深色共用；**颜色 token** 按模式各声明一次。

### 结构 token

| 分组 | token | 控制什么 |
|------|-------|----------|
| 形状 | `--radius`、`--badge-radius` | 表面与徽章的圆角 |
| 页面纹理 | `--dot`、`--grid`、`--page-texture`、`--page-texture-size`、`--page-texture-display` | 固定的 `body::before` 背景 |
| 卡片 | `--card-bg`、`--card-border`、`--card-radius`、`--card-texture`、`--card-texture-opacity` | 卡片表面及其纹理 |
| KPI 栅格 | `--stats-gap`、`--stats-frame-bg`、`--stats-frame-border`、`--stats-frame-radius` | 合并成一框（minimal）还是分开的卡片（emerald） |
| 用量条 | `--bar-color`、`--bar-opacity`、`--bar-hover` | 卡片上的小型用量条 |
| 表头 | `--th-transform`、`--th-spacing`、`--th-weight`、`--th-size`、`--th-color` | 列标题的排法 |
| 徽章与按钮 | `--badge-weight`、`--btn-primary-weight` | 强调字重 |

### 颜色 token

面板样式表收敛到的名字（issue [#1472](https://github.com/Oaklight/llm-rosetta/issues/1472)，P1）：

| 角色 | token | 说明 |
|------|-------|------|
| 表面 | `--bg`、`--surface`、`--surface-2` | 页面、面板、悬停/次级 |
| 线 | `--hairline`、`--hairline-strong` | 行与分隔线；强档是派生的 |
| 文字 | `--text`、`--text-dim`、`--text-muted` | |
| 强调 | `--accent`、`--accent-hover`、`--accent-subtle`、`--accent-on` | |
| 状态 | `--pos`、`--neg`、`--warn`、`--info` | 各自带 `-subtle`（半透明叠色）、`-bg`（叠在 `--surface` 上的不透明浅底）与 `-border`（派生） |
| 其他 | `--purple`、`--magenta`、`--cyan`、`--lime`、`--green-deep` | 能力与类型颜色 |

`--hairline-strong`、四个 `-border`、`--neg-bg`、`--warn-bg` 与 `--info-bg` 是用
`color-mix()` 从角色色算出的表达式 —— 与 `components.css` 生成浅底/边框的写法相同 —— 所以主题加上它们并不新增颜色值。它们是暂定值：目前没有组件绘制它们，第一个真正上屏的阶段（`P3`/`P4`）会用主题矩阵按实际对比度调整比例。

字号阶梯、间距与外壳几何（`--fs-*`、`--tracking-label`、`--space-1..6`、
`--side-w`、`--topbar-h`）不随主题变化，放在 `base.css` 的 `:root`。

### 兼容别名

P1 把取值搬到了上面的名字下，并把原来的名字（`--bg-card`、`--bg-hover`、
`--border`、`--green`、`--green-subtle`、`--green-bg`、`--red`、`--red-subtle`、
`--orange`、`--blue`、`--badge-radius`）留在每个主题文件末尾的一个别名块里，因此没有组件改变行为。这个块是临时的：**P5 删除它**，而防止它变成第二套 API 的规则是一次 grep —— 在该块之外，`admin/` 里不得残留任何旧名字。

深色块可以在深色背景需要不同强度时重新声明 `--dot` 与 `--grid`（emerald 就是这样）。除此之外，它不应重新声明结构块里的值——除非那确实是模式之间的差异，而不是方案之间的差异。

## 添加一个配色方案

矩阵默认的方案轴就是注册表的清单，因此新加的主题下次运行就会被覆盖；迭代单个方案时用 `--schemes` 收窄范围。下面这份清单就是全部改动面。

1. **写 `admin/css/themes/<name>.css`。** 一个共享块（`[data-scheme="<name>"]`）和每个模式一个块
   （`[data-scheme="<name>"][data-mode="light"]`、
   `[data-scheme="<name>"][data-mode="dark"]`）。声明的 token 名字要和默认主题一致——`tests/gateway/test_admin_themes.py` 会按模式比较两个主题可用的 token 集合，漏掉一个就失败——并且只声明取值：契约测试拒绝方案块里的真实属性。当取值确实随模式变化时，某个模式可以重新声明共享 token（emerald 的 `--dot`/`--grid`）。
2. **在 `admin/themes.py` 注册。**

   ```python
   Theme("<name>", "<Label>", "css/themes/<name>.css"),
   ```

   这一条就是引导脚本收到的内容、设置下拉框列出的内容、矩阵枚举的方案，以及 wheel 打包的（`css/**/*.css`）。没有第二份清单要改。
3. **（可选）在两种语言字典里加标签。** `scheme.<name>` 加进 `admin/js/core/i18n.js` 的英文与中文字典；不加的话，下拉框显示注册表里的标签。中文字典把非 ASCII 字符存成字面量 `\uXXXX` 转义，请用不会吞掉反斜杠的工具编辑。
4. **跑检查并附上截图。**

   ```bash
   make test-js      # 契约：方案可以写什么
   make theme-matrix # 渲染：scheme × mode × width，带截图
   ```

   `make theme-matrix` 会写出 `theme-matrix-out/report.md`，其中内嵌截图。把报告或图片放进 PR。

## 验证主题改动

| 检查 | 命令 | 覆盖什么 | 不覆盖什么 |
|------|------|----------|------------|
| 契约测试 | `make test-js` | 方案可以用哪些选择器和声明；装饰性伪元素是否抢点击；`@media` 嵌套 | 值渲染出来是否可读；运行时是否缺 token |
| 主题矩阵 | `make theme-matrix` | 对真实运行的面板跑遍每个 scheme × mode × width：方案已生效、每个方案 token 都能解析、每个页签都没有横向滚动、Token 卡片的详情链接可点，且不同方案/模式确实渲染出不同结果；Dashboard 与 Providers 两个页签的截图 | 截图范围之外的回归；像素级变化 |

只要网关包、JS 测试或矩阵工具本身有改动，CI 的 `theme-matrix` 作业就会运行。它没有像素基线，因此不受字体和 runner 镜像影响；失败的是行为或结构探针，截图留给评审的人看。

要把矩阵跑在带真实数据的网关上（Dashboard 数字更真实）：

```bash
python -m tools.theme_matrix --base-url http://127.0.0.1:8765 \
    --admin-password "$THEME_MATRIX_ADMIN_PASSWORD"
```

矩阵工具位于 `tools/theme_matrix/`，其 README 完整说明探针与选项。
