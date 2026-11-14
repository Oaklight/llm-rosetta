# 石形标记

本仓库里每一份已发布的 logo，都来自同一条描摹出来的罗塞塔石碑轮廓。各副本只在取景、尺寸和填充色上不同。这个文件夹里是用于从源路径重新生成这些文件的工具。

## 来源

`references/rosetta-stone-path.txt` 就是那条描摹路径，也是 `out/` 里每个文件的输入：一条闭合路径、126 段三次曲线、4,726 字节的 `d` 数据，包围盒 451.5 x 589.6 用户单位。

轮廓是对 Wikimedia Commons 上一个文件的算法描摹：[File:Rosetta Stone.svg](https://commons.wikimedia.org/wiki/File:Rosetta_Stone.svg)，作者 kinetoons（2014，来自 open clip art），以 CC0 1.0 发布。`references/SOURCE.md` 记录了文件、作者、许可证和注意事项。

CC0 不要求署名，所以这个标记本身不附带任何许可义务。形状是派生出来的描摹，上面的来源链记录了它的来历。如果这个标记将来被注册，或者项目被出售，仍应从有记录的来源重新派生，因为链条顶端的 CC0 并不会清除上传者从未持有的下游权利。

## 标准参数

| 用途 | 文件 | points | anchors | 取景 | 填充 |
| --- | --- | --- | --- | --- | --- |
| 应用图标 | `out/rosetta-icon.svg` | 48 | 12 | 600 画布，scale 0.88 | `#1e293b` / `#ededed` |
| README 深色 | `out/rosetta-icon-github.svg` | 48 | 12 | 600 画布，scale 0.88 | `#1e293b` / `#ededed` |
| README 浅色 | `out/rosetta-icon-docs.svg` | 48 | 12 | 600 画布，scale 0.88 | 仅 `#1e293b` |
| favicon | `out/favicon.svg` | 28 | 10 | 紧贴方形，pad 0.04 | `#666666` / `#ededed` |

`points` 是顶点预算。`anchors` 是永远不能被简化掉的结构顶点数量，按“删掉它会损失多少三角形面积”排序。超过约 64 个顶点后，标记在视觉上就不再变化，所以决定小尺寸标记是否可读的，是保留下来的是哪些顶点。

## 已发布副本（实测）

| 位置 | 文件 | 取景 | 填充 |
| --- | --- | --- | --- |
| README（中/英） | `design/logo/out/rosetta-icon-{github,docs}.svg` | 600 方框，scale 0.88 | `#1e293b` / `#ededed`；docs 仅浅色 |
| 各包 README | `out/rosetta-icon-github.svg` | 600 方框，scale 0.88 | `#1e293b` / `#ededed` |
| docs 站头 | `docs_{en,zh}/docs/images/rosetta_stone.svg` | 600 方框，scale 0.934 | `currentColor` |
| docs favicon | `docs_{en,zh}/docs/images/favicon.svg` | 600 方框，scale 0.934 | `#1a1a1a` / `#ededed` |
| 网关 favicon | `admin/routes/auth.py` 的 `_FAVICON_SVG` | viewBox `30 15 490 500` | `#666` |
| 网关侧栏/顶栏 | `admin/admin.html` 内联 `<svg>` | viewBox `30 15 490 500` | `currentColor` |

README 的深浅两份过去是逐字节相同的，`prefers-color-scheme` 切换会渲染同一个文件两次。现在浅色那份是纯浅色文件，深色那份保留自己的 media query。各包的 README 直接引用深色文件，它们仍能自行适应。

网关 favicon 裁剪到 viewBox `30 15 490 500`，而轮廓的 y 范围是 3.3..593.0，所以石头底部约 78 单位（13%）被切掉。这就是网关 favicon 与 README 标记对不上的原因。换成 `out/favicon.svg`（紧贴方形、4% 内边距）属于网关代码改动，留给单独的 PR。

`out/favicon-gateway.svg` 是网关当前实际提供内容的存档副本，这样裁剪在仓库里可见，而不只是藏在 `auth.py` 的一段字节串里。

## 工具

`simplify_path.py`（CLI，仅标准库）和 `simplify.html`（交互页面，不联网）跑同一条流水线，相同参数下产出逐字节相同的几何：

1. 解析路径；自适应展平三次曲线（`--tolerance`）；
2. 选取结构锚点（`--anchor`、`--anchor-mode area|corner|depth|hybrid`）；
3. 在不丢锚点的前提下压到顶点预算（`--points`、`--method visvalingam|douglas`）、
   按最大偏差裁剪（`--method fidelity`、`--fidelity`），或在同一误差上限下做边折叠
   （`--method decimate`）；
4. 可选的 Chaikin 圆角（`--smooth`）；
5. 形变：`--rotate`、`--aspect`、`--inflate`；
6. 取景：方形画布（`--canvas`、`--scale`）或紧贴方形（`--tight`、`--pad`）。

`--method fidelity` 不使用顶点预算。它删掉所有“删掉后轮廓偏差仍不超过 `--fidelity`
× 原轮廓最长边”的顶点，误差是逐点对未简化路径测量的。被固定的顶点原样保留，并把
环切成若干独立弧段，所以标记能守住选定的角，其余轮廓跟随保真设置。当顶点数应当由
形状决定、而不是反过来时，用它。

`--method decimate` 对应三维网格的保形削减：反复折叠代价最小的边，每次折叠后重算两个
邻居的代价，任何会让轮廓偏离超过 `--fidelity` 的折叠都拒绝。顶点预算仍然有效，哪个先
到就停。填充图形内部没有三角形可减，所以折叠作用在轮廓上；在石头上，同样误差下它与
`--method fidelity` 只差一两个顶点，分布略均匀，因为代价是每次折叠后全局重算的。

锚点模式对顶点的排序依据不同。`area` 用删掉该点会损失的三角形面积（偏好大的外凸弧线），`corner` 用转角（偏好尖角），`depth` 用到凸包内部的深度（偏好凹陷，因为凹口正是轮廓离开凸包的地方），`hybrid` 一半按面积、其余按深度补齐。在这条轮廓上，展平后有 46% 的顶点局部是反折，但 `area` 只选出 12 个中的 1 个，`corner` 选出 7 个，`depth` 选出 12 个且全部不在凸包上。

红色锚点在命令行里写成 `--lock "u0.3738@214.01,85.47"`。引用的是**轮廓长度的比例**，不是顶点下标：下标在 `--tolerance` 改变展平密度后指向完全不同的位置，这正是锚点会在轮廓上乱跳的原因。`--pin` 用同一种引用，纯整数仍按旧的下标形式读。锁定会把这个顶点拉进结果（即使预算要丢掉它），`214.01,85.47` 是它被固定住的画布坐标。形变滑块以红色锚点的重心为原点旋转、拉伸、膨胀，所以拖滑块时动的是轮廓其余部分，锚点原地不动；取景再把这个重心映射回标注时的画布点。多个锚点用 `;` 分隔。

```bash
cd design/logo

# 应用图标
python simplify_path.py references/rosetta-stone-path.txt \
    -o out/rosetta-icon.svg --points 48 --anchor 12 --scale 0.88

# README 两份：纯浅色 + 自适应
python simplify_path.py references/rosetta-stone-path.txt \
    -o out/rosetta-icon-docs.svg --points 48 --anchor 12 --scale 0.88 \
    --fill '#1e293b' --dark-fill ''
python simplify_path.py references/rosetta-stone-path.txt \
    -o out/rosetta-icon-github.svg --points 48 --anchor 12 --scale 0.88

# favicon：紧贴方形、4% 内边距、跟随主题
python simplify_path.py references/rosetta-stone-path.txt \
    -o out/favicon.svg --points 28 --anchor 10 --tight --pad 0.04 \
    --fill '#666666'

# 几个预算的开销对比
python simplify_path.py references/rosetta-stone-path.txt --points 16,32,48,64 --stats
```

## 草稿与局域网使用

页面把草稿存在 `localStorage`（保存草稿 / 载入草稿），以便下次继续；导出草稿 / 导入草稿 用来在机器之间搬移这份 JSON。草稿包含全部参数、源路径、手工锚点的固定与取消，以及视图缩放。

让别人从局域网里的另一台机器打开页面：

```bash
cd design/logo
python3 -m http.server 8771 --bind 0.0.0.0
# 然后打开 http://<lan-ip>:8771/simplify.html
```

除了静态文件没有服务端，草稿不会离开浏览器，除非你主动导出。

请通过任意静态服务器打开 `simplify.html`（`python -m http.server`），不要用 `file://`，这样剪贴板和下载功能才能正常工作。

交互：

- 点击顶点把它设为橙色锚点，点击橙色锚点取消。空心点是当前预算会丢弃的候选点，它们仍可点击，所以锚点在任何预算下都能取下再装回。
- 右键顶点附近即可设为红色绝对锚点，点击会吸附到几像素内最近的顶点，画布上不会弹出浏览器右键菜单。锚点按轮廓上的位置记录，所以改「曲线精度」时它仍停在轮廓的同一处，不会跳到别的顶点。它和橙色锚点一样不会被简化，并且被固定住：形变滑块以锚点重心为原点旋转、拉伸、膨胀，取景再把这个重心映射回你标注时的画布点；圆角平滑也不会切掉这个角。再右键一次或左键点红点即可解除。
- Ctrl + 滚轮以光标为中心缩放。放大后拖动即可平移；100% 时图形正好铺满画框，拖动只会提示这一点而不会移动。双击或 重置视图 按钮复位。复位同时把画布重新居中：红色锚点按居中后的布局重新取点，图形回到中间而不是留在偏移位置。
- 页眉的 重置 恢复全部参数、清空手工锚点与红色锚点并复位视图。
- 主题 开关用对应主题的填充色重绘预览：深色页面显示浅色标记，浅色页面显示深色标记，和存档 SVG 的配对一致。导出的文件仍然保留 `prefers-color-scheme` media query，所以在任何地方都能自适应。
- 宽屏下控件排成三列（简化、预览、形变）。低于 1500 px 收成两列，低于 980 px 收成一列，源路径面板默认收起。
- 画布边长只改变导出的坐标系。屏幕上的大小由 `scale` 决定。

## 版权

源文件是 [File:Rosetta Stone.svg](https://commons.wikimedia.org/wiki/File:Rosetta_Stone.svg)，其作者以 CC0 1.0 发布，因此这条描摹不带署名或 share-alike 义务。文物的年代不决定这件事：标记是对某一张具体图像的描摹，而描摹通常被视为那张图像的演绎作品。CC0 是这条来源链最终没有义务的原因。来源链和它唯一的注意事项记录在 `references/SOURCE.md`。

这个文件夹里的文件不是 CC0。它们是本项目的标记，条款见本目录的 [`LICENSE`](LICENSE) 和 [`TRADEMARKS.md`](../../TRADEMARKS.md)。简而言之：可用于指代本项目并保留声明；不要用作产品或服务的品牌，不要主张所有权，也不要注册成自己的。
