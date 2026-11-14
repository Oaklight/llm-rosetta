# 石形标记

本仓库里每一份已发布的 logo，都来自同一条描摹出来的罗塞塔石碑轮廓。各副本只在取景、尺寸和填充色上不同。这个文件夹里是用于从源路径重新生成这些文件的工具。

## 来源

`references/rosetta-stone-path.txt` 就是那条描摹路径，也是 `out/` 里每个文件的输入：一条闭合路径、126 段三次曲线、4,726 字节的 `d` 数据，包围盒 451.5 x 589.6 用户单位。

轮廓是对 Wikimedia Commons 上一个文件的算法描摹：[File:Rosetta Stone.svg](https://commons.wikimedia.org/wiki/File:Rosetta_Stone.svg)，作者 kinetoons（2014，来自 open clip art），以 CC0 1.0 发布。`references/SOURCE.md` 记录了文件、作者、许可证和注意事项。

CC0 不要求署名，所以这个标记本身不附带任何许可义务。形状是派生出来的描摹，上面的来源链记录了它的来历。如果这个标记将来被注册，或者项目被出售，仍应从有记录的来源重新派生，因为链条顶端的 CC0 并不会清除上传者从未持有的下游权利。

## 标准参数

设计就是一份草稿：`references/rosetta-draft.json` 是页面「导出草稿」写出的东西，两个前端都跑它。命令行用 `--draft` 读入，重新生成一份副本不必把十几个参数重打一遍，也就不会打错。

```bash
cd design/logo
python simplify_path.py --draft references/rosetta-draft.json -o out/rosetta-icon.svg
python simplify_path.py --draft references/rosetta-draft.json -o out/rosetta-icon-github.svg
python simplify_path.py --draft references/rosetta-draft.json --dark-fill '' -o out/rosetta-icon-docs.svg
python simplify_path.py --draft references/rosetta-draft.json --fill '#666666' -o out/favicon.svg
```

`--draft` 旁边一起写的参数会覆盖草稿，四份副本的差别因此只在填充色上。草稿要求 15 个顶点（保护先把预算盖过）、15 个结构锚点、`--tolerance 0.52` 展平、一轮 `--smooth 1` 圆角、4% 内边距的紧贴方形取景；另外带 7 个固定点、3 个取消点、4 个红色锚点、2 个锐点和 6 个圆点，全部按轮廓上的位置命名。

| 用途 | 文件 | 填充 |
| --- | --- | --- |
| 应用图标 | `out/rosetta-icon.svg` | `#1e293b` / `#ededed` |
| README 深色 | `out/rosetta-icon-github.svg` | `#1e293b` / `#ededed` |
| README 浅色 | `out/rosetta-icon-docs.svg` | 仅 `#1e293b` |
| favicon | `out/favicon.svg` | `#666666` / `#ededed` |

`points` 是顶点预算。`anchors` 是永远不能被简化掉的结构顶点数量，按“删掉它会损失多少三角形面积”排序。在这份草稿上起作用的是保护而不是预算：15 个结构锚点、7 个固定点、4 个红色锚点和锐点的细节合计 34 个顶点（展平后共 263 个），比预算要的 15 个还多，所以简化把这 34 个全部留下，再经过一轮圆角就是文件里的 128 个点。

## 已发布副本（实测）

| 位置 | 文件 | 填充 |
| --- | --- | --- |
| README（中/英） | `design/logo/out/rosetta-icon-{github,docs}.svg` | `#1e293b` / `#ededed`；docs 仅浅色 |
| 各包 README | `out/rosetta-icon-github.svg` | `#1e293b` / `#ededed` |
| docs 站头 | `docs_{en,zh}/docs/images/rosetta_stone.svg` | `currentColor` |
| docs favicon | `docs_{en,zh}/docs/images/favicon.svg` | `#1a1a1a` / `#ededed` |
| 网关 favicon | `admin/routes/auth.py` 的 `_FAVICON_SVG` | `#666666` / `#ededed`，紧贴方形 `635.22` |
| 网关侧栏/顶栏 | `admin/admin.html` 内联 `<svg>` | `currentColor`，600 方框 |

README、favicon 和 docs 这三处都由它生成，差别只在填充色，docs 的两份落在 docs 工作树里：

```bash
python design/logo/simplify_path.py \
    --draft design/logo/references/rosetta-draft.json \
    --fill 'currentColor' --dark-fill '' -o docs_en/docs/images/rosetta_stone.svg
python design/logo/simplify_path.py \
    --draft design/logo/references/rosetta-draft.json \
    --fill '#1a1a1a' -o docs_en/docs/images/favicon.svg
# docs_zh/docs/images/ 下同样两份
```

README 的深浅两份过去是逐字节相同的，`prefers-color-scheme` 切换会渲染同一个文件两次。现在浅色那份是纯浅色文件，深色那份保留自己的 media query。各包的 README 直接引用深色文件，它们仍能自行适应。

网关是例外。它的 favicon 已经是紧贴方形（`635.22`），但用的是旧的 48 点描摹；`admin.html` 里那两处内联标记仍是 600 方框。三处都没有草稿的 128 个点。换过来属于网关代码改动，留给单独的 PR。

`out/favicon-gateway.svg` 是网关当前实际提供内容的存档副本，这样它实际发布的标记在仓库里可见，而不只是藏在 `auth.py` 的一段字节串里。

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

锚点模式对顶点的排序依据不同。`area` 用删掉该点会损失的三角形面积（偏好大的外凸弧线），`corner` 用转角（偏好尖角），`depth` 用到凸包内部的深度（偏好凹陷，因为凹口正是轮廓离开凸包的地方），`hybrid` 一半按面积、其余按深度补齐。在随仓库的 `--tolerance 0.52` 下，263 个展平顶点里有 212 个是凹陷，但 `area` 的 15 个里只有 4 个落在凹陷，`corner` 8 个，`depth` 15 个全部。

红色锚点在命令行里写成 `--lock "u0.3738@214.01,85.47"`。引用的是**轮廓长度的比例**，不是顶点下标：下标在 `--tolerance` 改变展平密度后指向完全不同的位置，这正是锚点会在轮廓上乱跳的原因。`--pin` 用同一种引用，纯整数仍按旧的下标形式读。锁定会把这个顶点拉进结果（即使预算要丢掉它），`214.01,85.47` 是它被固定住的画布坐标。形变滑块以红色锚点的重心为原点旋转、拉伸、膨胀，所以拖滑块时动的是轮廓其余部分，锚点原地不动；取景再把这个重心映射回标注时的画布点。多个锚点用 `;` 分隔。

**局部质感。** `--sharp u0.10` 与 `--round u0.60` 只改轮廓某一段的转角处理，而不是全图。锐点中心不做圆角，这一段保持描摹时的原样，并额外保留标记点以及附近每隔一个顶点——轮廓上约 3% 的一段顶点数大约翻倍；这些额外顶点只参与减点保护，且锐点这一段不提供任何候选顶点——页面上只有那个紫色标记，所以不会被圆点挤满。圆点则在 `--smooth` 之上再加两轮圆角。两者都沿轮廓在 4.5% 的距离内渐变回全局值，所以是过渡而不是突变。页面用「左键模式」（锚点 / 锐 / 圆）标注，分别画成紫色和青色。在随仓库的草稿上，这两种处理占掉了标记的大部分细节：在 `--draft` 旁边写 `--sharp ''` 去掉两个锐点，128 个顶点只剩 82 个；写 `--round ''` 去掉六个圆点，只剩 42 个。这两个参数是替换草稿里的列表而不是叠加，对比就是这么得到的。

**居中。** `--center bbox`（默认）把外接框放到画布中心，`--center centroid` 放的是填充图形的面积重心。两者在这块石头上不一样，因为它下宽上窄：按草稿的取景，外框中心是 `(317.4, 317.4)`，而填充图形的重心在 `(308.2, 331.5)`，低了 14 个单位、偏左 9 个。页面用「居中基准」（中心 / 重心）切换，导出前点「居中」把它落定——红色锚点记录的画布点会一起平移，所以仍然贴在原来的顶点上。

**碑文。** Commons 那份文件是石头加刻字：一条黑色轮廓，上面叠着 22,040 条白色路径。`incisions.py` 把这些路径单独抽成一层，坐标系与轮廓相同：

```bash
python design/logo/incisions.py ~/Downloads/rosettastone.svg \
    -o /tmp/rosetta-incisions.svg
```

两位小数下这一层有 5.8 MB，比全部已发布副本加起来还大三个数量级，所以按需生成、不入库。`simplify_path.py --incisions FILE` 用它把标记挖开：路径变成 `<mask>`，多边形透过它填充，于是刻痕显示的是文件所在处的底色而不是只能对上某一个背景的颜色；又因为与轮廓同一套用户坐标，它跟着形变走。页面「碑文」面板可以贴网址直接载入（Commons 那份返回 `access-control-allow-origin: *`，可以从 Wikimedia 直接取；把文件放到 GitHub 上用 jsDelivr 镜像同样可以），也可以选本地文件；勾选「显示碑文」后，刻痕随形变与取景移动、裁在轮廓内，主视图和四个尺寸预览都带。轮廓按源文件的**第一条** path 识别——最长的那条其实是某个刻痕字形；`incisions.py` 生成的层只有一条 path，直接整条使用。层只留在内存：刷新要重新载入，也不会写进草稿（会撑爆 `localStorage` 配额）。图标尺寸下它读作纹理，不是字。

5.8 MB 太大的时候可以减。在 8.2 MB 的源文件、22,149 条刻痕上实测：

| 参数 | 层大小 | 刻痕数 |
| --- | --- | --- |
| *（无）* | 5.82 MB | 22,149 |
| `--simplify 5` | 1.72 MB | 22,149，每条轮廓压到 5 个顶点 |
| `--step 3` | 1.92 MB | 7,383 |
| `--min-area 0.5` | 4.90 MB | 12,655 |
| `--abstract` | 0.68 MB | 22,149，每条刻痕换成一个等面积方块 |
| `--abstract --step 3` | 0.23 MB | 7,383 |

`--abstract` 放弃字形、保留质量与位置——16–64 px 下留下的正是这些；`--simplify` 保留字形、削减顶点——大尺寸渲染要的是这个。`--min-area` 与 `--step` 负责删减条数，可与两者叠加。

**方言。** 石碑把同一段诏书刻了三遍，`json_incisions.py` 用同样的办法把三种 LLM 请求方言排进碑的三个区带——用的是这块石头唯一的字母表：每个不同字符固定映射到一个真实刻痕字形，所以同一个字符永远长得一样，成品是从石头自己的字形词汇里切出来的，而不是来自某套字体。映射由哈希决定，同样的输入永远得到同一层。

```bash
python design/logo/json_incisions.py \
    --source ~/Downloads/rosettastone.svg \
    --bands design/logo/references/dialects/chat_completions.json \
            design/logo/references/dialects/messages.json \
            design/logo/references/dialects/google_generate.json \
    -o /tmp/dialects.svg
```

`references/dialects/` 里的三份载荷是同一条任务的 Chat Completions、Messages 与 `generateContent` 请求；任何文本都可以，一到三份文件，每份一个区带。产出的层与描摹本身的刻痕同格式——一条 path、同一坐标系——所以 `--incisions` 和页面直接就能用，不需要特殊分支。用随仓库的三份载荷实测：26 行、661 个刻痕、0.13 MB。

```bash
cd design/logo

# 几个预算的开销对比
python simplify_path.py references/rosetta-stone-path.txt --points 16,32,48,64 --stats
```

## 草稿与局域网使用

页面把草稿存在 `localStorage`（保存草稿 / 载入草稿），以便下次继续；导出草稿 / 导入草稿 用来在机器之间搬移这份 JSON。草稿包含全部参数、源路径、手工锚点的固定与取消，以及视图缩放。

同一个文件也是命令行的输入：`--draft FILE` 读入它，写在 `--draft` 旁边的参数覆盖草稿里的值，所以四份已发布副本的差别只在填充色。设计由页面决定，命令行负责重放。

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
- 「左键模式」决定左键落什么：锚点（橙，简化时保留）、锐（紫，不做圆角并保留附近更多顶点）或圆（青，额外两轮圆角）。同一个顶点上的锐与圆互相替换，两者都像锚点一样不会被丢掉。
- 右键顶点附近即可设为红色绝对锚点，点击会吸附到几像素内最近的顶点，画布上不会弹出浏览器右键菜单。锚点按轮廓上的位置记录，所以改「曲线精度」时它仍停在轮廓的同一处，不会跳到别的顶点。它和橙色锚点一样不会被简化，并且被固定住：形变滑块以锚点重心为原点旋转、拉伸、膨胀，取景再把这个重心映射回你标注时的画布点；圆角平滑也不会切掉这个角。再右键一次或左键点红点即可解除。
- Ctrl + 滚轮以光标为中心缩放。放大后拖动即可平移；100% 时图形正好铺满画框，拖动只会提示这一点而不会移动。双击或 重置视图 按钮复位。复位同时把画布重新居中：红色锚点按居中后的布局重新取点，图形回到中间而不是留在偏移位置。
- 「居中基准」决定 居中 对齐的是什么——中心（外接框）或 重心（填充图形的面积重心）；导出前点 居中 应用，红色锚点记录的画布点随之平移，仍贴在原来的顶点上。
- 页眉的 重置 恢复全部参数、清空手工锚点与红色锚点并复位视图。
- 主题 开关用对应主题的填充色重绘预览：深色页面显示浅色标记，浅色页面显示深色标记，和存档 SVG 的配对一致。导出的文件仍然保留 `prefers-color-scheme` media query，所以在任何地方都能自适应。
- 宽屏下控件排成三列（简化、预览、形变）。低于 1500 px 收成两列，低于 980 px 收成一列，源路径面板默认收起。
- 画布边长只改变导出的坐标系。屏幕上的大小由 `scale` 决定。

## 版权

源文件是 [File:Rosetta Stone.svg](https://commons.wikimedia.org/wiki/File:Rosetta_Stone.svg)，其作者以 CC0 1.0 发布，因此这条描摹不带署名或 share-alike 义务。文物的年代不决定这件事：标记是对某一张具体图像的描摹，而描摹通常被视为那张图像的演绎作品。CC0 是这条来源链最终没有义务的原因。来源链和它唯一的注意事项记录在 `references/SOURCE.md`。

这个文件夹里的文件不是 CC0。它们是本项目的标记，条款见本目录的 [`LICENSE`](LICENSE) 和 [`TRADEMARKS.md`](../../TRADEMARKS.md)。简而言之：可用于指代本项目并保留声明；不要用作产品或服务的品牌，不要主张所有权，也不要注册成自己的。
