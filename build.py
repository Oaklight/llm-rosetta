import html, pathlib, re, subprocess
d = pathlib.Path(".")
def src(n): return html.escape((d / f"{n}.mmd").read_text(encoding="utf-8"), quote=False)
sections = [
 ("core", "1. core 现状:组件与依赖",
  "实线是模块顶层导入,虚线是函数内的懒加载。另一条路径已经隐含的实线导入没有画(传递性约简),带标签和带颜色的线一律保留;完整的边用 python gen_core.py --full 生成。ir、pipeline、formats、auto_detect、contracts 都不导入 shims 与 extensions,shim 名字经 contracts/resolver.py 读取。四个编排器(chat、embeddings、rerank、decisions)都在 pipeline/,converters 不导入 pipeline,ConversionError 只定义在 converters/base/errors.py。IR 是一个包 llm_rosetta.ir:ir/model(数据,只依赖标准库)在底层,ir/primitives、ir/tools、ir/passes 在它之上;types/ 只放提供方参考形状。shims 向 extensions 注册 shims 处理器(绿线),两者不互相导入。红色的依赖只有一条:contracts/transform.py 导入 ir.primitives(边表由 tests/test_core_layering.py 固定)。“转换库”组是三层:pipeline 在最上,converters 在中间,IR 在最下,自上而下依赖;没有 shim 时这一组就是完整的转换库。shims 和 extensions 是它外面的可选层;contracts 在共享基础层,被转换、shim、扩展、cli、devtools 共用;api 是宿主门面,core 内部不导入它(有测试保证)。api 只是把下面各包再导出一遍,这些再导出的边不画,免得遮住真正的依赖。tool_ops(入口便捷层)只画到 converters 的边,它到 ir 和 retired_names 的边在 CORE_EDGES 里有,图里不画。未画的:shims/providers 转发层、弃用的 gateway/ 与 observability/ 转发包。画的是 master 的树。"),
 ("gateway", "2. gateway 后端的组件与关系",
  "分层来自 tests/gateway/test_layers.py,只能向下导入;另一条路径已经隐含的包间导入没有画(传递性约简),带标签的线保留,完整的边用 python gen_gateway.py --full 生成;包间的边直接取自该文件的 GATEWAY_EDGES(由 gen_gateway.py 生成)。每个包都会用 core,图里只画了几条代表性的到 CORE 的线。admin 除了 admin_api 之外还有一批 legacy allowlist 导入(config、keystore、maintenance、middleware、observability、ops、quota、routing、shims、transport、updates),名单只允许缩小,图里没画。"),
 ("admin", "3. admin 页面与组件的关系",
  "已核对:页面→脚本、脚本→API 分组(来自各 JS 里的 /admin/api 调用)、9 个标签页及子区域(admin.html)。未核对:右半段“API 分组→后端组件”按名字推断;settings 页的 JS 接线只确认了文件;components/ 是否也被 models 等其他页使用。"),
("gap_actual", "4. gateway 现实:当前的完整架构与仍偏离设计的地方",
  "和第 2 页同一批节点、同一张包间导入表,红线是 #1830 评审量出来的、还没解决的问题:A1 admin 的边界:admin 只导入 admin、admin_api 和 _vendor.httpserver(纯规则的测试),但路由里仍有 107 处读 request.app.<attr>(61 对,shrink-only 的扫描测试;热重载是 AdminContext.reload_config,已生效配置只有一个家 app.gateway_config,但 admin 路由里还有直接读它的);ProviderInfo 的 58 个属性是已知现状,决定不拆。读取数来自 tests/gateway/test_admin_app_reads.py 的 _APP_READS(origin/master c72f32a265 之后)。 为了读得清楚,这一页不画能由其他路径推出的包间依赖(传递约简,A→B→C 已经隐含 A→C),红线和带标签的边总是画出;完整的边表在第 2 页,布局用 ELK。"),
("gap_ideal", "5. gateway 理想:最终的架构",
  "同一批节点,没有红线。admin 只经 AdminContext;热重载和已生效配置在 app 里,审计写入只经 ops;上游尝试的记账是 proxy 里的一个 attempt_upstream;ACP lane 解析在 transport/acp/lane.py 里,ACP 是 transport/ 下的一个文件夹(和 http/ 对应);quota 的读数和锁在 quota/(每个 app 一个读数库,重载后重新绑定到 provider)。这是目标态,每一项落地后,第 4 页对应的红线就去掉。 为了读得清楚,这一页不画能由其他路径推出的包间依赖(传递约简,A→B→C 已经隐含 A→C),红线和带标签的边总是画出;完整的边表在第 2 页,布局用 ELK。",
  "目标 · #1830", ""),
]

try:
    master = subprocess.run(["git", "-C", "..", "rev-parse", "--short", "origin/master"], capture_output=True, text=True, check=True).stdout.strip()
except Exception:
    master = "master"

def note_html(text):
    # one sentence per list item; the first sentence stays the lede
    parts = [p.strip() for p in re.split(r"(?<=。)", text) if p.strip()]
    lede = html.escape(parts[0])
    items = "".join(f"<li>{html.escape(p)}</li>" for p in parts[1:])
    return f'<p class="sub">{lede}</p>' + (f'<details class="notes"><summary>说明({len(parts) - 1} 条)</summary><ul>{items}</ul></details>' if items else "")

blocks = ""
toc = ""
for n, (key, title, note, *extra) in enumerate(sections, 1):
    tag_text, tag_class = extra if extra else (f"现状 · master @ {master}", "now")
    label = title.split(". ", 1)[1]
    toc += f'<a href="#{key}">{n:02d} {html.escape(label)}</a>'
    blocks += f"""
<section id="{key}">
  <h2><span class="num">{n:02d}</span>{html.escape(label)}</h2>
  {note_html(note)}
  <div class="panel">
    <span class="tag {tag_class}">{html.escape(tag_text)}</span>
    <div class="zoom"><label>缩放 <input type="range" min="40" max="200" value="100" step="5" data-target="{key}"> <span id="z-{key}">100%</span></label></div>
    <div class="scroll"><div class="stage" id="stage-{key}"><pre class="mmd-src" id="m-{key}">{src(key)}</pre></div></div>
  </div>
</section>"""

mansio_css = pathlib.Path(__file__).with_name("page.css").read_text(encoding="utf-8")
page = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>llm-rosetta 架构:现状</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
""" + mansio_css + """
</style></head>
<body>
<div class="wrap">
<header>
  <div class="eyebrow">LLM-Rosetta · Architecture</div>
  <h1>现状:core、gateway 与 admin 的组件和依赖,以及 gateway 的理想与现实</h1>
  <p class="lede">前三张图画 master 的实际结构,不画目标态;第 4、5 张把 #1830 评审出的问题画成“现实”与“理想”两张对照图。红色表示还没解决的反向依赖,绿色表示 core 与扩展之间的接线,虚线是函数内的懒加载。合并任一分层 PR 后,本页按 master 更新。</p>
  <div class="meta">
    <span>现状快照:<code>master @ """ + master + """</code></span>
    <span>分支:<code>docs_arch</code></span>
    <button id="themeBtn" type="button">深色主题</button>
  </div>
  <div class="legend">
    <span><i></i>模块顶层导入</span>
    <span><i class="dash"></i>函数内懒加载</span>
    <span><i class="red"></i>有问题的依赖(循环或反向)</span>
    <span><i class="green"></i>接线(注册处理器)</span>
  </div>
</header>
<nav class="toc">""" + toc + """</nav>
""" + blocks + """
<footer><span>由 <code>gen_core.py</code> 生成 core 图,<code>build.py</code> 生成本页;mermaid 在浏览器里渲染。</span><span>分支 <code>docs_arch</code></span></footer>
</div>
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
try {
  const elk = (await import("https://cdn.jsdelivr.net/npm/@mermaid-js/layout-elk@0/dist/mermaid-layout-elk.esm.min.mjs")).default;
  mermaid.registerLayoutLoaders(elk);
} catch (e) { console.warn("ELK layout unavailable, falling back to dagre: " + e); }
window.mermaid = mermaid;
(function () {
  var KEY = "llm-rosetta-arch-theme", btn = document.getElementById("themeBtn");
  var panes = Array.prototype.map.call(document.querySelectorAll("pre.mmd-src"), function (pre) {
    return { id: pre.id.replace("m-", ""), src: pre.textContent };
  });
  function preferred() {
    var s = null; try { s = localStorage.getItem(KEY); } catch (e) {}
    if (s === "light" || s === "dark") return s;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function apply(t) { document.documentElement.setAttribute("data-theme", t); btn.textContent = t === "dark" ? "浅色主题" : "深色主题"; }
  function zoomTo(id, v) {
    document.getElementById("stage-" + id).style.zoom = v / 100;
    document.getElementById("z-" + id).textContent = v + "%";
    document.querySelector('input[data-target="' + id + '"]').value = v;
  }
  var manual = {};
  async function render(theme) {
    document.body.dataset.done = "";
    mermaid.initialize({ startOnLoad: false, securityLevel: "loose", theme: theme === "dark" ? "dark" : "default",
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif',
      flowchart: { useMaxWidth: false, htmlLabels: true, curve: "basis", nodeSpacing: 40, rankSpacing: 60 } });
    try { await mermaid.parse("flowchart TB; a --> b"); } catch (e) {}
    for (var p of panes) {
      var stage = document.getElementById("stage-" + p.id);
      try {
        var r = await mermaid.render("svg-" + p.id + "-" + Date.now(), p.src);
        stage.innerHTML = '<div class="rendered" data-ok="1">' + r.svg + '</div>';
        if (!manual[p.id]) {
          var box = stage.parentElement;
          var w = stage.querySelector("svg").getBoundingClientRect().width / (Number(stage.style.zoom) || 1);
          zoomTo(p.id, Math.max(40, Math.min(100, Math.floor(((box.clientWidth - 28) / w) * 20) * 5)));
        }
      } catch (e) {
        stage.innerHTML = '<pre class="error" data-ok="0">' + String(e) + '</pre>';
      }
    }
    document.body.dataset.done = "1";
  }
  document.querySelectorAll("input[type=range]").forEach(function (r) {
    r.addEventListener("input", function () { manual[r.dataset.target] = true; zoomTo(r.dataset.target, Number(r.value)); });
  });
  var theme = preferred(); apply(theme); render(theme);
  btn.addEventListener("click", function () {
    var next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    apply(next); try { localStorage.setItem(KEY, next); } catch (e) {}
    render(next);
  });
})();
</script></body></html>"""
(d / "architecture.html").write_text(page, encoding="utf-8")
print("wrote", (d / "architecture.html").stat().st_size, "bytes")
