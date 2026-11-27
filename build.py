import html, pathlib, re, subprocess
d = pathlib.Path(".")
def src(n): return html.escape((d / f"{n}.mmd").read_text(encoding="utf-8"), quote=False)
sections = [
 ("core", "1. core 现状:组件与依赖",
  "实线是模块顶层导入,虚线是函数内的懒加载。已解决(#1708):ir、pipeline、formats、auto_detect 不再导入 shims,shim 名字经 contracts/resolver.py 读取(shims 注册,llm_rosetta/__init__ 接线),contracts 也不再导入 shims 和 extensions。已解决(#1709):四个编排器(chat、embeddings、rerank、decisions)都在 pipeline/,converters 不再导入 pipeline(任何深度),ConversionError 只定义在 converters/base/errors.py(各公开名字指向同一个类),pipeline 从 ir.context 取 ConversionContext;边界测试在 tests/test_core_layering.py。已解决(#1710):IR 合成一个包 llm_rosetta.ir:ir/model(原 types/ir,数据,只依赖标准库)在底层,ir/primitives、ir/tools、ir/passes 在它之上;types/ 只剩提供方参考形状,types.ir 是 0.13.0 的转发(警告,0.15.0 删除),只包含 0.13.0 出现过的 72 个名字和 15 个子模块。已解决(#1735):shims 与 extensions 不再互相依赖,shims 向 extensions 注册 shims 处理器(绿线)。仍是红色的依赖只剩一条:contracts/transform.py 导入 ir.primitives(contracts 里唯一的这类模块,边表由 tests/test_core_layering.py 固定,目标图接受它作为例外)。“转换库”组里是三层:pipeline 在最上,converters 在中间,IR 在最下(ir/model 在 ir/ 逻辑之下),自上而下依赖,没有 shim 时这一组就是完整的转换库;shims 和 extensions 是它外面的可选层;contracts 在共享基础层,被转换、shim、扩展、cli、devtools 共用;api 是宿主门面,core 内部不导入它(有测试保证)。api 只是把下面各包再导出一遍,0.14.0 dev 期间还会大改,这些再导出的边不画,免得遮住真正的依赖。tool_ops(入口便捷层)只画到 converters 的边,它到 ir 和 retired_names 的边(类型与错误消息)在 CORE_EDGES 里有,图里不画。未画的:shims/providers 转发层、已弃用的 gateway/ 与 observability/ 转发包。画的是 master 的树。"),
 ("gateway", "2. gateway 后端的组件与关系",
  "分层来自 tests/gateway/test_layers.py,只能向下导入。每个包都会用 core,图里只画了几条代表性的到 CORE 的线。admin 除了 admin_api 之外还有一批旧 allowlist 导入(config、keystore、maintenance、middleware、observability、ops、quota、routing、shims、transport、updates),名单只允许缩小,图里没画。shims 里的 manifest_sources 是要被 index.json 取代的部分。"),
 ("admin", "3. admin 页面与组件的关系",
  "已核对:页面→脚本、脚本→API 分组(来自各 JS 里的 /admin/api 调用)、9 个标签页及子区域(admin.html)。未核对:右半段“API 分组→后端组件”按名字推断;settings 页的 JS 接线只确认了文件;components/ 是否也被 models 等其他页使用。GET /admin/api/extensions 后端已有,目前没有页面调用它。"),
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
    return f'<p class="sub">{lede}</p>' + (f'<details class="notes"><summary>说明与已解决项({len(parts) - 1} 条)</summary><ul>{items}</ul></details>' if items else "")

blocks = ""
toc = ""
for n, (key, title, note) in enumerate(sections, 1):
    label = title.split(". ", 1)[1]
    toc += f'<a href="#{key}">{n:02d} {html.escape(label)}</a>'
    blocks += f"""
<section id="{key}">
  <h2><span class="num">{n:02d}</span>{html.escape(label)}</h2>
  {note_html(note)}
  <div class="panel">
    <span class="tag now">现状 · master @ {master}</span>
    <div class="zoom"><label>缩放 <input type="range" min="40" max="200" value="100" step="5" data-target="{key}"> <span id="z-{key}">100%</span></label></div>
    <div class="scroll"><div class="stage" id="stage-{key}"><pre class="mermaid" id="m-{key}">{src(key)}</pre></div></div>
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
  <h1>现状:core、gateway 与 admin 的组件和依赖</h1>
  <p class="lede">三张图都画 master 的实际结构,不画目标态。红色表示还没解决的反向依赖,绿色表示 core 与扩展之间的接线,虚线是函数内的懒加载。合并任一分层 PR 后,本页按 master 更新。</p>
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
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<script>
(function () {
  var KEY = "llm-rosetta-arch-theme", btn = document.getElementById("themeBtn");
  var panes = Array.prototype.map.call(document.querySelectorAll("pre.mermaid"), function (pre) {
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
