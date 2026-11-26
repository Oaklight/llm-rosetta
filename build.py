import html, pathlib
d = pathlib.Path(".")
def src(n): return html.escape((d / f"{n}.mmd").read_text(encoding="utf-8"), quote=False)
sections = [
 ("core", "1. core 现状:组件与依赖",
  "实线是模块顶层导入,虚线是函数内的懒加载。红色是有问题的依赖:shims 与 extensions 互相依赖(导入时加载 vs 注册 shim)、ir 与 shims 互相依赖、contracts 经 transform.py 依赖 ir。contracts 在共享基础层,被转换、shim、扩展、cli、devtools 共用;api 是宿主门面,core 内部不导入它(有测试保证)。未画的:shims/providers 转发层、已弃用的 gateway/ 与 observability/ 转发包。画的是 master(含 PR #1705)的树;extensions/_index、_commands、_probe 在草稿 PR #1707。"),
 ("core_target", "2. core 建议目标(未实现,待决定)",
  "绿色是建议新增的接线。extensions 只管包(清单、发现、安装、索引、兼容性、状态),只依赖 contracts,不认识 shim、transport、commands;shims、cli、gateway 各自向它注册种类处理器;由 __init__ 这个组合根在导入完成后调用 load_extensions(),shims 不再在导入时隐式调用它。contracts 拆成声明层(只依赖 types 和标准库)和 transform.py(唯一允许依赖 ir 的便利再导出),并用导入边界测试固定。"),
 ("gateway", "3. gateway 后端的组件与关系",
  "分层来自 tests/gateway/test_layers.py,只能向下导入。每个包都会用 core,图里只画了几条代表性的到 CORE 的线。admin 除了 admin_api 之外还有一批旧 allowlist 导入(config、keystore、maintenance、middleware、observability、ops、quota、routing、shims、transport、updates),名单只允许缩小,图里没画。shims 里的 manifest_sources 是要被 index.json 取代的部分。"),
 ("admin", "4. admin 页面与组件的关系",
  "已核对:页面→脚本、脚本→API 分组(来自各 JS 里的 /admin/api 调用)、9 个标签页及子区域(admin.html)。未核对:右半段“API 分组→后端组件”按名字推断;settings 页的 JS 接线只确认了文件;components/ 是否也被 models 等其他页使用。GET /admin/api/extensions 后端已有,目前没有页面调用它。"),
]
blocks = ""
for key, title, note in sections:
    blocks += f'''
<section id="{key}">
  <h2>{html.escape(title)}</h2>
  <p class="note">{html.escape(note)}</p>
  <div class="bar"><label>缩放 <input type="range" min="40" max="200" value="100" step="5" data-target="{key}"> <span id="z-{key}">100%</span></label></div>
  <div class="scroll"><div class="stage" id="stage-{key}"><pre class="mermaid" id="m-{key}">{src(key)}</pre></div></div>
</section>'''
page = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>llm-rosetta 架构图</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { color-scheme: light dark; }
body { font: 15px/1.5 system-ui, "Noto Sans CJK SC", sans-serif; margin: 0; padding: 24px 32px 80px; background: #fff; color: #1b1f24; }
@media (prefers-color-scheme: dark) { body { background:#14171a; color:#e6e8ea; } }
h1 { margin: 0 0 4px; } h2 { margin: 36px 0 6px; }
nav a { margin-right: 14px; }
.note { max-width: 1000px; color: #5b6570; margin: 4px 0 10px; }
.legend span { display:inline-block; margin-right:18px; }
.legend i { display:inline-block; width:28px; height:0; border-top:3px solid; vertical-align:middle; margin-right:6px; }
.bar { margin: 6px 0; }
.scroll { overflow: auto; border: 1px solid #c8ced4; border-radius: 6px; background: #fff; padding: 12px; max-height: 88vh; }
.stage { width: max-content; }
pre.mermaid { margin: 0; background: transparent; }
pre.error { color: #b00; white-space: pre-wrap; }
</style></head>
<body>
<h1>llm-rosetta 架构图</h1>
<p class="note">core 现状、core 建议目标、gateway 后端、gateway admin。由 mermaid 在浏览器里渲染,可滚动、可缩放。</p>
<p class="legend"><span><i style="border-color:#333"></i>模块顶层导入</span><span><i style="border-color:#333;border-top-style:dashed"></i>函数内懒加载</span><span><i style="border-color:#d33"></i>有问题的依赖(循环或反向)</span><span><i style="border-color:#1a8f3c"></i>建议新增的接线</span></p>
<nav><a href="#core">core 现状</a><a href="#core_target">core 建议目标</a><a href="#gateway">gateway 后端</a><a href="#admin">admin 页面</a></nav>
''' + blocks + '''
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<script>
mermaid.initialize({ startOnLoad: false, securityLevel: "loose", theme: "default",
  flowchart: { useMaxWidth: false, htmlLabels: true, curve: "basis", nodeSpacing: 40, rankSpacing: 60 } });
(async () => {
  for (const pre of document.querySelectorAll("pre.mermaid")) {
    const id = pre.id.replace("m-", "");
    try {
      const { svg } = await mermaid.render("svg-" + id, pre.textContent);
      pre.outerHTML = '<div class="rendered" data-ok="1">' + svg + '</div>';
      const stage = document.getElementById("stage-" + id), box = stage.parentElement;
      const w = stage.querySelector("svg").getBoundingClientRect().width;
      const fit = Math.max(40, Math.min(100, Math.floor(((box.clientWidth - 28) / w) * 20) * 5));
      const slider = document.querySelector('input[data-target="' + id + '"]');
      slider.value = fit; slider.dispatchEvent(new Event("input"));
    } catch (e) {
      pre.outerHTML = '<pre class="error" data-ok="0">' + String(e) + '</pre>';
    }
  }
  document.body.dataset.done = "1";
})();
document.querySelectorAll("input[type=range]").forEach(r => r.addEventListener("input", () => {
  const k = r.dataset.target, v = r.value / 100;
  document.getElementById("stage-" + k).style.zoom = v;
  document.getElementById("z-" + k).textContent = r.value + "%";
}));
</script></body></html>'''
(d / "architecture.html").write_text(page, encoding="utf-8")
print("wrote", (d / "architecture.html").stat().st_size, "bytes")
