RED = "stroke:#d33,stroke-width:2.5px,color:#d33"
GREEN = "stroke:#1a8f3c,stroke-width:2.5px,color:#1a8f3c"

def render(header, edges):
    lines = [header, ""]
    styles = []
    for i, (a, b, label, kind, color) in enumerate(edges):
        arrow = "-.->" if kind == "lazy" else "-->"
        lines.append(f"  {a} {arrow}|\"{label}\"| {b}" if label else f"  {a} {arrow} {b}")
        if color:
            styles.append(f"  linkStyle {i} {color}")
    return "\n".join(lines + [""] + styles) + "\n"

current_header = '''flowchart TB
  GW["gateway:transport 注册表<br/>经 api 注册 transports 处理器"]
  subgraph CORE["转换库(没有 shim 时,这一组就是完整的转换库)"]
    subgraph TIERS["三层,自上而下依赖"]
      PIPE["pipeline/<br/>chat · embeddings · rerank · decisions · typed<br/>enforcement(感知 shim 的能力 pass) · tool_locations"]
      CVT["converters/<br/>base(含 ConversionError) · llm/各格式 · embeddings · rerank · decisions(各含 lookup) · registry"]
      subgraph IRCORE["ir/(一个包)"]
        IRMODEL["ir/model/<br/>数据类 · 类型守卫 · 校验 · IR_VERSION<br/>只依赖标准库"]
        IROPS["ir/ 逻辑<br/>primitives · tools · passes · context"]
      end
    end
  end
  subgraph FACADE["入口与门面(core 内部不得导入 api)"]
    TOOL["tool_ops.py<br/>入口便捷层:为 toolregistry 直接转换工具定义/调用/结果;只被 llm_rosetta/__init__ 导入"]
    CLI["cli/ 包 · __main__<br/>分发器(__init__) · area 表与 commands 处理器(areas.py)"]
    CAP["capabilities.py<br/>0.13.0 的转发,警告,0.15.0 删除;无人导入"]
    TESTING["testing.py<br/>测试辅助:snapshot_registry · restore_registry(core 不导入)"]
    INIT["llm_rosetta/__init__<br/>convert() · get_converter_for_provider()"]
    DET["auto_detect.py<br/>弃用的转发层,无逻辑,0.15.0 删除;无人导入"]
    API["api.py<br/>宿主门面:只做再导出(再导出的边不画)"]
  end
  subgraph TOOLS["运行时附件与作者工具"]
    PROF["profiling/"]
    DEV["authoring/<br/>扩展作者工具(随包发布):ext new · ext check · ext migrate<br/>树契约 · kinds(shims/transports/commands 的规则)· 样本模板树"]
    FID["fidelity/"]
  end
  subgraph OPT["可选外层(没有它们,转换库照样完整)"]
    SHIM["shims/<br/>provider_shim 注册表 · _loader · builtin/各 vendor · providers 转发层"]
    EXTS["extensions/<br/>_discovery · _load · _dirs · _install · _index · _probe · _tools · _node · _manage · _cli<br/>ext install-cli:外部程序按精确版本装进 /config/tools;node 24.22.3 也装进 /config/tools/node(musl 用 unofficial-builds);_manage:移除计划(连程序一起删,共享 node 最后一个用户走才删)与 shim 的 needs"]
  end
  subgraph FOUND["共享基础(契约、格式目录、词汇、参考形状)"]
    REF["types/<br/>提供方参考形状:openai · google · messages<br/>types.ir 是 0.13.0 的转发,0.15.0 删除"]
    FMT["formats/<br/>catalog · detect · spec · batch"]
    subgraph CONS["contracts/"]
      CXF["transform.py<br/>再导出 ir 原语<br/>contracts 里唯一导入 ir.primitives 的模块"]
      CDECL["声明层<br/>shim 数据类 · resolver(叶子) · auth · quota · routing · usage · transport · command · extension · cli_login · _schema(清单校验用的 schema 解释器)<br/>依赖 ir.model · formats · retired_names(都按纯数据)"]
    end
    VOCAB["model_vocabulary · retired_names · conversion_warnings"]
  end
'''
current_edges = [
 ("INIT", "PIPE", "", "solid", None),
 ("SHIM", "CDECL", "注册 resolver", "solid", None),
 ("CLI", "DEV", "把 ExtVerb 传给 ext_main", "lazy", None),
 ("CVT", "IRMODEL", "", "solid", None),
 ("SHIM", "EXTS", "模块级:注册 shims 处理器;load_providers() 调 load_extensions()", "solid", GREEN),
 ("INIT", "SHIM", "接线(带保护的导入)", "solid", None),
 ("PIPE", "FID", "按需加载", "lazy", None),
 ("DEV", "SHIM", "按需加载", "lazy", None),
 ("EXTS", "CDECL", "", "solid", None),
 ("DEV", "CDECL", "", "solid", None),
 ("CVT", "CDECL", "", "solid", None),
 ("IROPS", "IRMODEL", "", "solid", None),
 ("CXF", "IROPS", "", "solid", RED),
 ("DEV", "EXTS", "ext new / check 的 ExtVerb 记录", "solid", None),
 ("PIPE", "CVT", "", "solid", None),
 ("CAP", "CVT", "", "solid", None),
 ("CDECL", "IRMODEL", "IREffort · IRMode · IR_VERSION", "solid", None),
 ("CDECL", "FMT", "usage.semantics", "solid", None),
 ("PIPE", "IROPS", "", "solid", None),
 ("CVT", "IROPS", "", "solid", None),
 ("PIPE", "CDECL", "", "solid", None),
 ("REF", "IRMODEL", "转发(带警告)", "solid", None),
 ("SHIM", "IROPS", "", "solid", None),
 ("CAP", "PIPE", "转发 enforcement", "solid", None),
 ("CVT", "FMT", "detect_provider 读格式目录", "solid", None),
 ("CLI", "EXTS", "注册 commands 处理器(加载器回放已加载的扩展)", "solid", GREEN),
 ("SHIM", "CVT", "清转换器缓存", "solid", None),
 ("DET", "CVT", "转发 get_converter_for_provider · detect_provider", "solid", None),
 ("INIT", "CVT", "", "solid", None),
 ("GW", "EXTS", "经 api 注册 transports 处理器", "lazy", GREEN),
 ("DET", "PIPE", "转发 convert · convert_response(pipeline/facade)", "solid", None),
 ("TOOL", "CVT", "", "solid", None),
 ("CLI", "CDECL", "", "solid", None),
 ("CDECL", "VOCAB", "retired_names", "solid", None),
 ("SHIM", "FMT", "", "solid", None),
 ("INIT", "TOOL", "", "solid", None),
 ("FMT", "IRMODEL", "", "solid", None),
]
open("core.mmd","w",encoding="utf-8").write(render(current_header, current_edges))

print("edges:", len(current_edges))
