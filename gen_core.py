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
  GW["gateway:transport 注册表<br/>经 api 注册 transports 处理器(#1774)"]
  subgraph FACADE["入口与门面(core 内部不得导入 api)"]
    API["api.py<br/>宿主门面:只做再导出(再导出的边不画)"]
    TESTING["testing.py<br/>测试辅助:snapshot_registry · restore_registry(core 不导入)"]
    INIT["llm_rosetta/__init__<br/>convert() · get_converter_for_provider()"]
    TOOL["tool_ops.py<br/>入口便捷层:为 toolregistry 直接转换工具定义/调用/结果;只被 llm_rosetta/__init__ 导入"]
    CLI["cli/ 包 · __main__<br/>分发器(__init__) · area 表与 commands 处理器(areas.py)(#1773、#1789)"]
  end
  subgraph CORE["转换库(没有 shim 时,这一组就是完整的转换库)"]
    subgraph TIERS["三层,自上而下依赖"]
      PIPE["pipeline/<br/>chat · embeddings · rerank · decisions · typed<br/>enforcement(感知 shim 的能力 pass,#1745) · tool_locations(#1744)"]
      CVT["converters/<br/>base(含 ConversionError) · llm/各格式 · embeddings · rerank · decisions(各含 lookup) · registry"]
      subgraph IRCORE["ir/(一个包,#1710)"]
        IROPS["ir/ 逻辑<br/>primitives · tools · passes · context"]
        IRMODEL["ir/model/<br/>数据类 · 类型守卫 · 校验 · IR_VERSION<br/>只依赖标准库"]
      end
    end
    subgraph SIDE["辅助"]
      DET["auto_detect.py"]
      CAP["capabilities.py<br/>0.13.0 的转发,警告,0.15.0 删除;无人导入"]
    end
  end
  subgraph OPT["可选外层(没有它们,转换库照样完整)"]
    SHIM["shims/<br/>provider_shim 注册表 · _loader · builtin/各 vendor · providers 转发层"]
    EXTS["extensions/<br/>_discovery · _load · _dirs · _install · _cli"]
  end
  subgraph FOUND["共享基础(契约、格式目录、词汇、参考形状)"]
    subgraph CONS["contracts/"]
      CDECL["声明层<br/>shim 数据类 · resolver(叶子) · auth · quota · routing · usage · transport · command · extension · cli_login · _schema(清单校验用的 schema 解释器)<br/>依赖 ir.model · formats · retired_names(都按纯数据)"]
      CXF["transform.py<br/>再导出 ir 原语<br/>contracts 里唯一导入 ir.primitives 的模块"]
    end
    FMT["formats/<br/>catalog · detect · spec · batch"]
    VOCAB["model_vocabulary · retired_names · conversion_warnings"]
    REF["types/<br/>提供方参考形状:openai · google · messages<br/>types.ir 是 0.13.0 的转发,0.15.0 删除"]
  end
  subgraph TOOLS["工具"]
    FID["fidelity/"]
    PROF["profiling/"]
    DEV["devtools/<br/>scaffold · ext_scaffold · 模板"]
  end
'''
current_edges = [
 ("INIT","PIPE","","solid",None), ("INIT","CVT","","solid",None),
 ("INIT","SHIM","接线(带保护的导入)","solid",None), ("INIT","DET","","solid",None),
 ("INIT","TOOL","","solid",None), ("TOOL","CVT","","solid",None), ("TOOL","IROPS","","solid",None), ("TOOL","VOCAB","retired_names","solid",None),
 ("CLI","CDECL","","solid",None), 
 ("GW","EXTS","经 api 注册 transports 处理器","lazy",GREEN),
 ("CLI","EXTS","注册 commands 处理器(加载器回放已加载的扩展)","solid",GREEN),
 ("PIPE","CVT","","solid",None), ("PIPE","IROPS","","solid",None), ("PIPE","CDECL","","solid",None),
 ("CVT","IROPS","","solid",None), ("CVT","IRMODEL","","solid",None), ("CVT","CDECL","","solid",None),
 ("IROPS","IRMODEL","","solid",None),
 ("SHIM","IROPS","","solid",None), ("SHIM","CDECL","注册 resolver","solid",None), ("SHIM","FMT","","solid",None),
 ("SHIM","EXTS","模块级:注册 shims 处理器;load_providers() 调 load_extensions()(#1735)","solid",GREEN),
 ("EXTS","CDECL","","solid",None), ("DEV","EXTS","ext new / check 的 ExtVerb 记录(#1752)","solid",None), ("CLI","DEV","把 ExtVerb 传给 ext_main","lazy",None),
 ("CDECL","IRMODEL","IREffort · IRMode · IR_VERSION","solid",None),
 ("CDECL","FMT","usage.semantics","solid",None),
 ("DET","CDECL","ProviderType 定义在 contracts.routing(#1738)","solid",None),
 ("CDECL","VOCAB","retired_names","solid",None),
 ("CXF","IROPS","","solid",RED),
 ("DET","FMT","","solid",None), ("FMT","IRMODEL","","solid",None),
 ("CAP","CVT","","solid",None), ("CAP","PIPE","转发 enforcement","solid",None), ("DEV","CDECL","","solid",None),
 ("REF","IRMODEL","转发(带警告)","solid",None),
 ("DET","CVT","按需加载(LAZY_ONLY_EDGES,#1787)","lazy",None), ("DET","PIPE","按需加载","lazy",None),
 ("DEV","CVT","按需加载","lazy",None), ("DEV","SHIM","按需加载","lazy",None),
 ("PIPE","FID","按需加载","lazy",None), ("SHIM","DET","按需加载:clear_converter_cache()","lazy",None),
]
open("core.mmd","w",encoding="utf-8").write(render(current_header, current_edges))

target_header = '''flowchart TB
  AUTHOR(["shim / 扩展作者"])
  GW["gateway:transport 注册表<br/>注册 transports 处理器"]
  subgraph FACADE["入口与门面(core 内部不得导入 api)"]
    API["api.py<br/>宿主门面:只做再导出(再导出的边不画)"]
    TESTING["testing.py<br/>测试辅助:snapshot_registry · restore_registry(core 不导入)"]
    INIT["llm_rosetta/__init__ 组合根"]
    CLI["cli.py · __main__<br/>分发器 · 注册 commands 处理器"]
  end
  subgraph CORE["转换库(没有 shim 时,这一组就是完整的转换库)"]
    subgraph TIERS["三层,自上而下依赖"]
      PIPE["pipeline/<br/>chat · embeddings · rerank · decisions<br/>enforcement(感知 shim 的能力 pass) · tool_locations"]
      CVT["converters/"]
      subgraph IRCORE["ir/(一个包,#1710 已落地)"]
        IROPS["ir/ 逻辑<br/>primitives · tools · passes · context"]
        IRMODEL["ir/model/<br/>数据类 · 类型守卫 · 校验 · IR_VERSION(原 types/ir)"]
      end
    end
  end
  subgraph OPT["可选外层(没有它们,转换库照样完整)"]
    EXTS["extensions/ 只管包<br/>清单校验 · 发现 · 安装/卸载/更新 · 索引 · 兼容性 · 状态<br/>种类处理器注册点;不认识 shim / transport / commands"]
    SHIM["shims/<br/>注册表 · 加载器 · builtin<br/>注册 shims 处理器"]
  end
  subgraph FOUND["共享基础"]
    subgraph CONS["contracts/"]
      CDECL["声明层(叶子)<br/>shim · resolver(#1708 已落地) · auth · quota · routing · usage · transport · command · extension · 清单<br/>只依赖 types / 标准库"]
      CXF["transform.py<br/>便利再导出 ir 原语<br/>唯一允许依赖 ir 的模块"]
    end
    FMT["formats/ · model_vocabulary · retired_names<br/>types/ 只剩提供方参考形状,不属于核心"]
  end
'''
target_edges = [
("INIT","PIPE","","solid",None), ("INIT","SHIM","","solid",None),
 ("SHIM","EXTS","注册 shims 处理器;load_providers() 调 load_extensions()","solid",GREEN),
 ("CLI","EXTS","注册 commands 处理器","solid",GREEN),
 ("GW","EXTS","经 api 注册 transports 处理器","lazy",GREEN),
 ("EXTS","CDECL","","solid",None),
 ("SHIM","CDECL","","solid",None), ("SHIM","IROPS","","solid",None),
 ("SHIM","FMT","","solid",None),
 ("PIPE","CVT","","solid",None), ("PIPE","IROPS","","solid",None),
 ("CVT","IROPS","","solid",None), ("CVT","IRMODEL","","solid",None), ("CVT","CDECL","","solid",None),
 ("IROPS","IRMODEL","","solid",None), ("FMT","IRMODEL","","solid",None),
 ("CXF","IROPS","","solid",None), ("CXF","CDECL","","solid",None),
 ("CDECL","FMT","","solid",None),
 ("AUTHOR","CDECL","只用 contracts","solid",None), ("AUTHOR","CXF","","solid",None),
]
open("core_target.mmd","w",encoding="utf-8").write(render(target_header, target_edges))
print("edges:", len(current_edges), len(target_edges))
