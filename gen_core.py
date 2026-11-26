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
  subgraph FACADE["入口与门面(core 内部不得导入 api)"]
    API["api.py<br/>宿主门面:只做再导出(再导出的边不画)"]
    INIT["llm_rosetta/__init__<br/>convert() · get_converter_for_provider()"]
    CLI["cli.py · __main__<br/>llm-rosetta 分发器"]
  end
  subgraph IMPL["实现"]
    subgraph CONV["转换"]
      PIPE["pipeline/<br/>chat · typed"]
      CVT["converters/<br/>base · llm/各格式 · embeddings · rerank · decisions · registry"]
      IR["ir/<br/>primitives · tools · passes · context"]
      DET["auto_detect.py"]
      CAP["capabilities.py"]
      TOOL["tool_ops.py"]
    end
    subgraph SHIMEXT["shim 与扩展"]
      SHIM["shims/<br/>provider_shim 注册表 · _loader · builtin/各 vendor · providers 转发层"]
      EXTS["extensions/<br/>_discovery · _load · _dirs · _install · _cli"]
    end
    subgraph TOOLS["工具"]
      FID["fidelity/"]
      PROF["profiling/"]
      DEV["devtools/<br/>scaffold · ext_scaffold · 模板"]
    end
  end
  subgraph FOUND["共享基础(被上面各层使用)"]
    CON["contracts/<br/>外部各方实现或接收的契约<br/>shim 数据类 · transform · auth · quota · routing · usage · transport · command · extension · cli_login · shim_manifest"]
    FMT["formats/<br/>catalog · detect · spec · batch"]
    TYP["types/<br/>IR 数据类 · 提供方形状(仅参考)"]
    VOCAB["model_vocabulary · retired_names · conversion_warnings"]
  end
'''
current_edges = [
("INIT","PIPE","","solid",None), ("INIT","CVT","","solid",None),
 ("INIT","SHIM","","solid",None), ("INIT","DET","","solid",None),
 ("CLI","CON","","solid",None), ("CLI","EXTS","懒加载","lazy",None),
 ("PIPE","CVT","","solid",None), ("PIPE","IR","","solid",None),
 ("PIPE","SHIM","","solid",None), ("PIPE","CON","","solid",None),
 ("CVT","IR","","solid",None), ("CVT","CON","","solid",None),
 ("IR","CON","","solid",None),
 ("IR","SHIM","ir 的 pass 用 shim 契约","solid",RED),
 ("SHIM","IR","","solid",RED),
 ("SHIM","CON","","solid",None), ("SHIM","FMT","","solid",None),
 ("SHIM","EXTS","导入时 load_extensions","lazy",RED),
 ("EXTS","SHIM","注册 shim","lazy",RED),
 ("EXTS","CON","","solid",None), ("EXTS","DEV","ext new / check","lazy",None),
 ("CON","TYP","数据类型","solid",None),
 ("CON","FMT","usage.semantics","solid",None),
 ("CON","DET","routing.py","solid",None),
 ("CON","IR","transform 再导出 ir 原语","solid",RED),
 ("DET","FMT","","solid",None), ("FMT","TYP","","solid",None),
 ("CAP","CVT","","solid",None), ("DEV","CON","","solid",None),
]
open("core.mmd","w",encoding="utf-8").write(render(current_header, current_edges))

target_header = '''flowchart TB
  AUTHOR(["shim / 扩展作者"])
  GW["gateway:transport 注册表<br/>注册 transports 处理器"]
  subgraph FACADE["入口与门面(core 内部不得导入 api)"]
    API["api.py<br/>宿主门面:只做再导出(再导出的边不画)"]
    INIT["llm_rosetta/__init__ 组合根<br/>导入完成后调用 load_extensions()"]
    CLI["cli.py · __main__<br/>分发器 · 注册 commands 处理器"]
  end
  subgraph IMPL["实现"]
    subgraph CONV["转换"]
      PIPE["pipeline/"]
      CVT["converters/"]
      IR["ir/"]
    end
    EXTS["extensions/ 只管包<br/>清单校验 · 发现 · 安装/卸载/更新 · 索引 · 兼容性 · 状态<br/>种类处理器注册点;不认识 shim / transport / commands"]
    SHIM["shims/<br/>注册表 · 加载器 · builtin<br/>注册 shims 处理器"]
  end
  subgraph FOUND["共享基础"]
    subgraph CONS["contracts/"]
      CDECL["声明层(叶子)<br/>shim · auth · quota · routing · usage · transport · command · extension · 清单<br/>只依赖 types / 标准库"]
      CXF["transform.py<br/>便利再导出 ir 原语<br/>唯一允许依赖 ir 的模块"]
    end
    FMT["formats/ · types/ · model_vocabulary · retired_names"]
  end
'''
target_edges = [
("INIT","PIPE","","solid",None), ("INIT","SHIM","","solid",None),
 ("INIT","EXTS","load_extensions()","solid",GREEN),
 ("SHIM","EXTS","注册 shims 处理器","solid",GREEN),
 ("CLI","EXTS","注册 commands 处理器","solid",GREEN),
 ("GW","EXTS","经 api 注册 transports 处理器","lazy",GREEN),
 ("EXTS","CDECL","","solid",None),
 ("SHIM","CDECL","","solid",None), ("SHIM","IR","","solid",None),
 ("SHIM","FMT","","solid",None),
 ("PIPE","CVT","","solid",None), ("PIPE","IR","","solid",None),
 ("CVT","IR","","solid",None), ("CVT","CDECL","","solid",None),
 ("IR","CDECL","","solid",None),
 ("CXF","IR","","solid",None), ("CXF","CDECL","","solid",None),
 ("CDECL","FMT","","solid",None),
 ("AUTHOR","CDECL","只用 contracts","solid",None), ("AUTHOR","CXF","","solid",None),
]
open("core_target.mmd","w",encoding="utf-8").write(render(target_header, target_edges))
print("edges:", len(current_edges), len(target_edges))
