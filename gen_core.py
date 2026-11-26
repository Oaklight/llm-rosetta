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
      subgraph IRCORE["ir/(一个包,#1710)"]
        IRMODEL["ir/model/<br/>数据类 · 类型守卫 · 校验 · IR_VERSION<br/>只依赖标准库"]
        IROPS["ir/ 逻辑<br/>primitives · tools · passes · context"]
      end
      PIPE["pipeline/<br/>chat · embeddings · rerank · decisions · typed"]
      CVT["converters/<br/>base(含 ConversionError) · llm/各格式 · embeddings · rerank · decisions(各含 lookup) · registry"]
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
  subgraph FOUND["共享基础(契约、格式目录、词汇、参考形状)"]
    CON["contracts/<br/>外部各方实现或接收的契约<br/>shim 数据类 · resolver(叶子) · transform · auth · quota · routing · usage · transport · command · extension · cli_login · shim_manifest"]
    FMT["formats/<br/>catalog · detect · spec · batch"]
    VOCAB["model_vocabulary · retired_names · conversion_warnings"]
    REF["types/<br/>提供方参考形状:openai · google · messages<br/>types.ir 是 0.13.0 的转发,0.15.0 删除"]
  end
'''
current_edges = [
 ("INIT","PIPE","","solid",None), ("INIT","CVT","","solid",None),
 ("INIT","SHIM","接线(带保护的导入)","solid",None), ("INIT","DET","","solid",None),
 ("CLI","CON","","solid",None), ("CLI","EXTS","懒加载","lazy",None),
 ("PIPE","CVT","","solid",None), ("PIPE","IROPS","","solid",None), ("PIPE","CON","","solid",None),
 ("CVT","IROPS","","solid",None), ("CVT","IRMODEL","","solid",None), ("CVT","CON","","solid",None),
 ("IROPS","IRMODEL","","solid",None), ("IROPS","CON","passes 用 shim 数据类","solid",None),
 ("SHIM","IROPS","","solid",None), ("SHIM","CON","注册 resolver","solid",None), ("SHIM","FMT","","solid",None),
 ("SHIM","EXTS","导入时 load_extensions","lazy",RED),
 ("EXTS","SHIM","注册 shim","lazy",RED),
 ("EXTS","CON","","solid",None), ("EXTS","DEV","ext new / check","lazy",None),
 ("CON","IRMODEL","数据类型","solid",None),
 ("CON","FMT","usage.semantics","solid",None),
 ("CON","DET","routing.py","solid",None),
 ("CON","IROPS","transform 再导出 ir 原语","solid",RED),
 ("DET","FMT","","solid",None), ("FMT","IRMODEL","","solid",None),
 ("CAP","CVT","","solid",None), ("DEV","CON","","solid",None),
 ("REF","IRMODEL","转发(带警告)","solid",None),
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
      PIPE["pipeline/<br/>chat · embeddings · rerank · decisions"]
      CVT["converters/"]
      subgraph IRCORE["ir/(一个包,#1710 已落地)"]
        IRMODEL["ir/model/<br/>数据类 · 类型守卫 · 校验 · IR_VERSION(原 types/ir)"]
        IROPS["ir/ 逻辑<br/>primitives · tools · passes · context"]
      end
    end
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
 ("INIT","EXTS","load_extensions()","solid",GREEN),
 ("SHIM","EXTS","注册 shims 处理器","solid",GREEN),
 ("CLI","EXTS","注册 commands 处理器","solid",GREEN),
 ("GW","EXTS","经 api 注册 transports 处理器","lazy",GREEN),
 ("EXTS","CDECL","","solid",None),
 ("SHIM","CDECL","","solid",None), ("SHIM","IROPS","","solid",None),
 ("SHIM","FMT","","solid",None),
 ("PIPE","CVT","","solid",None), ("PIPE","IROPS","","solid",None),
 ("CVT","IROPS","","solid",None), ("CVT","IRMODEL","","solid",None), ("CVT","CDECL","","solid",None),
 ("IROPS","CDECL","","solid",None), ("IROPS","IRMODEL","","solid",None), ("FMT","IRMODEL","","solid",None),
 ("CXF","IROPS","","solid",None), ("CXF","CDECL","","solid",None),
 ("CDECL","FMT","","solid",None),
 ("AUTHOR","CDECL","只用 contracts","solid",None), ("AUTHOR","CXF","","solid",None),
]
open("core_target.mmd","w",encoding="utf-8").write(render(target_header, target_edges))
print("edges:", len(current_edges), len(target_edges))
