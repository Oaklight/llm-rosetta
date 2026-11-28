"""Ideal versus actual: the gateway slice that the #1830 review measured.

Writes gap_actual.mmd (master, problems in red) and gap_ideal.mmd (after #1830).
The numbers come from the review of origin/master 215c832bbe; they are not recomputed here.
"""
import pathlib

OUT = pathlib.Path(__file__).parent


def emit(name, nodes, edges):
    lines = ["flowchart LR"]
    lines += [f"  {n}" for n in nodes]
    red = []
    for i, (a, b, label, bad) in enumerate(edges):
        arrow = "-.->" if label.startswith("~") else "-->"
        text = label.lstrip("~")
        lines.append(f'  {a} {arrow}|"{text}"| {b}' if text else f"  {a} {arrow} {b}")
        if bad:
            red.append(i)
    if red:
        lines.append(f"  linkStyle {','.join(map(str, red))} stroke:#c0392b,stroke-width:2px")
    (OUT / f"{name}.mmd").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(name, len(nodes), "nodes", len(edges), "edges", len(red), "red")


ACTUAL_NODES = [
    'ADMIN["admin/<br/>routes · js"]',
    'ADMINAPI["admin_api.py<br/>AdminContext(5 个成员)"]',
    'APP["app/<br/>factory · routes · admin_context"]',
    'STATE["app/_state._config<br/>+ app.gateway_config(两个落点)"]',
    'RELOAD["admin/routes/_shared.py<br/>_reload_gateway_config(A1)"]',
    'LOGIN["admin/routes/auth.py<br/>登录限流:总信任 X-Forwarded-For(A3)"]',
    'MWCTX["middleware/request_context<br/>extract_client_ip(遵守 trust_proxy)"]',
    'PROXY["proxy/<br/>engine · streaming"]',
    'PIPES["pipelines/typed.py"]',
    'RELAY["app/routes_compact + routes_count_tokens<br/>(各 140-150 行,复杂度 22-23)"]',
    'CONFIG["config/providers.py<br/>build_provider_info 294 行,复杂度 49<br/>+ 约 20 个 ACP lane 辅助(A8)"]',
    'PINFO["transport/provider_info.py<br/>ProviderInfo:58 个属性"]',
    'TRANSPORT["transport/acp.py(2535 行)"]',
    'ACPQ["transport/acp_quota.py<br/>进程级 _READINGS / _LOCKS(A9)"]',
    'QUOTA["quota/service.py"]',
    'MAINT["maintenance/startup.py(716 行)<br/>ProviderNotReady 已移走(A5,#1832 已合并)"]',
    'OPS["ops/<br/>OpsBase 子类"]',
    'OPSLOG["observability/ops_log"]',
    'EXTCORE["core extensions/_index.select_entry<br/>最新兼容版,含预发布"]',
    'EXTGW["gateway shims/manifest_sources<br/>select_release:稳定优先"]',
    'EXTDIR[("/config/extensions")]',
]
ACTUAL_EDGES = [
    ("ADMIN", "ADMINAPI", "", False),
    ("APP", "ADMINAPI", "适配器", False),
    ("ADMIN", "APP", "读 request.app.* 28 种,约 113 处(A1)", True),
    ("ADMIN", "CONFIG", "67 对 allowlist 导入(A1)", True),
    ("ADMIN", "QUOTA", "旧 allowlist 导入(A1)", True),
    ("ADMIN", "RELOAD", "", False),
    ("RELOAD", "STATE", "写 5 个 app 属性(A1)", True),
    ("APP", "STATE", "", False),
    ("ADMIN", "LOGIN", "", False),
    ("LOGIN", "MWCTX", "不调用(A3)", True),
    ("PROXY", "QUOTA", "自己记账(A4)", True),
    ("PIPES", "QUOTA", "自己记账(A4)", True),
    ("RELAY", "QUOTA", "自己记账(A4)", True),
    ("APP", "CONFIG", "", False),
    ("CONFIG", "PINFO", "", False),
    ("CONFIG", "TRANSPORT", "ACP lane 解析在 config 里(A8)", True),
    ("TRANSPORT", "ACPQ", "", False),
    ("ACPQ", "QUOTA", "状态放错地方(A9)", True),
    ("PROXY", "OPS", "", False),
    ("OPS", "OPSLOG", "", False),
    ("ADMIN", "OPSLOG", "绕过 ops 直接写(A7)", True),
    ("APP", "OPSLOG", "绕过 ops 直接写(A7)", True),
    ("EXTCORE", "EXTDIR", "ext install / update", False),
    ("EXTGW", "EXTDIR", "重建 shim 来源(A2:两套策略)", True),
]

IDEAL_NODES = [
    'ADMIN["admin/<br/>routes · js"]',
    'ADMINAPI["admin_api.py<br/>AdminContext:reload_config 等"]',
    'APP["app/<br/>factory · routes · admin_context<br/>热重载与唯一的已生效配置"]',
    'LOGIN["admin/routes/auth.py<br/>限流状态在 app.auth_state,有上限"]',
    'MWCTX["middleware/request_context<br/>extract_client_ip(遵守 trust_proxy)"]',
    'PROXY["proxy/<br/>engine · streaming<br/>attempt_upstream(...)(A4)"]',
    'PIPES["pipelines/typed.py"]',
    'RELAY["app/routes_compact + routes_count_tokens"]',
    'CONFIG["config/providers.py<br/>只做解析与装配"]',
    'ACPPROF["transport/acp_profile.py<br/>ACP lane 解析(A8)"]',
    'PINFO["transport/provider_info.py<br/>ProviderInfo:auth · ACP · quota 子记录"]',
    'TRANSPORT["transport/acp.py"]',
    'SINK["transport 的 quota 接收器<br/>(启动时由 quota 注册,A9)"]',
    'QUOTA["quota/<br/>读数与锁在这里"]',
    'PSTATE["transport/provider_readiness.py(叶子)<br/>ProviderInitState · ProviderNotReady(A5,#1832 已合并)"]',
    'MAINT["maintenance/startup.py"]',
    'OPS["ops/<br/>唯一的审计写入者(A7)"]',
    'OPSLOG["observability/ops_log"]',
    'EXTCORE["core 的唯一安装策略<br/>select_entry + 预发布开关(A2)"]',
    'EXTGW["gateway shims/<br/>调用核心策略"]',
    'EXTDIR[("/config/extensions")]',
]
IDEAL_EDGES = [
    ("ADMIN", "ADMINAPI", "只经这一条", False),
    ("APP", "ADMINAPI", "适配器(必须存在,无回退;A6,#1833 已合并)", False),
    ("ADMIN", "LOGIN", "", False),
    ("LOGIN", "MWCTX", "", False),
    ("PROXY", "QUOTA", "", False),
    ("PIPES", "PROXY", "attempt_upstream", False),
    ("RELAY", "PROXY", "attempt_upstream", False),
    ("APP", "CONFIG", "", False),
    ("CONFIG", "ACPPROF", "", False),
    ("CONFIG", "PINFO", "", False),
    ("ACPPROF", "TRANSPORT", "", False),
    ("TRANSPORT", "SINK", "", False),
    ("QUOTA", "SINK", "~注册", False),
    ("CONFIG", "PSTATE", "", False),
    ("PIPES", "PSTATE", "", False),
    ("MAINT", "PSTATE", "", False),
    ("PROXY", "OPS", "", False),
    ("APP", "OPS", "启动 · 关闭 · 重载", False),
    ("ADMINAPI", "OPS", "经 AdminContext", False),
    ("OPS", "OPSLOG", "", False),
    ("EXTGW", "EXTCORE", "", False),
    ("EXTCORE", "EXTDIR", "", False),
]

emit("gap_actual", ACTUAL_NODES, ACTUAL_EDGES)
emit("gap_ideal", IDEAL_NODES, IDEAL_EDGES)
