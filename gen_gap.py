"""Pages 4 and 5: the whole gateway as it is on master (problems in red) and as it ends up.

Both drawings reuse the nodes of gateway.base.mmd and the package edges of GATEWAY_EDGES, so they
stay comparable with page 2.  Reality marks the #1830 findings that are still open; the ideal page
is the same set of nodes after A1, A4, A8 and A9.  The numbers in the notes come from the review
of origin/master 215c832bbe and are not recomputed here.

    python gen_gap.py [REF]    write gap_actual.mmd and gap_ideal.mmd
"""
import json
import pathlib
import re
import sys

import gen_gateway as g

OUT = pathlib.Path(__file__).parent
ID = g.ID

ACTUAL_PATCH = {
    "ADMIN": "热重载写在 routes/_shared.py,写 5 个 app 属性;读 request.app.* 28 种,约 113 处(A1)",
    "APP": "已生效配置有两个落点:app.gateway_config 与 app/_state(A1)<br/>routes_compact · routes_count_tokens 自己记账(A4)",
    "PROXY": "上游尝试的记账是 5 份拷贝之一(A4)",
    "PIPES": "记账拷贝之一(A4)",
    "CONFIG": "providers.py:build_provider_info 294 行,复杂度 49;约 20 个 ACP 辅助在这里(A8)",
    "TRANSPORT": "provider_info.py:ProviderInfo 58 个属性(不拆) · acp.py 2535 行",
}
ACTUAL_NODES = ['ACPQ["transport/acp_quota.py<br/>进程级 _READINGS / _LOCKS(A9)"]']
ACTUAL_EXTRA = [
    ("ADMIN", "APP", "-->", "读 request.app.*(A1)", True),
    ("TRANSPORT", "ACPQ", "-->", "", False),
    ("ACPQ", "QUOTA", "-.->", "状态放错地方(A9)", True),
]
ACTUAL_RED = {
    ("admin", d) for d in (
        "config", "jobs", "keystore", "maintenance", "middleware", "model_types", "observability",
        "ops", "quota", "routing", "shims", "transport", "updates",
    )
} | {("proxy", "quota"), ("pipelines", "quota"), ("config", "transport")}
ACTUAL_LABEL = {
    ("admin", "config"): "legacy allowlist 导入(A1)",
    ("proxy", "quota"): "自己记账(A4)",
    ("pipelines", "quota"): "自己记账(A4)",
    ("config", "transport"): "ACP lane 解析在 config(A8)",
}

IDEAL_PATCH = {
    "ADMINAPI": "admin 到 gateway 的唯一通道,含 reload_config",
    "APP": "热重载与唯一的已生效配置",
    "PROXY": "attempt_upstream:唯一的上游尝试记账(A4);pipelines 与 relay 路由经它",
    "CONFIG": "解析与装配;ACP lane 解析不在这里",
    "TRANSPORT": "http/ · acp/(transport · chunks · ledger · mcp · meta_client · profile · lane(ACP lane 解析,A8)· usage(纯解码,A9),A13)<br/>lane_allowance(接收器接口,A9) · 不导入 quota",
    "QUOTA": "读数与锁在这里(A9)",
    "OPS": "唯一的审计写入者",
}
IDEAL_DROP = {pair for pair in ACTUAL_RED if pair[0] == "admin"} | {("pipelines", "quota")}
IDEAL_LABEL = {("quota", "transport"): ("-.->", "启动时注册接收器(A9)")}


def patch(lines, add):
    out = []
    for line in lines:
        m = re.match(r'^(\s+)(\w+)\["(.*)"\]\s*$', line)
        if m and m.group(2) in add:
            line = f'{m.group(1)}{m.group(2)}["{m.group(3)}<br/>{add[m.group(2)]}"]'
        out.append(line)
    return out


def emit(name, node_lines, edges, extras, red_pairs, labels, drop):
    base = (OUT / "gateway.base.mmd").read_text(encoding="utf-8").rstrip("\n").splitlines()
    lines = ["flowchart TB"] + [l for l in node_lines if l] + base[1:]
    arrow_labels = json.loads((OUT / "gateway.labels.json").read_text(encoding="utf-8"))
    red = []
    count = 0
    for src in sorted(edges):
        for dst in sorted(edges[src]):
            if (src, dst) in drop:
                continue
            a, label = arrow_labels.get(f"{ID[src]}>{ID[dst]}", ["-->", ""])
            if (src, dst) in labels:
                spec = labels[(src, dst)]
                a, label = spec if isinstance(spec, tuple) else (a, spec)
            text = f'|"{label}"|' if label else ""
            lines.append(f"  {ID[src]} {a}{text} {ID[dst]}")
            if (src, dst) in red_pairs:
                red.append(count)
            count += 1
    for s, d, a, label, bad in extras:
        text = f'|"{label}"|' if label else ""
        lines.append(f"  {s} {a}{text} {d}")
        if bad:
            red.append(count)
        count += 1
    for l in (OUT / "gateway.extra.mmd").read_text(encoding="utf-8").rstrip("\n").splitlines():
        lines.append(l)
        count += 1
    if red:
        lines.append(f"  linkStyle {','.join(map(str, red))} stroke:#c0392b,stroke-width:2px")
    (OUT / f"{name}.mmd").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(name, count, "edges", len(red), "red")


if __name__ == "__main__":
    ref = sys.argv[1] if len(sys.argv) > 1 else "origin/master"
    edges = g.gateway_edges(ref)
    emit("gap_actual", [], edges, ACTUAL_EXTRA, ACTUAL_RED, ACTUAL_LABEL, set())
    emit("gap_ideal", [], edges, [], set(), IDEAL_LABEL, IDEAL_DROP)
    # node text patches go on top of the shared node block
    for name, add, extra_nodes in (("gap_actual", ACTUAL_PATCH, ACTUAL_NODES), ("gap_ideal", IDEAL_PATCH, [])):
        p = OUT / f"{name}.mmd"
        text = p.read_text(encoding="utf-8").splitlines()
        text = patch(text, add)
        for n in extra_nodes:
            text.insert(1, "  " + n)
        p.write_text("\n".join(text) + "\n", encoding="utf-8")
