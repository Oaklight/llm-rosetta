"""gateway.mmd = hand-kept nodes (gateway.base.mmd) + package edges read from GATEWAY_EDGES.

The package edges are the table that tests/gateway/test_layers.py enforces, parsed from a git
ref (default origin/master), so the picture cannot drift from the test.  Edge labels live in
gateway.labels.json; edges that are not package imports (core, client, upstream) live in
gateway.extra.mmd.

    python gen_gateway.py [REF]      write gateway.mmd
    python gen_gateway.py --check    exit 1 when gateway.mmd is not what REF would give
"""
import ast
import json
import pathlib
import subprocess
import sys

from reduce import reduction

OUT = pathlib.Path(__file__).parent
REPO = pathlib.Path("/home/pding/projects/llm-rosetta")
ID = {
    "keystore": "KEYSTORE", "maintenance": "MAINT", "jobs": "JOBS", "quota": "QUOTA",
    "routing": "ROUTING", "config": "CONFIG", "middleware": "MW", "transport": "TRANSPORT",
    "observability": "OBS", "model_types": "MT", "shims": "SHIMS", "proxy": "PROXY",
    "pipelines": "PIPES", "ops": "OPS", "cli": "CLI", "admin_api": "ADMINAPI",
    "admin": "ADMIN", "app": "APP", "surface": "SURF", "updates": "UPD",
}


def gateway_edges(ref):
    src = subprocess.check_output(
        ["git", "-C", str(REPO), "show", f"{ref}:tests/gateway/test_layers.py"], text=True
    )
    for node in ast.parse(src).body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "GATEWAY_EDGES":
            return ast.literal_eval(node.value)
    raise SystemExit("GATEWAY_EDGES not found")


def render(ref, full=False):
    labels = json.loads((OUT / "gateway.labels.json").read_text(encoding="utf-8"))
    edges = gateway_edges(ref)
    unknown = {p for k, v in edges.items() for p in (k, *v)} - set(ID)
    if unknown:
        raise SystemExit(f"package without a node id: {sorted(unknown)} (add it to ID and the base)")
    lines = ["---", "config:", "  layout: elk", "---", "flowchart TB"]
    lines += (OUT / "gateway.base.mmd").read_text(encoding="utf-8").rstrip("\n").splitlines()[1:]
    keep = {(a, b) for a in edges for b in edges[a] if labels.get(f"{ID[a]}>{ID[b]}", ["", ""])[1]}
    implied = set() if full else reduction(edges, keep)
    for src in sorted(edges):
        for dst in sorted(edges[src]):
            if (src, dst) in implied:
                continue
            arrow, label = labels.get(f"{ID[src]}>{ID[dst]}", ["-->", ""])
            text = f'|"{label}"|' if label else ""
            lines.append(f"  {ID[src]} {arrow}{text} {ID[dst]}")
    lines += (OUT / "gateway.extra.mmd").read_text(encoding="utf-8").rstrip("\n").splitlines()
    return "\n".join(lines) + "\n", sum(len(v) for v in edges.values()) - len(implied)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    text, n = render(args[0] if args else "origin/master", "--full" in sys.argv)
    if "--check" in sys.argv:
        same = (OUT / "gateway.mmd").read_text(encoding="utf-8") == text
        print("gateway.mmd up to date" if same else "gateway.mmd is stale")
        raise SystemExit(0 if same else 1)
    (OUT / "gateway.mmd").write_text(text, encoding="utf-8")
    print("gateway.mmd:", n, "package edges")
