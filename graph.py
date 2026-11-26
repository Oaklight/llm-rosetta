import ast, sys, collections
from pathlib import Path
def walk_module_level(tree):
    out = []
    def visit(node, in_func=False):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return
        if isinstance(node, ast.If):
            t = node.test
            name = getattr(t, "id", None) or getattr(t, "attr", None)
            if name == "TYPE_CHECKING":
                for n in node.orelse: visit(n)
                return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child)
    for n in tree.body: visit(n)
    return out

def graph(root: Path, pkg: str, skip=("_vendor",), core_prefix=None):
    edges = collections.defaultdict(set)
    base = root / pkg
    for path in base.rglob("*.py"):
        rel = path.relative_to(base)
        if rel.parts[0] in skip or "__pycache__" in rel.parts: continue
        owner = rel.parts[0] if len(rel.parts) > 1 else rel.stem
        mod_parts = [pkg, *rel.with_suffix("").parts]
        if mod_parts[-1] == "__init__": mod_parts = mod_parts[:-1]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in walk_module_level(tree):
            targets = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    pkgpath = mod_parts[: len(mod_parts) - node.level + (0 if path.name == "__init__.py" else 0)]
                    # relative base: for a module file, parent package is mod_parts[:-1]
                    parent = mod_parts if path.name == "__init__.py" else mod_parts[:-1]
                    baseparts = parent[: len(parent) - (node.level - 1)]
                    name = ".".join(baseparts + ([node.module] if node.module else []))
                else:
                    name = node.module or ""
                targets.append((name, [a.name for a in node.names]))
            elif isinstance(node, ast.Import):
                for a in node.names: targets.append((a.name, []))
            for name, names in targets:
                parts = name.split(".")
                if parts[0] == pkg:
                    if len(parts) > 1:
                        t = parts[1]
                    else:
                        # from llm_rosetta import X  -> X is a module/package
                        for n in names:
                            edges[owner].add(n)
                        continue
                    edges[owner].add(t)
                elif core_prefix and parts[0] == core_prefix:
                    sub = parts[1] if len(parts) > 1 else "(root)"
                    edges[owner].add(f"CORE:{core_prefix}.{sub}")
    return edges
if __name__ == "__main__":
    which = sys.argv[1]
    if which == "core":
        e = graph(Path("llm-rosetta/src"), "llm_rosetta")
    else:
        e = graph(Path("llm-rosetta-gateway/src"), "llm_rosetta_gateway", core_prefix="llm_rosetta")
    nodes = set(e) | {t for v in e.values() for t in v}
    for k in sorted(e):
        ts = sorted(t for t in e[k] if t != k and (t in nodes))
        print(f"{k} -> {', '.join(ts)}")
