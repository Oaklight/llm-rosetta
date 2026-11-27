"""Static reading of an extension's modules: no import of the extension."""

from __future__ import annotations

import ast


def literal_keys(tree: ast.Module, name: str) -> set[str] | None:
    """Return the string keys of the module-level dict literal *name*, if any."""
    keys: set[str] | None = None
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        value = getattr(node, "value", None)
        if any(isinstance(t, ast.Name) and t.id == name for t in targets) and (
            isinstance(value, ast.Dict)
        ):
            keys = {
                key.value
                for key in value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    return keys
