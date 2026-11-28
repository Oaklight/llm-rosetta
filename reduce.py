"""Transitive reduction shared by the generators."""


def reduction(edges, keep):
    """Package edges that another path already implies (transitive reduction), minus *keep*."""
    graph = {a: set(b) for a, b in edges.items()}

    def reaches(src, dst, skip):
        seen, todo = set(), [src]
        while todo:
            n = todo.pop()
            for m in graph.get(n, ()):
                if (n, m) == skip or m in seen:
                    continue
                if m == dst:
                    return True
                seen.add(m)
                todo.append(m)
        return False

    return {(a, b) for a, bs in edges.items() for b in bs if (a, b) not in keep and reaches(a, b, (a, b))}
