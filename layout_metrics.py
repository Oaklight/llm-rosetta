"""Measure a mermaid layout: edge crossings, edges through unrelated nodes, total edge length, size.

    python layout_metrics.py FILE.mmd [dagre|elk] [FLOWCHART_JSON]

Renders in headless Chromium with the same mermaid 11 that architecture.html loads, reads the SVG
geometry back and counts.  A lower crossing count and fewer through-node edges is a better layout.
"""
import json
import pathlib
import sys

from playwright.sync_api import sync_playwright

HTML = """<!doctype html><meta charset=utf-8><body><div id=out></div>
<script type=module>
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
import elk from 'https://cdn.jsdelivr.net/npm/@mermaid-js/layout-elk@0/dist/mermaid-layout-elk.esm.min.mjs';
mermaid.registerLayoutLoaders(elk);
window.go = async (src, layout, flow) => {
  mermaid.initialize({startOnLoad: false, securityLevel: 'loose', layout, flowchart: Object.assign({useMaxWidth: false}, flow)});
  const r = await mermaid.render('s' + Date.now(), src);
  document.getElementById('out').innerHTML = r.svg;
  const svg = document.querySelector('#out svg');
  const nodes = {};
  svg.querySelectorAll('g.node').forEach(g => { const m = (g.id||'').match(/flowchart-(\\w+)-/); if (!m) return; const r = g.getBoundingClientRect(); nodes[m[1]] = [r.x, r.y, r.width, r.height]; });
  const edges = [];
  svg.querySelectorAll('path.flowchart-link').forEach(p => {
    const n = Math.max(12, Math.ceil(p.getTotalLength()/8)); const pts = [];
    const ctm = p.getScreenCTM();
    for (let i = 0; i <= n; i++) { const q = p.getPointAtLength(p.getTotalLength()*i/n); pts.push([ctm.a*q.x+ctm.c*q.y+ctm.e, ctm.b*q.x+ctm.d*q.y+ctm.f]); }
    edges.push({id: p.id, pts});
  });
  const b = svg.getBoundingClientRect();
  return {nodes, edges, w: b.width, h: b.height};
};
document.body.dataset.ready = '1';
</script>"""


def cross(a, b, c, d):
    def o(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    return o(a, b, c)*o(a, b, d) < 0 and o(c, d, a)*o(c, d, b) < 0


def measure(mmd, layout="dagre", flow=None):
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"])
        pg = b.new_page(viewport={"width": 2600, "height": 1800})
        pg.set_content(HTML)
        pg.wait_for_function("document.body.dataset.ready === '1'", timeout=60000)
        d = pg.evaluate("([s,l,f]) => window.go(s,l,f)", [mmd, layout, flow or {}])
        b.close()
    ids = [l.split()[0] for l in mmd.splitlines() if "-->" in l or "-.->" in l]
    edges = d["edges"]
    segs = [[(e["pts"][i], e["pts"][i+1]) for i in range(len(e["pts"])-1)] for e in edges]
    crossings = 0
    for i in range(len(segs)):
        for j in range(i+1, len(segs)):
            hit = any(cross(a1, a2, b1, b2) for a1, a2 in segs[i][::2] for b1, b2 in segs[j][::2])
            crossings += hit
    length = sum(sum(((s[1][0]-s[0][0])**2 + (s[1][1]-s[0][1])**2) ** .5 for s in sg) for sg in segs)
    through = 0
    for e in edges:
        for name, (x, y, w, h) in d["nodes"].items():
            inside = [pt for pt in e["pts"][2:-2] if x+4 < pt[0] < x+w-4 and y+4 < pt[1] < y+h-4]
            through += bool(inside)
    return {"nodes": len(d["nodes"]), "edges": len(edges), "crossings": crossings,
            "through_node": through, "length": round(length), "w": round(d["w"]), "h": round(d["h"])}


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    layout = sys.argv[2] if len(sys.argv) > 2 else "dagre"
    flow = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
    print(pathlib.Path(sys.argv[1]).name, layout, measure(src, layout, flow))
