import sys
from playwright.sync_api import sync_playwright
JS = """(key) => {
  const root = document.getElementById('stage-' + key);
  const box = (el) => { const r = el.getBoundingClientRect(); return {x: r.x, y: r.y, w: r.width, h: r.height, cx: r.x + r.width/2, cy: r.y + r.height/2}; };
  const nodes = {};
  root.querySelectorAll('g.node').forEach(g => { const m = (g.id || '').match(/flowchart-([A-Za-z]+)-/); if (m) nodes[m[1]] = box(g); });
  const clusters = {};
  root.querySelectorAll('g.cluster').forEach(g => { const t = (g.textContent || '').trim().slice(0, 14); clusters[t] = box(g.querySelector('rect')); });
  return {nodes, clusters};
}"""
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1500, "height": 1000})
    pg.goto(f"file://{sys.argv[1]}/architecture.html")
    pg.wait_for_function("document.body.dataset.done === '1'", timeout=60000)
    for key in ("core", "core_target"):
        d = pg.evaluate(JS, key)
        n, c = d["nodes"], d["clusters"]
        order = [k for k in ("PIPE", "CVT", "IROPS", "IRMODEL") if k in n]
        ys = [round(n[k]["cy"]) for k in order]
        print(key, "tier order (top to bottom should rise):", dict(zip(order, ys)), "->", "OK" if ys == sorted(ys) and len(set(ys)) == len(ys) else "NOT ORDERED")
        core = next((v for t, v in c.items() if t.startswith("转换库")), None)
        if core:
            for k in ("SHIM", "EXTS"):
                if k in n:
                    inside = core["x"] <= n[k]["cx"] <= core["x"] + core["w"] and core["y"] <= n[k]["cy"] <= core["y"] + core["h"]
                    print(f"  {k} inside the conversion group: {inside} (should be False)")
            for k in ("PIPE", "CVT", "IROPS", "IRMODEL"):
                if k in n:
                    inside = core["x"] <= n[k]["cx"] <= core["x"] + core["w"] and core["y"] <= n[k]["cy"] <= core["y"] + core["h"]
                    print(f"  {k} inside the conversion group: {inside} (should be True)")
        print("  clusters:", sorted(c))
    b.close()
