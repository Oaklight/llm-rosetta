import sys, pathlib
from playwright.sync_api import sync_playwright
d = pathlib.Path(sys.argv[1])
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1500, "height": 1000})
    msgs = []
    pg.on("console", lambda m: msgs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: msgs.append(f"pageerror: {e}"))
    pg.goto(f"file://{d}/architecture.html")
    pg.wait_for_function("document.body.dataset.done === '1'", timeout=60000)
    for key in ("core", "gateway", "admin"):
        ok = pg.evaluate(f"(() => {{ const s=document.getElementById('stage-{key}'); const r=s.querySelector('.rendered'); const e=s.querySelector('pre.error'); return {{ok: !!r && !e, nodes: r? r.querySelectorAll('.node').length:0, edges: r? r.querySelectorAll('.edgePath, path.flowchart-link').length:0, clusters: r? r.querySelectorAll('.cluster').length:0, err: e? e.textContent.slice(0,300):'', w: r? r.querySelector('svg').getBoundingClientRect().width:0, h: r? r.querySelector('svg').getBoundingClientRect().height:0}}; }})()")
        print(key, ok, "zoom=", pg.evaluate(f"document.getElementById('z-{key}').textContent"))
        el = pg.query_selector(f"#stage-{key}")
        el.screenshot(path=str(d / f"{key}.png"))
    print("console issues:", msgs[:5])
    b.close()
