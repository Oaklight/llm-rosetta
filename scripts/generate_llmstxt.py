#!/usr/bin/env python3
"""Generate llms.txt and llms-full.txt from mkdocs.yml nav structure.

Reads the nav, site_name, site_description, and site_url from mkdocs.yml,
then produces:
  - llms.txt       index file linking to per-page .md sources
  - llms-full.txt  all docs concatenated into one file
  - per-page .md   copies of source files in site output dir

Usage:
    python scripts/generate_llmstxt.py [-c mkdocs.yml] [-s site] [-d docs]
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", default="mkdocs.yml", help="mkdocs.yml path")
    p.add_argument("-s", "--site-dir", default="site", help="build output directory")
    p.add_argument("-d", "--docs-dir", default="docs", help="docs source directory")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def load_config(path: str) -> dict:
    loader = yaml.SafeLoader
    loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda l, _, n: None)
    with open(path) as f:
        return yaml.load(f, Loader=loader)


def extract_leaves(node) -> list[tuple[str, str]]:
    """Recursively extract (title, path) leaf entries from a nav node."""
    results = []
    if isinstance(node, str):
        return [(node, node)]
    if isinstance(node, dict):
        for title, value in node.items():
            if isinstance(value, str):
                results.append((title, value))
            else:
                results.extend(extract_leaves(value))
    if isinstance(node, list):
        for item in node:
            results.extend(extract_leaves(item))
    return results


def resolve_title(title: str, path: str, docs_dir: Path) -> str:
    """If title looks like a file path, try to extract H1 from the .md file."""
    if not title.endswith(".md"):
        return title
    source = docs_dir / path
    if source.exists():
        for line in source.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^#\s+(.+)", line)
            if m:
                return m.group(1).strip()
    return Path(path).stem.replace("-", " ").replace("_", " ").title()


def build_sections(nav: list) -> list[tuple[str, list[tuple[str, str]]]]:
    """Convert top-level nav into (section_name, [(title, path), ...])."""
    sections = []
    for item in nav:
        if isinstance(item, dict):
            for section_name, children in item.items():
                if isinstance(children, str):
                    sections.append((section_name, [(section_name, children)]))
                else:
                    leaves = extract_leaves(children)
                    sections.append((section_name, leaves))
        elif isinstance(item, str):
            sections.append((item, [(item, item)]))
    return sections


def generate_llms_txt(
    site_name: str,
    site_description: str,
    site_url: str,
    docs_dir: Path,
    sections: list[tuple[str, list[tuple[str, str]]]],
) -> str:
    base = site_url.rstrip("/")
    lines = [
        f"# {site_name}",
        "",
        f"> {site_description}",
        "",
        f"For all content in a single file, see [{site_name} full docs]({base}/llms-full.txt).",
    ]
    for section_name, leaves in sections:
        lines.append("")
        lines.append(f"## {section_name}")
        lines.append("")
        for title, path in leaves:
            display = resolve_title(title, path, docs_dir)
            lines.append(f"- [{display}]({base}/{path})")
    lines.append("")
    return "\n".join(lines)


def generate_llms_full(
    site_name: str,
    site_description: str,
    docs_dir: Path,
    sections: list[tuple[str, list[tuple[str, str]]]],
) -> str:
    lines = [f"# {site_name}", "", f"> {site_description}"]
    seen: set[str] = set()
    for section_name, leaves in sections:
        lines.append("")
        lines.append(f"## {section_name}")
        for _title, path in leaves:
            if path in seen:
                continue
            seen.add(path)
            source = docs_dir / path
            if source.exists():
                content = source.read_text(encoding="utf-8").strip()
                lines.append("")
                lines.append(content)
            else:
                lines.append("")
                lines.append(f"<!-- {path} not found -->")
    lines.append("")
    return "\n".join(lines)


LLMSTXT_LINKS_JS = """\
<script>document.addEventListener("DOMContentLoaded",function(){
var a=document.querySelector(".md-content__inner");if(!a)return;
var c=document.querySelector('link[rel="canonical"]');if(!c)return;
var h=c.getAttribute("href"),m=h.replace(/\\/$/,"");
var s=m.split("/"),l=s[s.length-1];
m=(!l||l==="latest")?h+"index.md":m+".md";
var b=h.match(/^https?:\\/\\/[^/]+(\\/[^/]+\\/[^/]+\\/)/);
var u=b?b[0]+"llms.txt":"/llms.txt";
var md='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M20.56 18H3.44C2.65 18 2 17.37 2 16.59V7.41C2 6.63 2.65 6 3.44 6h17.12c.79 0 1.44.63 1.44 1.41v9.18c0 .78-.65 1.41-1.44 1.41M3.44 6.94c-.26 0-.48.21-.48.47v9.19c0 .25.22.46.48.46h17.12c.26 0 .48-.21.48-.46V7.41c0-.26-.22-.47-.48-.47zm1.45 8.25V8.81h1.92l1.92 2.35 1.92-2.35h1.93v6.38h-1.93v-3.66l-1.92 2.35-1.92-2.35v3.66zm12.01 0-2.9-3.1h1.94V8.81h1.92v3.28h1.93z"/></svg>';
var rb='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M17.5 15.5c0 1.11-.89 2-2 2s-2-.89-2-2 .9-2 2-2 2 .9 2 2m-9-2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.89 2-2-.89-2-2-2M23 15v3c0 .55-.45 1-1 1h-1v1c0 1.11-.89 2-2 2H5a2 2 0 0 1-2-2v-1H2c-.55 0-1-.45-1-1v-3c0-.55.45-1 1-1h1c0-3.87 3.13-7 7-7h1V5.73c-.6-.34-1-.99-1-1.73 0-1.1.9-2 2-2s2 .9 2 2c0 .74-.4 1.39-1 1.73V7h1c3.87 0 7 3.13 7 7h1c.55 0 1 .45 1 1m-2 1h-2v-2c0-2.76-2.24-5-5-5h-4c-2.76 0-5 2.24-5 5v2H3v1h2v3h14v-3h2z"/></svg>';
var e=document.createElement("a");e.href=m;e.title="View page source";
e.className="md-content__button md-icon";e.innerHTML=md;
var f=document.createElement("a");f.href=u;f.title="llms.txt";
f.className="md-content__button md-icon";f.innerHTML=rb;
a.insertBefore(f,a.firstChild);a.insertBefore(e,a.firstChild);
});</script>"""


# Unique sentinel so pages that merely mention "llms.txt" are not skipped.
LLMSTXT_MARKER = 'e.title="View page source"'


def inject_links_into_html(site_dir: Path) -> int:
    """Inject llms.txt link buttons into all HTML pages."""
    count = 0
    for html_file in site_dir.rglob("*.html"):
        content = html_file.read_text(encoding="utf-8")
        if "</head>" in content and LLMSTXT_MARKER not in content:
            content = content.replace("</head>", LLMSTXT_LINKS_JS + "\n</head>")
            html_file.write_text(content, encoding="utf-8")
            count += 1
    return count


def copy_md_sources(
    docs_dir: Path, site_dir: Path, sections: list[tuple[str, list[tuple[str, str]]]]
) -> int:
    seen: set[str] = set()
    count = 0
    for _section, leaves in sections:
        for _title, path in leaves:
            if path in seen:
                continue
            seen.add(path)
            src = docs_dir / path
            dst = site_dir / path
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                count += 1
    return count


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    site_name = config.get("site_name", "Documentation")
    site_description = config.get("site_description", "")
    site_url = config.get("site_url", "")
    nav = config.get("nav", [])

    docs_dir = Path(args.docs_dir)
    site_dir = Path(args.site_dir)

    sections = build_sections(nav)

    llms_txt = generate_llms_txt(site_name, site_description, site_url, docs_dir, sections)
    (site_dir / "llms.txt").write_text(llms_txt, encoding="utf-8")
    if args.verbose:
        print(f"Generated llms.txt ({len(sections)} sections)")

    llms_full = generate_llms_full(site_name, site_description, docs_dir, sections)
    (site_dir / "llms-full.txt").write_text(llms_full, encoding="utf-8")
    if args.verbose:
        print(f"Generated llms-full.txt ({len(llms_full)} chars)")

    count = copy_md_sources(docs_dir, site_dir, sections)
    if args.verbose:
        print(f"Copied {count} .md files to {site_dir}")

    injected = inject_links_into_html(site_dir)
    if args.verbose:
        print(f"Injected link buttons into {injected} HTML pages")


if __name__ == "__main__":
    main()
