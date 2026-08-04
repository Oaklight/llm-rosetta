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


if __name__ == "__main__":
    main()
