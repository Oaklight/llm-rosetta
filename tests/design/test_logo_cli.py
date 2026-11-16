"""The logo CLI runs a draft the page exported.

The page and the CLI share one pipeline, and the page is where a design is
made: ``--draft`` is what keeps the two from drifting apart.  These tests pin
the two rules that depends on -- a draft alone reproduces every shipped copy,
and a flag written beside ``--draft`` wins over the draft's value.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

LOGO = Path(__file__).resolve().parents[2] / "design" / "logo"
DRAFT = LOGO / "references" / "rosetta-draft.json"
AUTH = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "llm-rosetta-gateway"
    / "src"
    / "llm_rosetta_gateway"
    / "admin"
    / "routes"
    / "auth.py"
)


def _load_cli() -> ModuleType:
    """Import ``design/logo/simplify_path.py``, which is not a package module."""
    spec = importlib.util.spec_from_file_location(
        "_logo_simplify_path", LOGO / "simplify_path.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


@pytest.fixture(scope="module")
def small_draft(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A minimal draft, so the override rules are tested by themselves."""
    src = (LOGO / "references" / "rosetta-stone-path.txt").read_text()
    path = tmp_path_factory.mktemp("draft") / "draft.json"
    path.write_text(
        json.dumps(
            {
                "v": 3,
                "src": src,
                "params": {
                    "points": 15,
                    "anchor": 15,
                    "tol": 0.52,
                    "smooth": 1,
                    "tight": True,
                    "pad": 0.04,
                    "fill": "#1e293b",
                    "darkFill": "#ededed",
                },
                "pins": [0.238014, 0.452824],
                "unpins": [0.700878],
                "locks": [[0.238014, 434.96, 49.06]],
                "sharp": [0.157055, 0.728426],
                "round": [0.086083, 0.984273],
                "centerRef": "bbox",
            }
        )
    )
    return path


def _args(*argv: str):
    args = cli.build_parser().parse_args(list(argv))
    cli.apply_draft(args, list(argv))
    return args


@pytest.mark.parametrize(
    "flag, attr, given, expected",
    [
        ("--points", "points", "3", "3"),
        ("--scale", "scale", "0.5", 0.5),
        ("--fill", "fill", "#000000", "#000000"),
        ("--pin", "pin", "u0.5", "u0.5"),
        ("--unpin", "unpin", "u0.5", "u0.5"),
        ("--sharp", "sharp", "u0.5", "u0.5"),
        ("--round", "round", "u0.5", "u0.5"),
        ("--lock", "lock", "u0.5@1.00,2.00", "u0.5@1.00,2.00"),
        ("--center", "center", "centroid", "centroid"),
    ],
)
def test_flag_beside_draft_wins(
    small_draft: Path, flag: str, attr: str, given: str, expected: object
) -> None:
    args = _args("--draft", str(small_draft), flag, given)
    assert getattr(args, attr) == expected


def test_draft_alone_supplies_the_lists(small_draft: Path) -> None:
    args = _args("--draft", str(small_draft))
    assert args.points == "15"
    assert args.sharp == "u0.157055,u0.728426"
    assert args.round == "u0.086083,u0.984273"
    assert args.pin == "u0.238014,u0.452824"
    assert args.unpin == "u0.700878"
    assert args.lock == "u0.238014@434.96,49.06"
    assert args.center == "bbox"


@pytest.mark.parametrize(
    "name, extra",
    [
        ("rosetta-icon.svg", []),
        ("rosetta-icon-github.svg", []),
        ("rosetta-icon-docs.svg", ["--dark-fill", ""]),
        ("favicon.svg", ["--fill", "#666666"]),
        ("favicon-gateway.svg", ["--fill", "#666666"]),
    ],
)
def test_shipped_copies_come_from_the_draft(
    tmp_path: Path, name: str, extra: list[str]
) -> None:
    """Every file in ``out/`` is the draft plus at most a fill override."""
    out = tmp_path / name
    cli.main(["--draft", str(DRAFT), *extra, "-o", str(out)])
    assert out.read_bytes() == (LOGO / "out" / name).read_bytes()


def test_the_gateway_archive_is_what_the_gateway_serves() -> None:
    """``out/favicon-gateway.svg`` is a copy of ``_FAVICON_SVG``.

    The gateway favicon is the draft at ``--fill '#666666'``; the archive is
    the byte string the gateway actually serves.
    """
    tree = ast.parse(AUTH.read_text())
    served = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", "") == "_FAVICON_SVG"
    )
    assert (LOGO / "out" / "favicon-gateway.svg").read_bytes() == served
