"""The retired-names table: shape, messages, switch and the boundaries it guards.

The provider-type and shim alias tables are gone (#919).  A retired spelling is
no longer translated; it fails at every boundary with a message that names its
replacement.  The two model-type spellings (``embedding``, ``decision``) still
translate while :data:`~llm_rosetta.retired_names.ALIASES_ENABLED` is ``True``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import warnings

import pytest

from llm_rosetta import ConversionPipeline, convert, convert_response, retired_names
from llm_rosetta import get_converter_for_provider
from llm_rosetta.converters.embeddings.lookup import get_embedding_converter
from llm_rosetta.retired_names import (
    MODEL_TYPE,
    PROVIDER_TYPE,
    RETIRED_NAMES,
    SHIM,
    check_not_retired,
    lookup,
    retired_name_message,
)
from llm_rosetta.shims import ProviderShim, get_shim, resolve_model_type
from llm_rosetta.shims._loader import load_providers_from_dir

BODY = dict(
    model="gpt-4o",
    messages=[dict(role="user", content="hello")],
)
RESPONSE = dict(
    id="chatcmpl-1",
    object="chat.completion",
    model="gpt-4o",
    choices=[
        dict(
            index=0,
            message=dict(role="assistant", content="hi"),
            finish_reason="stop",
        )
    ],
)

#: Bare chat formats that are also listed as retired *shim* names.  In a shim
#: slot they are formats, not shims, and must not be rejected.
FORMAT_SHIM_NAMES = frozenset(["google_generate", "google_interactions"])

ROWS = [pytest.param(row, id=f"{row.kind}:{row.name}") for row in RETIRED_NAMES]
PROVIDER_ROWS = [r for r in RETIRED_NAMES if r.kind == PROVIDER_TYPE]
SHIM_ROWS = [r for r in RETIRED_NAMES if r.kind == SHIM]
SHIM_SLOT_ROWS = [r for r in SHIM_ROWS if r.name not in FORMAT_SHIM_NAMES]
MODEL_TYPE_ROWS = [r for r in RETIRED_NAMES if r.kind == MODEL_TYPE]


def _ids(rows):
    return [pytest.param(r, id=r.name) for r in rows]


def _replacement(name: str, *kinds: str) -> str:
    """The replacement a boundary reading *name* under *kinds* names."""
    for kind in kinds:
        row = lookup(name, kind)
        if row is not None:
            return row.replacement
    raise AssertionError(f"{name!r} is not retired under {kinds}")


# ---------------------------------------------------------------------------
# The alias tables are gone
# ---------------------------------------------------------------------------


def test_provider_names_module_is_gone() -> None:
    assert importlib.util.find_spec("llm_rosetta.provider_names") is None


def test_alias_functions_are_not_exported() -> None:
    import llm_rosetta
    from llm_rosetta import api

    for name in (
        "canonical_provider_name",
        "canonical_shim_name",
        "legacy_names",
        "normalize_provider_name",
        "warn_legacy_provider_names",
        "warn_legacy_shim_names",
    ):
        assert not hasattr(api, name), name
        assert not hasattr(llm_rosetta, name), name
        assert name not in api.__all__
        assert name not in llm_rosetta.__all__


def test_provider_type_literal_has_no_legacy_members() -> None:
    from typing import get_args

    from llm_rosetta import ProviderType

    members = set(get_args(ProviderType))
    assert not members.intersection(row.name for row in PROVIDER_ROWS)
    assert members.issuperset(["chat_completions", "messages", "google_generate"])


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_the_model_type_rows() -> None:
    assert {row.name: row.replacement for row in MODEL_TYPE_ROWS} == dict(
        embedding="embeddings", decision="decisions"
    )


def test_table_size_by_kind() -> None:
    sizes = dict(
        (kind, sum(1 for row in RETIRED_NAMES if row.kind == kind))
        for kind in retired_names.KINDS
    )
    assert sizes == dict([(PROVIDER_TYPE, 9), (SHIM, 20), (MODEL_TYPE, 2)])


def test_no_duplicate_rows() -> None:
    keys = [(row.kind, row.name) for row in RETIRED_NAMES]
    assert len(keys) == len(set(keys))


def test_no_chains() -> None:
    """A replacement is never itself retired under the same kind."""
    for row in RETIRED_NAMES:
        assert lookup(row.replacement, row.kind) is None, row
        assert row.replacement != row.name


def test_kinds_are_known() -> None:
    assert set(retired_names.KINDS).issuperset(row.kind for row in RETIRED_NAMES)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", ROWS)
def test_message_names_the_replacement(row: retired_names.RetiredName) -> None:
    message = retired_name_message(row.name, row.kind)
    assert message is not None
    assert f"{row.name!r} was renamed to {row.replacement!r} in 0.14" in message
    assert "no longer accepted" in message
    if row.install:
        assert f"shims.install: ['{row.install}']" in message
    else:
        assert "shims.install" not in message


@pytest.mark.parametrize("row", ROWS)
def test_message_ends_with_the_migrate_hint(row: retired_names.RetiredName) -> None:
    message = retired_name_message(row.name, row.kind)
    assert message is not None
    assert message.endswith(retired_names.MIGRATE_HINT)


# ---------------------------------------------------------------------------
# The hint names a command the reader has (#1657, C8)
# ---------------------------------------------------------------------------


def test_hint_names_the_command_when_the_gateway_is_installed() -> None:
    hint = retired_names._migrate_hint(lambda name: object())
    assert hint.startswith(
        "run 'llm-rosetta gateway migrate --config <path>' (add --write to apply), "
    )
    assert retired_names.MIGRATION_GUIDE_URL in hint


def test_hint_names_only_the_guide_for_a_core_only_install() -> None:
    hint = retired_names._migrate_hint(lambda name: None)
    assert "llm-rosetta gateway migrate" not in hint
    assert hint == f"see the migration guide ({retired_names.MIGRATION_GUIDE_URL})"


@pytest.mark.parametrize("error", [ModuleNotFoundError, ValueError])
def test_a_finder_that_refuses_the_gateway_means_core_only(error) -> None:
    """``find_spec`` can raise; the hint falls back instead of failing import."""

    def refuse(name: str) -> None:
        raise error(name)

    hint = retired_names._migrate_hint(refuse)
    assert "llm-rosetta gateway migrate" not in hint


def test_hint_looks_up_the_gateway_distribution() -> None:
    asked: list[str] = []
    retired_names._migrate_hint(lambda name: asked.append(name))
    assert asked == ["llm_rosetta_gateway"]


def test_this_install_gets_the_hint_of_its_own_branch() -> None:
    installed = importlib.util.find_spec("llm_rosetta_gateway") is not None
    assert ("llm-rosetta gateway migrate" in retired_names.MIGRATE_HINT) is installed


def test_the_guide_url_names_the_migration_guide() -> None:
    assert retired_names.MIGRATION_GUIDE_URL.endswith("/guide/migrating-to-0.14/")


def test_computing_the_hint_does_not_import_the_gateway() -> None:
    """``find_spec`` locates the package; core must never import it."""
    code = (
        "import sys\n"
        "import llm_rosetta.retired_names as r\n"
        "assert 'llm-rosetta gateway migrate' in r.MIGRATE_HINT, r.MIGRATE_HINT\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('llm_rosetta_gateway'))\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_a_core_only_interpreter_gets_the_guide() -> None:
    """A core-only install, simulated: ``llm_rosetta_gateway`` cannot be found.

    A ``None`` entry in ``sys.modules`` is how the import system marks a
    module as absent; ``find_spec`` then returns ``None``.
    """
    code = (
        "import sys\n"
        "sys.modules['llm_rosetta_gateway'] = None\n"
        "import llm_rosetta.retired_names as r\n"
        "print(r.MIGRATE_HINT)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "llm-rosetta gateway migrate" not in result.stdout
    assert retired_names.MIGRATION_GUIDE_URL in result.stdout


@pytest.mark.parametrize("name", ["openai_chatt", "chat_completions", "", "Embedding"])
def test_unretired_names_get_no_hint(name: str) -> None:
    for kind in retired_names.KINDS:
        assert retired_name_message(name, kind) is None


# ---------------------------------------------------------------------------
# Every row raises; there is no switch
# ---------------------------------------------------------------------------


def test_the_switch_is_gone() -> None:
    assert not hasattr(retired_names, "ALIASES_ENABLED")


@pytest.mark.parametrize("row", ROWS)
def test_check_raises_for_every_row(row: retired_names.RetiredName) -> None:
    with pytest.raises(ValueError, match="was renamed") as info:
        check_not_retired(row.name, row.kind)
    assert row.replacement in str(info.value)


def test_check_tries_the_kinds_in_order() -> None:
    with pytest.raises(ValueError, match="'google_generate'"):
        check_not_retired("google", PROVIDER_TYPE, SHIM)
    with pytest.raises(ValueError, match="'google--google_generate'"):
        check_not_retired("google", SHIM, PROVIDER_TYPE)


def test_non_string_is_ignored() -> None:
    check_not_retired(None, PROVIDER_TYPE, SHIM)
    check_not_retired(3, PROVIDER_TYPE, SHIM)


def test_retired_key_message() -> None:
    message = retired_names.retired_key_message("embedding_path", "embeddings_path")
    assert "'embedding_path' was renamed to 'embeddings_path' in 0.14" in message
    assert message.endswith(retired_names.MIGRATE_HINT)


# ---------------------------------------------------------------------------
# Retired model types fail at every library boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", _ids(MODEL_TYPE_ROWS))
def test_the_model_type_vocabulary_is_gone(row: retired_names.RetiredName) -> None:
    import llm_rosetta.formats as formats

    for name in ("LEGACY_MODEL_TYPES", "canonical_model_type", "legacy_model_type"):
        assert not hasattr(formats, name), name


@pytest.mark.parametrize("row", _ids(MODEL_TYPE_ROWS))
def test_registry_and_pipelines_reject_a_retired_model_type(
    row: retired_names.RetiredName,
) -> None:
    from llm_rosetta.api import typed_response_usage
    from llm_rosetta.converters.registry import (
        converter_map,
        formats_for,
        get_converter_class,
    )
    from llm_rosetta.fidelity import FidelityChecker

    expected = repr(row.replacement)
    with pytest.raises(ValueError, match=expected):
        formats_for(row.name)
    with pytest.raises(ValueError, match=expected):
        converter_map(row.name)
    with pytest.raises(ValueError, match=expected):
        get_converter_class(row.name, "openai")
    with pytest.raises(ValueError, match=expected):
        typed_response_usage(row.name, "openai", {})
    with pytest.raises(ValueError, match=expected):
        FidelityChecker(model_type=row.name)
    with pytest.raises(ValueError, match=expected):
        resolve_model_type(row.name)
    with pytest.raises(ValueError, match=expected):
        ProviderShim(name="x", format="openai", model_type=row.name)


@pytest.mark.parametrize("row", _ids(MODEL_TYPE_ROWS))
def test_a_shim_yaml_with_a_retired_model_type_is_rejected(
    row: retired_names.RetiredName, tmp_path
) -> None:

    leaf = tmp_path / "leaf"
    leaf.mkdir()
    (leaf / "provider.yaml").write_text(
        f"name: x\nmodel_type: {row.name}\nformat: openai\n"
    )
    with pytest.raises(ValueError, match=repr(row.replacement)):
        load_providers_from_dir(tmp_path, on_error="raise")


@pytest.mark.parametrize(
    ("old", "new", "fmt"),
    [("embedding", "embeddings", "openai"), ("decision", "decisions", "systemone")],
)
def test_a_shim_yaml_with_a_retired_type_section_is_rejected(
    old: str, new: str, fmt: str, tmp_path
) -> None:

    leaf = tmp_path / "leaf"
    leaf.mkdir()
    (leaf / "provider.yaml").write_text(
        f"name: x\nmodel_type: {new}\nformat: {fmt}\n{old}:\n  dimensions: 3\n"
    )
    with pytest.raises(ValueError) as info:
        load_providers_from_dir(tmp_path, on_error="raise")
    assert f"{old}:" in str(info.value) and f"{new}:" in str(info.value)


# ---------------------------------------------------------------------------
# Retired provider types fail at every library boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", _ids(PROVIDER_ROWS))
def test_get_converter_names_the_replacement(row: retired_names.RetiredName) -> None:
    with pytest.raises(ValueError, match="was renamed") as info:
        get_converter_for_provider(row.name)
    assert repr(row.replacement) in str(info.value)


@pytest.mark.parametrize("row", _ids(PROVIDER_ROWS))
def test_pipeline_rejects_a_retired_source_and_target(
    row: retired_names.RetiredName,
) -> None:
    with pytest.raises(ValueError, match=repr(row.replacement)):
        ConversionPipeline(row.name, "messages")
    with pytest.raises(ValueError, match=repr(row.replacement)):
        ConversionPipeline("messages", row.name)


@pytest.mark.parametrize("row", _ids(PROVIDER_ROWS))
def test_convert_rejects_a_retired_target_and_source(
    row: retired_names.RetiredName,
) -> None:
    with pytest.raises(ValueError, match=repr(row.replacement)):
        convert(BODY, row.name)
    with pytest.raises(ValueError, match=repr(row.replacement)):
        convert(BODY, "messages", source_provider=row.name)
    with pytest.raises(ValueError, match=repr(row.replacement)):
        convert_response(RESPONSE, BODY, row.name, "chat_completions")
    with pytest.raises(ValueError, match=repr(row.replacement)):
        convert_response(RESPONSE, BODY, "chat_completions", row.name)


@pytest.mark.parametrize("row", _ids(PROVIDER_ROWS))
def test_embedding_converter_rejects_a_retired_format(
    row: retired_names.RetiredName,
) -> None:
    with pytest.raises(ValueError, match=repr(row.replacement)):
        get_embedding_converter(row.name)


@pytest.mark.parametrize("row", _ids(PROVIDER_ROWS))
def test_provider_shim_rejects_a_retired_format(
    row: retired_names.RetiredName,
) -> None:
    with pytest.raises(ValueError, match=repr(row.replacement)):
        ProviderShim(name="x", format=row.name)


# ---------------------------------------------------------------------------
# Retired shim names fail where a shim name enters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", _ids(SHIM_SLOT_ROWS))
def test_pipeline_rejects_a_retired_shim_string(
    row: retired_names.RetiredName,
) -> None:
    attempts = [
        lambda: ConversionPipeline(
            "chat_completions", "chat_completions", source_shim=row.name
        ),
        lambda: ConversionPipeline(
            "chat_completions", "chat_completions", target_shim=row.name
        ),
    ]
    for attempt in attempts:
        with pytest.raises(ValueError, match=repr(row.replacement)) as info:
            attempt()
        if row.install:
            assert f"shims.install: ['{row.install}']" in str(info.value)


@pytest.mark.parametrize("row", _ids(SHIM_SLOT_ROWS))
def test_convert_rejects_a_retired_shim_as_provider(
    row: retired_names.RetiredName,
) -> None:
    expected = _replacement(row.name, PROVIDER_TYPE, SHIM)
    with pytest.raises(ValueError, match=repr(expected)):
        convert(BODY, row.name)


@pytest.mark.parametrize("row", _ids(SHIM_SLOT_ROWS))
def test_a_retired_shim_name_is_not_a_registered_shim(
    row: retired_names.RetiredName,
) -> None:
    assert get_shim(row.name) is None


@pytest.mark.parametrize("name", sorted(FORMAT_SHIM_NAMES))
def test_a_base_format_is_not_rejected_in_a_shim_slot(name: str) -> None:
    """``google_generate`` is retired as a shim name but is a format."""
    ConversionPipeline("chat_completions", "chat_completions", target_shim=name)
    convert(BODY, name)


def test_replacement_shim_names_still_work() -> None:
    assert get_shim("openai--chat_completions") is not None
    assert get_shim("openai--open_responses") is not None
    ConversionPipeline(
        "chat_completions",
        "chat_completions",
        target_shim="deepseek--chat_completions",
    )


# ---------------------------------------------------------------------------
# Typos and canonical names
# ---------------------------------------------------------------------------


def test_typos_keep_the_plain_message() -> None:
    with pytest.raises(ValueError, match="Unsupported provider: openai_chatt") as info:
        get_converter_for_provider("openai_chatt")
    assert "was renamed" not in str(info.value)
    with pytest.raises(ValueError) as info:
        get_embedding_converter("googl")
    assert "was renamed" not in str(info.value)


def test_canonical_names_never_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        convert(BODY, "chat_completions")
        ConversionPipeline("chat_completions", "messages")
        get_converter_for_provider("open_responses")
        get_converter_for_provider("openai--open_responses")


# ---------------------------------------------------------------------------
# Lookups no longer translate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", _ids(PROVIDER_ROWS))
def test_resolve_format_leaves_a_retired_name_alone(
    row: retired_names.RetiredName,
) -> None:
    from llm_rosetta.shims import resolve_format

    assert resolve_format(row.name) == row.name


def test_register_shim_keys_on_the_exact_name() -> None:
    """A shim named like a retired spelling is reachable under that name only."""
    import llm_rosetta.shims.provider_shim as ps

    saved = dict(ps._SHIM_REGISTRY)
    try:
        ps._SHIM_REGISTRY.clear()
        neutral = ProviderShim(name="open_responses", format="open_responses")
        profile = ProviderShim(name="openai_responses", format="open_responses")
        ps.register_shim(neutral)
        ps.register_shim(profile)
        assert ps.get_shim("open_responses") is neutral
        assert ps.get_shim("openai_responses") is profile
        assert ps.get_shim("openai--open_responses") is None
        from llm_rosetta.contracts.resolver import resolve_format

        assert resolve_format("openai_responses") == "open_responses"
    finally:
        ps._SHIM_REGISTRY.clear()
        ps._SHIM_REGISTRY.update(saved)


# ---------------------------------------------------------------------------
# The declared surface covers every use (#919 decision 4, #1630)
# ---------------------------------------------------------------------------


_MODULE = "llm_rosetta.retired_names"
#: Packages whose attribute ``retired_names`` is the module (``llm_rosetta``
#: itself and ``llm_rosetta.api``, which re-exports the module object).
_PARENTS = ("llm_rosetta", "llm_rosetta.api")


def _module_bindings(tree) -> tuple[set[str], set[str]]:
    """Local names bound to the module, and names imported out of it.

    Covers ``from llm_rosetta import retired_names [as x]`` (also from
    ``llm_rosetta.api`` and the relative ``from . import retired_names``),
    ``import llm_rosetta.retired_names as x``, and
    ``from llm_rosetta.retired_names import NAME`` (``NAME`` is a read).
    """
    import ast  # noqa: PLC0415

    bound: set[str] = set()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == _MODULE or (node.level and module == "retired_names"):
                imported.update(alias.name for alias in node.names)
            elif module in _PARENTS or (node.level and not module):
                bound.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "retired_names"
                )
        elif isinstance(node, ast.Import):
            bound.update(
                alias.asname
                for alias in node.names
                if alias.name == _MODULE and alias.asname
            )
    return bound, imported


def _is_module(node, bound: set[str]) -> bool:
    """Whether *node* evaluates to the module: a bound name or ``<x>.retired_names``."""
    import ast  # noqa: PLC0415

    if isinstance(node, ast.Name):
        return node.id in bound
    return isinstance(node, ast.Attribute) and node.attr == "retired_names"


def _reads(source: str) -> set[str]:
    """The ``retired_names`` attributes one source file reads.

    A read is ``<m>.NAME`` or ``getattr(<m>, "NAME")`` where ``<m>`` is a
    name bound to the module (any ``as`` rename included) or any expression
    ending in ``.retired_names`` (``llm_rosetta.retired_names``,
    ``api.retired_names``, ``__import__("llm_rosetta").retired_names``), plus
    every name of ``from llm_rosetta.retired_names import NAME``.  Not
    covered: a module object passed through a variable or a container, and
    ``getattr`` with a non-literal name; those are not statically decidable.
    """
    import ast  # noqa: PLC0415

    tree = ast.parse(source)
    bound, used = _module_bindings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _is_module(node.value, bound):
            used.add(node.attr)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and _is_module(node.args[0], bound)
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            used.add(node.args[1].value)
    return used


def _attribute_uses() -> set[str]:
    """Every ``retired_names`` attribute read in the two distributions' sources."""
    from pathlib import Path  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1] / "packages"
    used: set[str] = set()
    for path in root.glob("*/src/**/*.py"):
        if "_vendor" in path.parts:
            continue
        used |= _reads(path.read_text(encoding="utf-8"))
    return used


@pytest.mark.parametrize(
    "source",
    [
        "from llm_rosetta import retired_names\nretired_names.X\n",
        "from llm_rosetta import retired_names as _rn\n_rn.X\n",
        "from llm_rosetta.api import retired_names as rn\nrn.X\n",
        "from . import retired_names as rn\nrn.X\n",
        "import llm_rosetta.retired_names as rn\nrn.X\n",
        "import llm_rosetta.retired_names\nllm_rosetta.retired_names.X\n",
        "from llm_rosetta import api\napi.retired_names.X\n",
        "__import__('llm_rosetta').retired_names.X\n",
        "from llm_rosetta import retired_names as rn\ngetattr(rn, 'X')\n",
        "from llm_rosetta.retired_names import X\n",
    ],
)
def test_the_read_scan_resolves_bindings(source: str) -> None:
    assert "X" in _reads(source)


def test_the_read_scan_ignores_other_modules() -> None:
    assert _reads("from llm_rosetta import formats as rn\nrn.X\n") == set()


def test_every_attribute_core_and_the_gateway_read_is_declared() -> None:
    used = _attribute_uses()
    assert used, "the scan found no use; it is broken"
    assert sorted(used - set(retired_names.__all__)) == []


def test_the_declared_names_all_exist() -> None:
    assert all(hasattr(retired_names, name) for name in retired_names.__all__)


# ---------------------------------------------------------------------------
# Vendors that moved to contrib (#1660)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "vendor"),
    [
        ("opencode--chat_completions", "opencode"),
        ("openai_chatgpt--open_responses", "openai_chatgpt"),
        ("xai", "xai"),
        ("zhipu--rerank", "zhipu"),
        ("opencode--a_lane_contrib_adds_later", "opencode"),
        ("openrouter--openai_chat", "openrouter"),
    ],
)
def test_a_moved_shim_is_traced_to_its_vendor(name: str, vendor: str) -> None:
    row = retired_names.moved_vendor(name)
    assert row is not None and row.vendor == vendor
    assert row.package == "llm-rosetta-ext-contrib"


@pytest.mark.parametrize(
    "name",
    [
        "openai--chat_completions",
        "openai",
        "xai_like--x",
        "zhipuish",
        "",
        # MiniMax stays in core as the minimax embeddings format's vendor.
        "minimax--chat_completions",
        "minimax--embedding",
    ],
)
def test_a_core_or_unknown_name_is_not_moved(name: str) -> None:
    assert retired_names.moved_vendor(name) is None


def test_the_codex_probe_is_traced_to_the_plan_vendor() -> None:
    row = retired_names.moved_probe_vendor("openai_codex")
    assert row is not None and row.vendor == "openai_chatgpt"
    assert retired_names.moved_probe_vendor("deepseek") is None


def test_no_moved_shim_name_is_also_retired() -> None:
    """A moved name is current: renaming it would break the configs it serves."""
    for row in retired_names.MOVED_VENDORS:
        for name in row.shims:
            assert row.owns(name), name
            assert lookup(name, SHIM) is None, name


def test_a_library_lookup_of_a_moved_shim_names_the_contrib_repository() -> None:
    with pytest.raises(ValueError) as caught:
        get_converter_for_provider("opencode--chat_completions")
    message = str(caught.value)
    assert "Unsupported provider: opencode--chat_completions" in message
    assert (
        "moved in 0.14 to the extension repository llm-rosetta-ext-contrib" in message
    )
    assert "llm-rosetta ext install opencode" in message
    assert "llm_rosetta.extensions.load_extensions()" in message


def test_a_pipeline_with_a_moved_target_shim_names_the_contrib_repository() -> None:
    with pytest.raises(ValueError, match="llm-rosetta-ext-contrib"):
        ConversionPipeline("chat_completions", "chat_completions", target_shim="xai")


def test_an_installed_moved_shim_works_from_the_library(moved_vendors) -> None:
    ConversionPipeline("chat_completions", "chat_completions", target_shim="xai")
    assert get_converter_for_provider("opencode--messages") is not None


def test_a_retired_spelling_of_a_moved_shim_says_both_things() -> None:
    message = retired_name_message("openrouter--openai_chat", SHIM)
    assert message is not None
    assert "'openrouter--chat_completions'" in message
    assert "moved to llm-rosetta-ext-contrib" in message
