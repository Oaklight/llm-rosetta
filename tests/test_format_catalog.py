"""The format catalog: invariants, and the derived surfaces it feeds.

The catalog is the single home for per-format facts.  These tests pin the
invariants that keep the derived tables honest, and the "fake format" test
below proves that adding one record is enough for a format to reach every
derived surface.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from llm_rosetta.formats import BY_PAIR, FORMAT_SPECS, MODEL_TYPES
import llm_rosetta.formats.detect as detect_mod

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pairs_are_unique():
    pairs = [(spec.model_type, spec.format) for spec in FORMAT_SPECS]
    assert len(pairs) == len(set(pairs))
    assert set(pairs) == set(BY_PAIR)


def test_model_types_cover_the_catalog():
    assert set(MODEL_TYPES) == {spec.model_type for spec in FORMAT_SPECS}
    assert {spec.model_type for spec in FORMAT_SPECS} <= set(MODEL_TYPES)


def test_every_detector_key_resolves():
    for spec in FORMAT_SPECS:
        if spec.detect is None:
            continue
        assert spec.detect in detect_mod.DETECTORS, spec.format


def test_llm_specs_carry_a_route_and_an_sse_key():
    from llm_rosetta_gateway.transport.sse_format import SSE_FORMATTERS

    for spec in FORMAT_SPECS:
        if spec.model_type != "llm":
            continue
        assert spec.route_path, spec.format
        assert spec.sse in SSE_FORMATTERS, spec.format


def test_detection_order_follows_priorities():
    """Records are listed for display; detection runs them by priority."""
    order = [(priority, name) for priority, name, _ in detect_mod._ordered_detectors()]
    assert order[0] == (detect_mod._DECISION_PRIORITY, "decision")
    assert [priority for priority, _ in order] == sorted(p for p, _ in order)
    assert {name for _, name in order[1:]} == set(detect_mod.DETECTORS)

    declared = {
        spec.detect: spec.detect_priority
        for spec in FORMAT_SPECS
        if spec.detect is not None
    }
    assert {name: priority for priority, name in order[1:]} == declared


def test_gateway_tables_cover_the_llm_catalog():
    from llm_rosetta_gateway.providers import (
        BASE_FORMATS,
        _BASE_TYPE_HINTS,
        _CUSTOM_TOOLS_BY_TYPE,
        _PROVIDER_REGISTRY,
    )

    for spec in FORMAT_SPECS:
        if spec.model_type != "llm":
            continue
        if spec.url_template is not None:
            assert spec.format in _PROVIDER_REGISTRY, spec.format
            assert spec.format in BASE_FORMATS, spec.format
        if spec.base_url_hint and spec.api_key_env_hint:
            assert spec.format in _BASE_TYPE_HINTS, spec.format
        assert spec.format in _CUSTOM_TOOLS_BY_TYPE, spec.format


FAKE_FORMAT_SCRIPT = """
import importlib

# Imported first, on a pristine catalog, so the registry's catalog/converter
# consistency check runs against the real thing.
import llm_rosetta.converters.registry as registry
import llm_rosetta.formats as formats
import llm_rosetta.formats.detect as detect
from llm_rosetta.formats import catalog, spec

fake = spec.FormatSpec(
    model_type="llm",
    format="fake_wire",
    wire_family="anthropic",
    detect="fake_wire",
    detect_priority=25,
    picker_order=25,
    auth="anthropic",
    url_template="{base_url}/fake",
    route_path="/v1/fake_wire",
    sse="chat_completions",
    usage_shape="openai_details",
    models_path="/fake/models",
)

# A second fake stands in for a Google-family route, so the error-envelope
# derivation is pinned too: the family comes from the catalog's ``wire_family``
# and the path from its ``route_path``.
fake_google = spec.FormatSpec(
    model_type="llm",
    format="fake_google_wire",
    wire_family="google",
    route_path="/v1beta/fake_google",
)

patched = formats.FORMAT_SPECS + (fake, fake_google)
catalog.FORMAT_SPECS = patched
formats.FORMAT_SPECS = patched
catalog.BY_PAIR[("llm", "fake_wire")] = fake
catalog.BY_PAIR[("llm", "fake_google_wire")] = fake_google
detect.DETECTORS["fake_wire"] = lambda body: "fake_marker" in body


# A format's converter class is the one thing it must bring itself; bind a
# placeholder the way a real format binds its converter.
class FakeConverter:
    pass


registry._CONVERTER_CLASSES["llm"]["fake_wire"] = FakeConverter
registry.CONVERTER_REGISTRY["llm"]["fake_wire"] = FakeConverter
registry.FORMATS_BY_TYPE["llm"] = registry.FORMATS_BY_TYPE["llm"] | {"fake_wire"}

# Reload every module that derives a table from the catalog at import time.
import llm_rosetta.model_vocabulary as vocabulary

importlib.reload(vocabulary)
assert "fake_wire" in vocabulary.FORMATS_BY_TYPE["llm"], "format vocabulary"

import llm_rosetta_gateway.providers as providers

importlib.reload(providers)
assert (
    providers._PROVIDER_REGISTRY["fake_wire"]["url_template"] == "{base_url}/fake"
), "provider registry"
assert providers.BASE_FORMATS.index("fake_wire") == (
    providers.BASE_FORMATS.index("open_responses") + 1
), "picker order"
assert providers.default_models_path("fake_wire") == "/fake/models", "models path"

import llm_rosetta_gateway.transport.sse_format as sse_format

importlib.reload(sse_format)
assert "fake_wire" in sse_format.SSE_FORMATTERS, "SSE formatter"

import json

import llm_rosetta_gateway.middleware.error_format as error_format

importlib.reload(error_format)
assert error_format._PROVIDER_FORMAT_MAP["fake_wire"] == "anthropic", "error family"
assert error_format._PROVIDER_FORMAT_MAP["fake_google_wire"] == "google"

# The route -> envelope family and the envelope itself both derive from the
# catalog record; this is the path the `/v1beta/interactions` change rides on.
assert error_format.detect_api_format("/v1/fake_wire") == "anthropic", "route family"
assert error_format.detect_api_format("/v1beta/fake_google") == "google", "route family"
assert json.loads(
    error_format.format_error_response(
        error_format.detect_api_format("/v1beta/fake_google"), 400, "boom"
    ).body
) == {
    "error": {"code": 400, "message": "boom", "status": "INVALID_ARGUMENT"}
}, "google envelope"
assert json.loads(
    error_format.format_error_response(
        error_format.detect_api_format("/v1/fake_wire"), 400, "boom"
    ).body
) == {"type": "error", "error": {"type": "invalid_request_error", "message": "boom"}}, (
    "anthropic envelope"
)
# The disclosed change: the interactions route resolves to the Google
# envelope because its catalog record says ``wire_family="google"``.
assert error_format.detect_api_format("/v1beta/interactions") == "google"

assert error_format.detect_api_format("/v1/chat/completions") == "openai"
assert json.loads(error_format.format_error_response("openai", 400, "boom").body) == {
    "error": {"message": "boom", "type": "invalid_request_error", "code": None}
}, "openai envelope"

import llm_rosetta.observability.usage.semantics as semantics

importlib.reload(semantics)
assert semantics.shape_for("fake_wire") is semantics.OPENAI_DETAILS, "usage shape"

from llm_rosetta.auto_detect import detect_provider

assert detect_provider({"fake_marker": 1}) == "fake_wire", "auto-detection"
assert detect_provider({"messages": [{"role": "user"}]}) == "chat_completions"

from llm_rosetta_gateway.model_types import get_model_type

llm = get_model_type("llm")
assert "fake_wire" in llm.formats, "admin payload"
routes = [(route.path, route.source_format) for route in llm.routes]
assert ("/v1/fake_wire", "fake_wire") in routes, "route registration"

print("OK")
"""


def test_fake_format_reaches_every_derived_surface():
    """One catalog record is enough; nothing outside ``formats/`` is edited."""
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(FAKE_FORMAT_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert proc.stdout.strip().endswith("OK")
