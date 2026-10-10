"""Open Responses / OpenAI Responses converter hierarchy and shim wiring.

Verifies the spec-correct inversion introduced for issue #493: the
vendor-neutral Open Responses spec is the base converter, and the OpenAI
Responses API is a derived profile, with the ``OpenAIResponses*`` names kept
as backward-compatible aliases.
"""

from __future__ import annotations

from llm_rosetta.auto_detect import get_converter_for_provider
from llm_rosetta.converters.openai_responses import (
    OpenAIResponsesConfigOps,
    OpenAIResponsesContentOps,
    OpenAIResponsesConverter,
    OpenAIResponsesMessageOps,
    OpenAIResponsesStreamContext,
    OpenAIResponsesToolOps,
    OpenResponsesConfigOps,
    OpenResponsesContentOps,
    OpenResponsesConverter,
    OpenResponsesMessageOps,
    OpenResponsesStreamContext,
    OpenResponsesToolOps,
)
from llm_rosetta.shims import get_shim, resolve_base


class TestHierarchy:
    def test_openai_profile_derives_from_spec_base(self):
        assert issubclass(OpenAIResponsesConverter, OpenResponsesConverter)
        assert OpenAIResponsesConverter is not OpenResponsesConverter

    def test_converter_tags_and_prefixes(self):
        assert OpenResponsesConverter._CONVERTER_TAG == "open_responses"
        assert OpenResponsesConverter._RESPONSE_ID_PREFIX == ""
        assert OpenAIResponsesConverter._CONVERTER_TAG == "openai_responses"
        assert OpenAIResponsesConverter._RESPONSE_ID_PREFIX == "resp_"

    def test_store_default_differs(self):
        """The spec is stateless; only the OpenAI profile forces ``store``."""
        assert "store" not in OpenResponsesConverter._REQUIRED_DEFAULTS
        assert OpenAIResponsesConverter._REQUIRED_DEFAULTS["store"] is True

    def test_preserve_fields_differs(self):
        """OpenAI lifecycle fields are not part of the vendor-neutral spec."""
        assert "billing" in OpenAIResponsesConverter._PRESERVE_FIELDS
        assert "billing" not in OpenResponsesConverter._PRESERVE_FIELDS
        assert "store" in OpenResponsesConverter._PRESERVE_FIELDS

    def test_dispatch_returns_distinct_classes(self):
        assert type(get_converter_for_provider("open_responses")) is (
            OpenResponsesConverter
        )
        assert type(get_converter_for_provider("openai_responses")) is (
            OpenAIResponsesConverter
        )

    def test_ops_aliases_are_identical_objects(self):
        assert OpenAIResponsesContentOps is OpenResponsesContentOps
        assert OpenAIResponsesToolOps is OpenResponsesToolOps
        assert OpenAIResponsesMessageOps is OpenResponsesMessageOps
        assert OpenAIResponsesConfigOps is OpenResponsesConfigOps
        assert OpenAIResponsesStreamContext is OpenResponsesStreamContext


class TestShim:
    def test_shim_is_vendor_neutral(self):
        shim = get_shim("open_responses")
        assert shim is not None
        assert shim.base == "open_responses"
        # No canonical host and no vendor response-id prefix.
        assert shim.connection.base_url is None
        assert shim.response_id_prefix == ""
        assert shim.logo is not None and shim.logo.startswith("https://")

    def test_resolve_base_is_identity(self):
        assert resolve_base("open_responses") == "open_responses"
