"""Provider-level overrides for a lane that names ``transport_options``.

The shim carries the vendor facts; a provider entry carries deployment facts.
The gateway merges the two per key and checks only that the entry's value is
an object: it knows no transport's option names, and the transport validates
the content.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.transport.acp.lane import _transport_options


def _shim(options) -> SimpleNamespace:
    return SimpleNamespace(connection=SimpleNamespace(transport_options=options))


class TestTheOptionsOverlay:
    def test_the_shim_supplies_the_default(self):
        shim = _shim({"argv": ["claude"], "args": ["--x"]})

        assert _transport_options({}, shim, "claude") == {
            "argv": ["claude"],
            "args": ["--x"],
        }

    def test_the_entry_wins_per_key_and_keeps_the_other_shim_keys(self):
        shim = _shim({"argv": ["claude"], "upstream": "https://shim.example"})
        cfg = {"transport_options": {"argv": ["/opt/claude"]}}

        assert _transport_options(cfg, shim, "claude") == {
            "argv": ["/opt/claude"],
            "upstream": "https://shim.example",
        }

    def test_a_nested_value_is_replaced_whole(self):
        shim = _shim({"env": {"A": "1", "B": "2"}})
        cfg = {"transport_options": {"env": {"A": "9"}}}

        assert _transport_options(cfg, shim, "claude") == {"env": {"A": "9"}}

    def test_neither_side_declares_any(self):
        assert _transport_options({}, _shim(None), "claude") == {}
        assert _transport_options({}, None, "claude") == {}

    def test_the_result_does_not_alias_the_shim(self):
        shim = _shim({"env": {"A": "1"}})

        merged = _transport_options({"transport_options": {"x": 1}}, shim, "claude")
        merged["x"] = 2

        assert shim.connection.transport_options == {"env": {"A": "1"}}

    def test_the_lane_home_is_expanded_in_nested_strings(self):
        shim = _shim(
            {"env": {"DIR": "{acp_home}/claude"}, "args": ["--d", "{acp_state_dir}"]}
        )

        assert _transport_options({}, shim, "claude", "/data/acp/c") == {
            "env": {"DIR": "/data/acp/c/claude"},
            "args": ["--d", "/data/acp/c"],
        }

    def test_option_content_is_not_validated_by_the_gateway(self):
        cfg = {"transport_options": {"argv": "not-a-list", "anything": 7}}

        assert _transport_options(cfg, None, "claude") == cfg["transport_options"]

    @pytest.mark.parametrize("bad", ["x", ["argv"], 7, True])
    def test_a_non_object_entry_value_is_refused(self, bad):
        with pytest.raises(ValueError, match="transport_options must be an object"):
            _transport_options({"transport_options": bad}, _shim({}), "claude")
