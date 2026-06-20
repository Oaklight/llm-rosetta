"""Tests for image truncation utility."""

from llm_rosetta.converters.base.helpers.image_limit import truncate_images


def _make_request(image_counts_per_message: list[int]) -> dict:
    """Build a minimal IR request with the given image distribution."""
    messages = []
    for count in image_counts_per_message:
        content = []
        for i in range(count):
            content.append(
                {"type": "image", "image_url": f"https://example.com/img{i}.png"}
            )
        messages.append({"role": "user", "content": content})
    return {"messages": messages}


def _count_images(ir_request: dict) -> int:
    return sum(
        1
        for msg in ir_request["messages"]
        for part in msg.get("content", [])
        if part.get("type") == "image"
    )


def _count_placeholders(ir_request: dict) -> int:
    return sum(
        1
        for msg in ir_request["messages"]
        for part in msg.get("content", [])
        if part.get("type") == "text" and "image omitted" in part.get("text", "")
    )


class TestTruncateImages:
    def test_no_truncation_needed(self):
        req = _make_request([10, 10, 10])  # 30 images, limit 50
        result = truncate_images(req, 50)
        assert result is req  # same object, no copy
        assert _count_images(result) == 30

    def test_exact_limit(self):
        req = _make_request([25, 25])  # 50 images, limit 50
        result = truncate_images(req, 50)
        assert result is req
        assert _count_images(result) == 50

    def test_truncation_to_limit(self):
        req = _make_request([60])  # 60 images, limit 50
        result = truncate_images(req, 50)
        assert _count_images(result) == 50
        assert _count_placeholders(result) == 10

    def test_keeps_most_recent(self):
        """Oldest images replaced, most recent kept."""
        req = _make_request([3])  # 3 images, limit 2
        result = truncate_images(req, 2)
        content = result["messages"][0]["content"]
        assert content[0]["type"] == "text"  # oldest replaced
        assert content[1]["type"] == "image"  # kept
        assert content[2]["type"] == "image"  # kept

    def test_cross_message_truncation(self):
        """Truncation works across message boundaries."""
        req = _make_request([3, 3])  # 6 images across 2 messages, limit 4
        result = truncate_images(req, 4)
        assert _count_images(result) == 4
        assert _count_placeholders(result) == 2

    def test_placeholder_text_mentions_limit(self):
        req = _make_request([3])
        result = truncate_images(req, 2)
        placeholder = result["messages"][0]["content"][0]
        assert "50" not in placeholder["text"] or "2" in placeholder["text"]
        assert "image omitted" in placeholder["text"]

    def test_original_not_mutated(self):
        req = _make_request([3])
        original_content = req["messages"][0]["content"][:]
        truncate_images(req, 2)
        # Original should be unchanged
        assert req["messages"][0]["content"] == original_content


class TestTruncateToolResultImages:
    """Images embedded in tool_result.result lists must also be counted."""

    def _make_tool_result_request(
        self, direct_images: int, tool_result_images: int
    ) -> dict:
        """Build a request with images in both content and tool results."""
        content: list[dict] = []
        for i in range(direct_images):
            content.append(
                {"type": "image", "image_url": f"https://example.com/img{i}.png"}
            )
        content.append(
            {
                "type": "tool_result",
                "tool_call_id": "call_1",
                "result": [
                    {
                        "type": "image",
                        "image_data": {
                            "data": f"base64data{i}",
                            "media_type": "image/png",
                        },
                    }
                    for i in range(tool_result_images)
                ],
            }
        )
        return {"messages": [{"role": "user", "content": content}]}

    def _total_images(self, req: dict) -> int:
        """Count all images including those in tool results."""
        count = 0
        for msg in req["messages"]:
            for part in msg.get("content", []):
                if part.get("type") in ("image", "image_url"):
                    count += 1
                elif part.get("type") == "tool_result":
                    result = part.get("result")
                    if isinstance(result, list):
                        count += sum(
                            1
                            for b in result
                            if isinstance(b, dict)
                            and b.get("type") in ("image", "image_url")
                        )
        return count

    def test_tool_result_images_counted(self):
        """49 direct + 2 in tool result = 51, should truncate to 50."""
        req = self._make_tool_result_request(49, 2)
        assert self._total_images(req) == 51
        result = truncate_images(req, 50)
        assert result is not req
        assert self._total_images(result) == 50

    def test_tool_result_only_images(self):
        """All images inside tool results, none direct."""
        req = self._make_tool_result_request(0, 55)
        assert self._total_images(req) == 55
        result = truncate_images(req, 50)
        assert self._total_images(result) == 50

    def test_no_truncation_with_tool_results(self):
        """Images in tool results under limit — no truncation."""
        req = self._make_tool_result_request(20, 10)
        assert self._total_images(req) == 30
        result = truncate_images(req, 50)
        assert result is req

    def test_mixed_keeps_most_recent(self):
        """Oldest images replaced first, whether direct or in tool results."""
        # 2 direct images (oldest), then tool result with 2 images, limit 3
        content: list[dict] = [
            {"type": "image", "image_url": "https://example.com/old1.png"},
            {"type": "image", "image_url": "https://example.com/old2.png"},
            {
                "type": "tool_result",
                "tool_call_id": "call_1",
                "result": [
                    {
                        "type": "image",
                        "image_data": {"data": "new1", "media_type": "image/png"},
                    },
                    {
                        "type": "image",
                        "image_data": {"data": "new2", "media_type": "image/png"},
                    },
                ],
            },
        ]
        req = {"messages": [{"role": "user", "content": content}]}
        result = truncate_images(req, 3)
        # Oldest direct image replaced, 3 kept (1 direct + 2 in tool result)
        assert self._total_images(result) == 3

    def test_original_not_mutated_with_tool_results(self):
        req = self._make_tool_result_request(49, 2)
        import copy

        original = copy.deepcopy(req)
        truncate_images(req, 50)
        assert req == original


class TestApplyImageLimitPattern:
    """Tests for model-scoped max_images_pattern logic."""

    def _make_shim(self, max_images: int, pattern: str | None):
        """Build a minimal ProviderShim-like object."""
        from unittest.mock import MagicMock

        shim = MagicMock()
        shim.max_images = max_images
        shim.max_images_pattern = pattern
        return shim

    def _apply(
        self, req: dict, max_images: int, pattern: str | None, model: str
    ) -> dict:
        """Drive truncation logic directly without a real shim registry."""
        import re

        from llm_rosetta.converters.base.helpers.image_limit import truncate_images

        if pattern is not None and not re.search(pattern, model):
            return req
        return truncate_images(req, max_images)

    def test_gpt_model_truncated(self):
        req = _make_request([60])
        result = self._apply(req, 50, r"^(gpt|o\d)", "gpt-4o")
        assert _count_images(result) == 50

    def test_o_model_truncated(self):
        req = _make_request([60])
        result = self._apply(req, 50, r"^(gpt|o\d)", "o3-mini")
        assert _count_images(result) == 50

    def test_gemini_model_not_truncated(self):
        req = _make_request([60])
        result = self._apply(req, 50, r"^(gpt|o\d)", "gemini-2.0-flash")
        assert result is req  # no truncation
        assert _count_images(result) == 60

    def test_claude_model_not_truncated(self):
        req = _make_request([60])
        result = self._apply(req, 50, r"^(gpt|o\d)", "claude-opus-4-5")
        assert result is req
        assert _count_images(result) == 60

    def test_no_pattern_always_truncates(self):
        """When max_images_pattern is None, apply to all models."""
        req = _make_request([60])
        result = self._apply(req, 50, None, "gemini-2.0-flash")
        assert _count_images(result) == 50
