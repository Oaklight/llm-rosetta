"""Tests for image truncation utility."""

import pytest
from llm_rosetta.converters.base.image_limit import truncate_images


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
