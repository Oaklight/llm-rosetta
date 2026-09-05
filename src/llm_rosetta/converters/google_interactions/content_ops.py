"""
LLM-Rosetta - Google Interactions Content Operations

Bidirectional conversion between Interactions API Content types
(TextContent, ImageContent, etc.) and IR content parts.
"""

from typing import Any

from ...types.ir import (
    CitationPart,
    ImageData,
    ImagePart,
    ReasoningPart,
    TextPart,
)
from ...types.ir.parts import ContentPart
from ..base import BaseContentOps


class GoogleInteractionsContentOps(BaseContentOps):
    """Google Interactions API content conversion operations."""

    # ── Text ───────────────────────────────────────────────────────

    @staticmethod
    def ir_text_to_p(ir_text: TextPart, **kwargs: Any) -> dict:
        """IR TextPart → Interactions TextContent."""
        return {"type": "text", "text": ir_text["text"]}

    @staticmethod
    def p_text_to_ir(provider_text: Any, **kwargs: Any) -> TextPart:
        """Interactions TextContent → IR TextPart."""
        return {"type": "text", "text": provider_text["text"]}

    # ── Image ──────────────────────────────────────────────────────

    @staticmethod
    def ir_image_to_p(ir_image: ImagePart, **kwargs: Any) -> dict:
        """IR ImagePart → Interactions ImageContent."""
        result: dict[str, Any] = {"type": "image"}
        if "image_data" in ir_image:
            data = ir_image["image_data"]
            result["data"] = data["data"]
            result["mime_type"] = data["media_type"]
        elif "image_url" in ir_image:
            result["uri"] = ir_image["image_url"]
        return result

    @staticmethod
    def p_image_to_ir(provider_image: Any, **kwargs: Any) -> ImagePart:
        """Interactions ImageContent → IR ImagePart."""
        result: ImagePart = {"type": "image"}
        if "data" in provider_image:
            result["image_data"] = ImageData(
                data=provider_image["data"],
                media_type=provider_image.get("mime_type", "image/png"),
            )
        elif "uri" in provider_image:
            result["image_url"] = provider_image["uri"]
        return result

    # ── Reasoning (ThoughtStep) ────────────────────────────────────

    @staticmethod
    def p_thought_to_ir(thought_step: dict, **kwargs: Any) -> ReasoningPart:
        """Interactions ThoughtStep → IR ReasoningPart."""
        part: ReasoningPart = {"type": "reasoning"}
        if "signature" in thought_step:
            part["signature"] = thought_step["signature"]
        summary = thought_step.get("summary")
        if summary and isinstance(summary, list):
            texts = [s["text"] for s in summary if isinstance(s, dict) and "text" in s]
            if texts:
                part["reasoning"] = "\n".join(texts)
        return part

    @staticmethod
    def ir_reasoning_to_p(ir_reasoning: ReasoningPart, **kwargs: Any) -> dict:
        """IR ReasoningPart → Interactions ThoughtStep."""
        step: dict[str, Any] = {"type": "thought"}
        if "signature" in ir_reasoning:
            step["signature"] = ir_reasoning["signature"]
        reasoning_text = ir_reasoning.get("reasoning")
        if reasoning_text:
            step["summary"] = [{"type": "text", "text": reasoning_text}]
        return step

    # ── Annotations → Citations ────────────────────────────────────

    @staticmethod
    def p_annotations_to_ir(annotations: list[dict]) -> list[CitationPart]:
        """Interactions TextContent.annotations → IR CitationParts."""
        parts: list[CitationPart] = []
        for ann in annotations:
            part: CitationPart = {"type": "citation"}
            if "url" in ann:
                part["url_citation"] = {
                    "url": ann["url"],
                }
                if "title" in ann:
                    part["url_citation"]["title"] = ann["title"]
                if "start_index" in ann:
                    part["url_citation"]["start_index"] = ann["start_index"]
                if "end_index" in ann:
                    part["url_citation"]["end_index"] = ann["end_index"]
            parts.append(part)
        return parts

    # ── Generic content dispatch ───────────────────────────────────

    @staticmethod
    def p_content_to_ir(content: dict, **kwargs: Any) -> ContentPart | None:
        """Dispatch a single Interactions Content item to IR part."""
        ops = GoogleInteractionsContentOps
        ctype = content.get("type")
        if ctype == "text":
            return ops.p_text_to_ir(content)
        elif ctype == "image":
            return ops.p_image_to_ir(content)
        return None

    # ── Stubs for base class ───────────────────────────────────────

    @staticmethod
    def ir_file_to_p(ir_file: Any, **kwargs: Any) -> Any:
        return {}

    @staticmethod
    def p_file_to_ir(provider_file: Any, **kwargs: Any) -> Any:
        return {"type": "file"}

    @staticmethod
    def ir_audio_to_p(ir_audio: Any, **kwargs: Any) -> Any:
        return {}

    @staticmethod
    def p_audio_to_ir(provider_audio: Any, **kwargs: Any) -> Any:
        return {"type": "audio"}

    @staticmethod
    def ir_reasoning_to_p_part(ir_reasoning: ReasoningPart, **kwargs: Any) -> Any:
        return GoogleInteractionsContentOps.ir_reasoning_to_p(ir_reasoning, **kwargs)

    @staticmethod
    def p_reasoning_to_ir(provider_reasoning: Any, **kwargs: Any) -> ReasoningPart:
        return GoogleInteractionsContentOps.p_thought_to_ir(provider_reasoning)

    @staticmethod
    def ir_refusal_to_p(ir_refusal: Any, **kwargs: Any) -> Any:
        return {}

    @staticmethod
    def p_refusal_to_ir(provider_refusal: Any, **kwargs: Any) -> Any:
        return {"type": "refusal", "refusal": ""}

    @staticmethod
    def ir_citation_to_p(ir_citation: Any, **kwargs: Any) -> Any:
        return {}

    @staticmethod
    def p_citation_to_ir(provider_citation: Any, **kwargs: Any) -> CitationPart:
        return {"type": "citation"}
