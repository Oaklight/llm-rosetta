"""
LLM-Rosetta - Anthropic Content Operations

Anthropic Messages API content conversion operations.
Handles bidirectional conversion of text, image, file, reasoning,
and other content parts.

Self-contained: does not depend on utils/FieldMapper.
"""

import warnings
from typing import Any

from ...types.ir import (
    AudioPart,
    CitationPart,
    FilePart,
    ImagePart,
    ReasoningPart,
    RefusalPart,
    TextPart,
)
from ..base import BaseContentOps


class AnthropicContentOps(BaseContentOps):
    """Anthropic Messages API content conversion operations.

    All methods are static and stateless. Handles TextPart, ImagePart,
    FilePart, ReasoningPart bidirectional conversion.
    Audio raises NotImplementedError.
    """

    # ==================== Text ====================

    @staticmethod
    def ir_text_to_p(ir_text: TextPart, **kwargs: Any) -> dict:
        """IR TextPart → Anthropic text content block.

        Args:
            ir_text: IR text part.

        Returns:
            Anthropic text content dict: ``{"type": "text", "text": "..."}``
        """
        return {"type": "text", "text": ir_text["text"]}

    @staticmethod
    def p_text_to_ir(provider_text: Any, **kwargs: Any) -> TextPart:
        """Anthropic text content → IR TextPart.

        Supports both string and dict input formats.

        Args:
            provider_text: Either a plain string or ``{"type": "text", "text": "..."}``.

        Returns:
            IR TextPart.

        Raises:
            ValueError: If input cannot be converted to TextPart.
        """
        if isinstance(provider_text, str):
            return TextPart(type="text", text=provider_text)
        if isinstance(provider_text, dict) and provider_text.get("type") == "text":
            return TextPart(type="text", text=provider_text["text"])
        raise ValueError(f"Cannot convert to TextPart: {provider_text!r}")

    # ==================== Image ====================

    @staticmethod
    def ir_image_to_p(ir_image: ImagePart, **kwargs: Any) -> dict:
        """IR ImagePart → Anthropic image content block.

        Handles both URL and base64 image data.

        Args:
            ir_image: IR image part with ``image_url`` or ``image_data``.

        Returns:
            Anthropic image content dict.

        Raises:
            ValueError: If neither ``image_url`` nor ``image_data`` is present.
        """
        image_data = ir_image.get("image_data")
        if image_data:
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_data["media_type"],
                    "data": image_data["data"],
                },
            }

        image_url = ir_image.get("image_url")
        if image_url:
            return {
                "type": "image",
                "source": {"type": "url", "url": image_url},
            }

        raise ValueError("ImagePart must have either image_url or image_data")

    @staticmethod
    def p_image_to_ir(provider_image: Any, **kwargs: Any) -> ImagePart:
        """Anthropic image content → IR ImagePart.

        Args:
            provider_image: Anthropic image content dict with ``source``.

        Returns:
            IR ImagePart.
        """
        source = provider_image.get("source", {})
        if source.get("type") == "base64":
            return ImagePart(
                type="image",
                image_data={
                    "data": source.get("data", ""),
                    "media_type": source.get("media_type", ""),
                },
            )
        elif source.get("type") == "url":
            return ImagePart(type="image", image_url=source.get("url", ""))
        return ImagePart(type="image")

    # ==================== File ====================

    @staticmethod
    def ir_file_to_p(ir_file: FilePart, **kwargs: Any) -> dict:
        """IR FilePart → Anthropic document content block.

        Anthropic uses ``document`` type for file content.

        Args:
            ir_file: IR file part with ``file_data`` or ``file_url``.

        Returns:
            Anthropic document content dict.

        Raises:
            ValueError: If neither ``file_data`` nor ``file_url`` is present.
        """
        file_data = ir_file.get("file_data")
        if file_data:
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": file_data["media_type"],
                    "data": file_data["data"],
                },
            }

        file_url = ir_file.get("file_url")
        if file_url:
            return {
                "type": "document",
                "source": {"type": "url", "url": file_url},
            }

        raise ValueError("FilePart must have either file_data or file_url")

    @staticmethod
    def p_file_to_ir(provider_file: Any, **kwargs: Any) -> FilePart:
        """Anthropic document content → IR FilePart.

        Args:
            provider_file: Anthropic document content dict with ``source``.

        Returns:
            IR FilePart.
        """
        source = provider_file.get("source", {})
        if source.get("type") == "base64":
            return FilePart(
                type="file",
                file_data={
                    "data": source["data"],
                    "media_type": source["media_type"],
                },
            )
        elif source.get("type") == "url":
            return FilePart(type="file", file_url=source["url"])
        return FilePart(type="file")

    # ==================== Audio (not supported) ====================

    @staticmethod
    def ir_audio_to_p(ir_audio: AudioPart, **kwargs: Any) -> Any:
        """IR AudioPart → Anthropic audio content.

        Raises:
            NotImplementedError: Anthropic does not support audio content parts.
        """
        raise NotImplementedError(
            "Anthropic Messages API does not support audio content parts."
        )

    @staticmethod
    def p_audio_to_ir(provider_audio: Any, **kwargs: Any) -> AudioPart:
        """Anthropic audio content → IR AudioPart.

        Raises:
            NotImplementedError: Anthropic does not support audio content parts.
        """
        raise NotImplementedError(
            "Anthropic Messages API does not support audio content parts."
        )

    # ==================== Reasoning ====================

    @staticmethod
    def ir_reasoning_to_p(ir_reasoning: ReasoningPart, **kwargs: Any) -> dict:
        """IR ReasoningPart → Anthropic thinking content block.

        Anthropic uses ``thinking`` type with ``thinking`` field for reasoning.

        Args:
            ir_reasoning: IR reasoning part.

        Returns:
            Anthropic thinking content dict.
        """
        result: dict = {
            "type": "thinking",
            "thinking": ir_reasoning.get("reasoning", ""),
        }

        signature = ir_reasoning.get("signature")
        if signature:
            result["signature"] = signature

        return result

    @staticmethod
    def p_reasoning_to_ir(provider_reasoning: Any, **kwargs: Any) -> ReasoningPart:
        """Anthropic thinking content → IR ReasoningPart.

        Args:
            provider_reasoning: Anthropic thinking content dict.

        Returns:
            IR ReasoningPart.
        """
        result = ReasoningPart(
            type="reasoning",
            reasoning=provider_reasoning.get("thinking", ""),
        )

        signature = provider_reasoning.get("signature")
        if signature:
            result["signature"] = signature

        return result

    # ==================== Refusal ====================

    @staticmethod
    def ir_refusal_to_p(ir_refusal: RefusalPart, **kwargs: Any) -> dict | None:
        """IR RefusalPart → Anthropic refusal content.

        Anthropic does not have a dedicated refusal content type.
        Converts to a text block with the refusal message.

        Args:
            ir_refusal: IR refusal part.

        Returns:
            Anthropic text content dict with refusal text.
        """
        warnings.warn(
            "Anthropic does not have a dedicated refusal type, "
            "converting to text block",
            stacklevel=2,
        )
        return {"type": "text", "text": f"[Refusal] {ir_refusal['refusal']}"}

    @staticmethod
    def p_refusal_to_ir(provider_refusal: Any, **kwargs: Any) -> RefusalPart:
        """Anthropic refusal content → IR RefusalPart.

        Anthropic does not produce dedicated refusal blocks.
        This is provided for completeness.

        Args:
            provider_refusal: Refusal text string.

        Returns:
            IR RefusalPart.
        """
        if isinstance(provider_refusal, str):
            return RefusalPart(type="refusal", refusal=provider_refusal)
        return RefusalPart(type="refusal", refusal=str(provider_refusal))

    # ==================== Citation ====================

    @staticmethod
    def ir_citation_to_p(ir_citation: CitationPart, **kwargs: Any) -> dict | None:
        """IR CitationPart → Anthropic citation content.

        Anthropic citations are part of TextBlock (as ``citations`` field).
        This method returns None and emits a warning since citations
        cannot be standalone content blocks in Anthropic.

        Args:
            ir_citation: IR citation part.

        Returns:
            None (citations are not standalone blocks in Anthropic).
        """
        warnings.warn(
            "Anthropic citations are part of TextBlock, "
            "cannot be converted to standalone content block",
            stacklevel=2,
        )
        return None

    @staticmethod
    def p_citation_to_ir(provider_citation: Any, **kwargs: Any) -> CitationPart:
        """Anthropic citation → IR CitationPart.

        Handles Anthropic's text citation format.

        Args:
            provider_citation: Anthropic citation dict.

        Returns:
            IR CitationPart.
        """
        citation_type = provider_citation.get("type", "")

        if citation_type == "char_location":
            return CitationPart(
                type="citation",
                text_citation={
                    "cited_text": provider_citation.get("cited_text", ""),
                },
            )

        # URL citation
        if citation_type == "url_citation":
            return CitationPart(
                type="citation",
                url_citation={
                    "start_index": provider_citation.get("start_index", 0),
                    "end_index": provider_citation.get("end_index", 0),
                    "title": provider_citation.get("title", ""),
                    "url": provider_citation.get("url", ""),
                },
            )

        # Fallback
        return CitationPart(type="citation")
