"""
LLM-Rosetta - OpenAI Chat Content Operations

OpenAI Chat Completions API content conversion operations.
Handles bidirectional conversion of text, image, and other content parts.
"""

import re
import warnings
from typing import Any, Optional

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


class OpenAIChatContentOps(BaseContentOps):
    """OpenAI Chat Completions content conversion operations.

    All methods are static and stateless. Handles TextPart, ImagePart
    bidirectional conversion. File/Audio raise NotImplementedError.
    """

    # ==================== Text ====================

    @staticmethod
    def ir_text_to_p(ir_text: TextPart, **kwargs: Any) -> dict:
        """IR TextPart → OpenAI text content part.

        Args:
            ir_text: IR text part.

        Returns:
            OpenAI text content dict: ``{"type": "text", "text": "..."}``
        """
        return {"type": "text", "text": ir_text["text"]}

    @staticmethod
    def p_text_to_ir(provider_text: Any, **kwargs: Any) -> TextPart:
        """OpenAI text content → IR TextPart.

        Supports both string and dict input formats.

        Args:
            provider_text: Either a plain string or ``{"type": "text", "text": "..."}``.

        Returns:
            IR TextPart.
        """
        if isinstance(provider_text, str):
            return TextPart(type="text", text=provider_text)
        if isinstance(provider_text, dict) and provider_text.get("type") == "text":
            return TextPart(type="text", text=provider_text["text"])
        raise ValueError(f"Cannot convert to TextPart: {provider_text!r}")

    # ==================== Image ====================

    @staticmethod
    def ir_image_to_p(ir_image: ImagePart, **kwargs: Any) -> dict:
        """IR ImagePart → OpenAI image_url content part.

        Handles both URL and base64 image data. Base64 data is converted
        to a data URI.

        Args:
            ir_image: IR image part with ``image_url`` or ``image_data``.

        Returns:
            OpenAI image content dict.

        Raises:
            ValueError: If neither ``image_url`` nor ``image_data`` is present.
        """
        detail = ir_image.get("detail", "auto")
        url = ir_image.get("image_url")

        if url:
            return {"type": "image_url", "image_url": {"url": url, "detail": detail}}

        image_data = ir_image.get("image_data")
        if image_data:
            data_url = f"data:{image_data['media_type']};base64,{image_data['data']}"
            return {
                "type": "image_url",
                "image_url": {"url": data_url, "detail": detail},
            }

        raise ValueError("ImagePart must have either image_url or image_data")

    @staticmethod
    def p_image_to_ir(provider_image: Any, **kwargs: Any) -> ImagePart:
        """OpenAI image_url content → IR ImagePart.

        Parses data URIs back into ``image_data`` with media_type.

        Args:
            provider_image: OpenAI image content dict with ``image_url``.

        Returns:
            IR ImagePart.
        """
        image_url_data = provider_image.get("image_url", {})
        url = image_url_data.get("url", "")
        detail = image_url_data.get("detail", "auto")

        if url and url.startswith("data:"):
            match = re.match(r"data:([^;]+);base64,(.+)", url)
            if match:
                media_type, data = match.groups()
                return ImagePart(
                    type="image",
                    image_data={"data": data, "media_type": media_type},
                    detail=detail,
                )

        return ImagePart(type="image", image_url=url, detail=detail)

    # ==================== File (not supported) ====================

    @staticmethod
    def ir_file_to_p(ir_file: FilePart, **kwargs: Any) -> Any:
        """IR FilePart → OpenAI file content.

        Raises:
            NotImplementedError: OpenAI Chat Completions does not support file input.
        """
        raise NotImplementedError(
            "OpenAI Chat Completions does not support file input. "
            "Use OpenAI Responses API converter for file support."
        )

    @staticmethod
    def p_file_to_ir(provider_file: Any, **kwargs: Any) -> FilePart:
        """OpenAI file content → IR FilePart.

        Raises:
            NotImplementedError: OpenAI Chat Completions does not support file input.
        """
        raise NotImplementedError(
            "OpenAI Chat Completions does not support file input. "
            "Use OpenAI Responses API converter for file support."
        )

    # ==================== Audio (not supported) ====================

    @staticmethod
    def ir_audio_to_p(ir_audio: AudioPart, **kwargs: Any) -> Any:
        """IR AudioPart → OpenAI audio content.

        Raises:
            NotImplementedError: OpenAI Chat Completions does not support audio parts.
        """
        raise NotImplementedError(
            "OpenAI Chat Completions does not support audio content parts."
        )

    @staticmethod
    def p_audio_to_ir(provider_audio: Any, **kwargs: Any) -> AudioPart:
        """OpenAI audio content → IR AudioPart.

        Raises:
            NotImplementedError: OpenAI Chat Completions does not support audio parts.
        """
        raise NotImplementedError(
            "OpenAI Chat Completions does not support audio content parts."
        )

    # ==================== Reasoning ====================

    @staticmethod
    def ir_reasoning_to_p(ir_reasoning: ReasoningPart, **kwargs: Any) -> Optional[dict]:
        """IR ReasoningPart → OpenAI reasoning content.

        OpenAI Chat Completions does not support reasoning content in requests.
        Returns None and emits a warning.

        Args:
            ir_reasoning: IR reasoning part.

        Returns:
            None (reasoning is dropped with a warning).
        """
        warnings.warn(
            "Reasoning content not supported in OpenAI Chat Completions, ignored",
            stacklevel=2,
        )
        return None

    @staticmethod
    def p_reasoning_to_ir(provider_reasoning: Any, **kwargs: Any) -> ReasoningPart:
        """OpenAI reasoning content → IR ReasoningPart.

        Raises:
            NotImplementedError: OpenAI Chat Completions does not produce reasoning parts.
        """
        raise NotImplementedError(
            "OpenAI Chat Completions does not produce reasoning content parts."
        )

    # ==================== Refusal ====================

    @staticmethod
    def ir_refusal_to_p(ir_refusal: RefusalPart, **kwargs: Any) -> dict:
        """IR RefusalPart → OpenAI refusal field value.

        Returns the refusal text string. The caller (message_ops) is responsible
        for placing it in the ``refusal`` field of the assistant message.

        Args:
            ir_refusal: IR refusal part.

        Returns:
            Dict with refusal text for embedding in assistant message.
        """
        return {"refusal": ir_refusal["refusal"]}

    @staticmethod
    def p_refusal_to_ir(provider_refusal: Any, **kwargs: Any) -> RefusalPart:
        """OpenAI refusal field → IR RefusalPart.

        Args:
            provider_refusal: Refusal string from OpenAI assistant message.

        Returns:
            IR RefusalPart.
        """
        if isinstance(provider_refusal, str):
            return RefusalPart(type="refusal", refusal=provider_refusal)
        return RefusalPart(type="refusal", refusal=str(provider_refusal))

    # ==================== Citation ====================

    @staticmethod
    def ir_citation_to_p(ir_citation: CitationPart, **kwargs: Any) -> Optional[dict]:
        """IR CitationPart → OpenAI annotation.

        Maps URL citations to OpenAI annotation format.

        Args:
            ir_citation: IR citation part.

        Returns:
            OpenAI annotation dict, or None if no URL citation data.
        """
        url_citation = ir_citation.get("url_citation")
        if url_citation:
            return {
                "type": "url_citation",
                "start_index": url_citation.get("start_index", 0),
                "end_index": url_citation.get("end_index", 0),
                "title": url_citation.get("title", ""),
                "url": url_citation.get("url", ""),
            }
        return None

    @staticmethod
    def p_citation_to_ir(provider_citation: Any, **kwargs: Any) -> CitationPart:
        """OpenAI annotation → IR CitationPart.

        Args:
            provider_citation: OpenAI annotation dict.

        Returns:
            IR CitationPart.
        """
        citation_type = provider_citation.get("type", "")
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
        # Fallback: store raw data
        return CitationPart(type="citation")
