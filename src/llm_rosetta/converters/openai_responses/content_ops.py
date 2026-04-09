"""
LLM-Rosetta - OpenAI Responses Content Operations

OpenAI Responses API content conversion operations.
Handles bidirectional conversion of text, image, file, reasoning, and other content parts.

Note: Responses API uses input_text/output_text for text, input_image for images,
and input_file for files. Images and files are always input-only.
"""

import re
import warnings
from typing import Any

from ...types.ir import (
    AudioPart,
    CitationPart,
    FileData,
    FilePart,
    ImagePart,
    ReasoningPart,
    RefusalPart,
    TextPart,
)
from ..base import BaseContentOps


class OpenAIResponsesContentOps(BaseContentOps):
    """OpenAI Responses API content conversion operations.

    All methods are static and stateless. Handles TextPart, ImagePart,
    FilePart, and ReasoningPart bidirectional conversion.
    """

    # ==================== Text ====================

    @staticmethod
    def ir_text_to_p(ir_text: TextPart, **kwargs: Any) -> dict:
        """IR TextPart → OpenAI Responses text content part.

        The ``context`` kwarg determines the text type:
        - ``"input"`` (default) → ``input_text``
        - ``"output"`` → ``output_text``

        Args:
            ir_text: IR text part.
            **kwargs: May contain ``context`` ("input" or "output").

        Returns:
            OpenAI Responses text content dict.
        """
        context = kwargs.get("context", "input")
        text_type = "output_text" if context == "output" else "input_text"
        return {"type": text_type, "text": ir_text["text"]}

    @staticmethod
    def p_text_to_ir(provider_text: Any, **kwargs: Any) -> TextPart:
        """OpenAI Responses text content → IR TextPart.

        Supports string, input_text, output_text, and plain text dict formats.

        Args:
            provider_text: Text content in various formats.

        Returns:
            IR TextPart.
        """
        if isinstance(provider_text, str):
            return TextPart(type="text", text=provider_text)
        if isinstance(provider_text, dict):
            part_type = provider_text.get("type")
            if part_type in ("input_text", "output_text", "text"):
                return TextPart(type="text", text=provider_text["text"])
        raise ValueError(f"Cannot convert to TextPart: {provider_text!r}")

    # ==================== Image ====================

    @staticmethod
    def ir_image_to_p(ir_image: ImagePart, **kwargs: Any) -> dict:
        """IR ImagePart → OpenAI Responses input_image content part.

        Images are always input_image in Responses API.
        Handles both URL and base64 image data.

        Args:
            ir_image: IR image part with ``image_url`` or ``image_data``.

        Returns:
            OpenAI Responses image content dict.

        Raises:
            ValueError: If neither ``image_url`` nor ``image_data`` is present.
        """
        result: dict = {
            "type": "input_image",
            "detail": ir_image.get("detail", "auto"),
        }

        # Support multiple URL field names (image_url, url)
        url = ir_image.get("image_url") or ir_image.get("url")
        image_data = ir_image.get("image_data")

        if url:
            result["image_url"] = url
        elif image_data:
            data_url = f"data:{image_data['media_type']};base64,{image_data['data']}"
            result["image_url"] = data_url
        elif "provider_ref" in ir_image and "file_id" in ir_image["provider_ref"]:
            result["file_id"] = ir_image["provider_ref"]["file_id"]
        else:
            raise ValueError("Image part must have either image_url/url or image_data")

        return result

    @staticmethod
    def p_image_to_ir(provider_image: Any, **kwargs: Any) -> ImagePart:
        """OpenAI Responses input_image content → IR ImagePart.

        Parses data URIs back into ``image_data`` with media_type.

        Args:
            provider_image: OpenAI Responses image content dict.

        Returns:
            IR ImagePart.
        """
        detail = provider_image.get("detail", "auto")

        if "image_url" in provider_image:
            url = provider_image["image_url"]
            if url.startswith("data:"):
                match = re.match(r"data:([^;]+);base64,(.+)", url)
                if match:
                    media_type, data = match.groups()
                    return ImagePart(
                        type="image",
                        image_data={"data": data, "media_type": media_type},
                        detail=detail,
                    )
                else:
                    return ImagePart(type="image", image_url=url, detail=detail)
            else:
                return ImagePart(type="image", image_url=url, detail=detail)
        elif "file_id" in provider_image:
            return ImagePart(
                type="image",
                provider_ref={"file_id": provider_image["file_id"]},
                detail=detail,
            )

        return ImagePart(type="image", detail=detail)

    # ==================== File ====================

    @staticmethod
    def ir_file_to_p(ir_file: FilePart, **kwargs: Any) -> dict:
        """IR FilePart → OpenAI Responses input_file content part.

        Files are always input_file in Responses API.

        Args:
            ir_file: IR file part.

        Returns:
            OpenAI Responses file content dict.

        Raises:
            ValueError: If neither ``file_data`` nor ``file_url`` is present.
        """
        result: dict = {
            "type": "input_file",
            "filename": ir_file.get("file_name", "unknown"),
        }

        if "file_data" in ir_file:
            result["file_data"] = ir_file["file_data"]["data"]
        elif "file_url" in ir_file:
            result["file_url"] = ir_file["file_url"]
        elif "provider_ref" in ir_file and "file_id" in ir_file["provider_ref"]:
            result["file_id"] = ir_file["provider_ref"]["file_id"]
        else:
            raise ValueError("File part must have either file_data or file_url")

        return result

    @staticmethod
    def p_file_to_ir(provider_file: Any, **kwargs: Any) -> FilePart:
        """OpenAI Responses input_file content → IR FilePart.

        Args:
            provider_file: OpenAI Responses file content dict.

        Returns:
            IR FilePart.
        """
        file_name = provider_file.get("filename", "unknown")

        if "file_data" in provider_file:
            return FilePart(
                type="file",
                file_name=file_name,
                file_data=FileData(
                    data=provider_file["file_data"],
                    media_type="application/octet-stream",
                ),
            )
        elif "file_url" in provider_file:
            return FilePart(
                type="file", file_name=file_name, file_url=provider_file["file_url"]
            )
        elif "file_id" in provider_file:
            return FilePart(
                type="file",
                file_name=file_name,
                provider_ref={"file_id": provider_file["file_id"]},
            )

        return FilePart(type="file", file_name=file_name)

    # ==================== Audio (not supported) ====================

    @staticmethod
    def ir_audio_to_p(ir_audio: AudioPart, **kwargs: Any) -> Any:
        """IR AudioPart → OpenAI Responses audio content.

        Raises:
            NotImplementedError: OpenAI Responses API does not support audio parts.
        """
        raise NotImplementedError(
            "OpenAI Responses API does not support audio content parts."
        )

    @staticmethod
    def p_audio_to_ir(provider_audio: Any, **kwargs: Any) -> AudioPart:
        """OpenAI Responses audio content → IR AudioPart.

        Raises:
            NotImplementedError: OpenAI Responses API does not support audio parts.
        """
        raise NotImplementedError(
            "OpenAI Responses API does not support audio content parts."
        )

    # ==================== Reasoning ====================

    @staticmethod
    def ir_reasoning_to_p(ir_reasoning: ReasoningPart, **kwargs: Any) -> dict:
        """IR ReasoningPart → OpenAI Responses reasoning item.

        Args:
            ir_reasoning: IR reasoning part.

        Returns:
            OpenAI Responses reasoning item dict.
        """
        result: dict[str, Any] = {
            "type": "reasoning",
        }

        # Recover the original reasoning item id for round-trip fidelity.
        metadata = ir_reasoning.get("provider_metadata") or {}
        item_id = metadata.get("responses_reasoning_id")
        if item_id:
            result["id"] = item_id

        # Recover structured summary if available, otherwise use flat content.
        summary = metadata.get("responses_reasoning_summary")
        if summary:
            result["summary"] = summary
        else:
            content = ir_reasoning.get("reasoning", "")
            # The Responses API uses a summary array, not a flat content field.
            if content:
                result["summary"] = [{"type": "summary_text", "text": content}]
            else:
                result["summary"] = []

        # Open Responses spec: raw `content` field
        raw_content = metadata.get("responses_reasoning_content")
        if raw_content:
            result["content"] = raw_content

        # Preserve encryption signature for encrypted reasoning.
        signature = ir_reasoning.get("signature")
        if signature:
            result["encrypted_content"] = signature

        return result

    @staticmethod
    def p_reasoning_to_ir(
        provider_reasoning: Any, **kwargs: Any
    ) -> ReasoningPart | None:
        """OpenAI Responses reasoning item → IR ReasoningPart.

        Handles cases where reasoning content may be None (e.g., o4-mini).
        Preserves the item id and structured summary for round-trip fidelity.

        Args:
            provider_reasoning: OpenAI Responses reasoning item dict.

        Returns:
            IR ReasoningPart, or None if content is empty/null.
        """
        # Extract text from structured summary array or flat content.
        summary = provider_reasoning.get("summary")
        if isinstance(summary, list):
            texts = [
                s.get("text", "")
                for s in summary
                if isinstance(s, dict) and s.get("type") == "summary_text"
            ]
            reasoning_content = "".join(texts)
        else:
            reasoning_content = provider_reasoning.get(
                "reasoning"
            ) or provider_reasoning.get("content")

        # Build provider_metadata for round-trip preservation.
        metadata: dict[str, Any] = {}
        item_id = provider_reasoning.get("id")
        if item_id:
            metadata["responses_reasoning_id"] = item_id
        if isinstance(summary, list):
            metadata["responses_reasoning_summary"] = summary

        # Open Responses spec: raw `content` field (distinct from `summary`)
        raw_content = provider_reasoning.get("content")
        if raw_content:
            metadata["responses_reasoning_content"] = raw_content
            if not reasoning_content:
                reasoning_content = raw_content

        part = ReasoningPart(type="reasoning")

        if reasoning_content:
            part["reasoning"] = str(reasoning_content)

        # Preserve encrypted_content as signature.
        encrypted = provider_reasoning.get("encrypted_content")
        if encrypted:
            part["signature"] = str(encrypted)

        if metadata:
            part["provider_metadata"] = metadata

        # Return part if it has any meaningful content or metadata to preserve.
        if reasoning_content or encrypted or metadata:
            return part

        return None

    # ==================== Refusal ====================

    @staticmethod
    def ir_refusal_to_p(ir_refusal: RefusalPart, **kwargs: Any) -> dict | None:
        """IR RefusalPart → OpenAI Responses refusal content.

        OpenAI Responses API does not have a dedicated refusal format.
        Returns None and emits a warning.

        Args:
            ir_refusal: IR refusal part.

        Returns:
            None (refusal is dropped with a warning).
        """
        warnings.warn(
            "Refusal content not directly supported in OpenAI Responses API, ignored",
            stacklevel=2,
        )
        return None

    @staticmethod
    def p_refusal_to_ir(provider_refusal: Any, **kwargs: Any) -> RefusalPart:
        """OpenAI Responses refusal content → IR RefusalPart.

        Raises:
            NotImplementedError: OpenAI Responses API does not produce refusal parts.
        """
        raise NotImplementedError(
            "OpenAI Responses API does not produce refusal content parts."
        )

    # ==================== Citation ====================

    @staticmethod
    def ir_citation_to_p(ir_citation: CitationPart, **kwargs: Any) -> dict | None:
        """IR CitationPart → OpenAI Responses annotation.

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
        """OpenAI Responses annotation → IR CitationPart.

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
