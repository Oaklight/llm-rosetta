"""ACP content operations: the content blocks this lane speaks.

ACP carries a prompt as content blocks — text and inline images — and the
agent answers with text chunks and thought chunks.  Files and audio exist in
the protocol but the Antigravity agent advertises images only, so a file or
audio part is reported as a loss rather than sent as a block agy would
refuse.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta.conversion_warnings import DROPPED, DOWNGRADED, report_loss

from llm_rosetta.ir.model.parts import (
    AudioPart,
    CitationPart,
    FilePart,
    ImagePart,
    ReasoningPart,
    RefusalPart,
    TextPart,
)
from ...base.content import BaseContentOps


class AcpContentOps(BaseContentOps):
    """Content vocabulary of the ACP lane."""

    @staticmethod
    def ir_text_to_p(ir_text: TextPart, **kwargs: Any) -> dict[str, Any]:
        return {"type": "text", "text": ir_text.get("text", "")}

    @staticmethod
    def p_text_to_ir(provider_text: Any, **kwargs: Any) -> TextPart:
        block = provider_text if isinstance(provider_text, dict) else {}
        return TextPart(type="text", text=str(block.get("text") or ""))

    @staticmethod
    def ir_image_to_p(ir_image: ImagePart, **kwargs: Any) -> dict[str, Any] | None:
        data = ir_image.get("image_data") or {}
        if data.get("data"):
            return {
                "type": "image",
                "data": data["data"],
                "mimeType": data.get("media_type") or "image/png",
            }
        report_loss(
            "ACP lanes take inline images only; an image URL was dropped.",
            code=DROPPED,
            subject="image_url",
            warnings=kwargs.get("warnings"),
        )
        return None

    @staticmethod
    def p_image_to_ir(provider_image: Any, **kwargs: Any) -> ImagePart:
        block = provider_image if isinstance(provider_image, dict) else {}
        return ImagePart(
            type="image",
            image_data={
                "data": str(block.get("data") or ""),
                "media_type": str(
                    block.get("mimeType") or block.get("mime_type") or "image/png"
                ),
            },
        )

    @staticmethod
    def ir_file_to_p(ir_file: FilePart, **kwargs: Any) -> dict[str, Any] | None:
        report_loss(
            "ACP lanes cannot carry file parts yet; the part was dropped.",
            code=DROPPED,
            subject="file",
            warnings=kwargs.get("warnings"),
        )
        return None

    @staticmethod
    def p_file_to_ir(provider_file: Any, **kwargs: Any) -> FilePart:
        raise NotImplementedError("ACP lanes do not read file parts")

    @staticmethod
    def ir_audio_to_p(ir_audio: AudioPart, **kwargs: Any) -> dict[str, Any] | None:
        report_loss(
            "ACP lanes cannot carry audio parts; the part was dropped.",
            code=DROPPED,
            subject="audio",
            warnings=kwargs.get("warnings"),
        )
        return None

    @staticmethod
    def p_audio_to_ir(provider_audio: Any, **kwargs: Any) -> AudioPart:
        raise NotImplementedError("ACP lanes do not read audio parts")

    @staticmethod
    def ir_reasoning_to_p(ir_reasoning: ReasoningPart, **kwargs: Any) -> Any:
        # A replayed turn's thinking has no place in a prompt turn: ACP has no
        # block for it, and sending it as user text would show the model its
        # own past reasoning as instructions.
        report_loss(
            "ACP prompts cannot replay earlier reasoning; the part was dropped.",
            code=DROPPED,
            subject="reasoning",
            warnings=kwargs.get("warnings"),
        )
        return None

    @staticmethod
    def p_reasoning_to_ir(provider_reasoning: Any, **kwargs: Any) -> ReasoningPart:
        block = provider_reasoning if isinstance(provider_reasoning, dict) else {}
        return ReasoningPart(type="reasoning", reasoning=block.get("text") or "")

    @staticmethod
    def ir_refusal_to_p(ir_refusal: RefusalPart, **kwargs: Any) -> Any:
        # ACP has no refusal block.  The words a replayed refusal carried are
        # still words the model said, so they go as text rather than vanish.
        report_loss(
            "ACP prompts have no refusal block; the words were replayed as text.",
            code=DOWNGRADED,
            subject="refusal",
            warnings=kwargs.get("warnings"),
        )
        return {"type": "text", "text": ir_refusal.get("refusal") or ""}

    @staticmethod
    def p_refusal_to_ir(provider_refusal: Any, **kwargs: Any) -> RefusalPart:
        block = provider_refusal if isinstance(provider_refusal, dict) else {}
        return RefusalPart(type="refusal", refusal=str(block.get("text") or ""))

    @staticmethod
    def ir_citation_to_p(ir_citation: CitationPart, **kwargs: Any) -> Any:
        report_loss(
            "ACP lanes carry no citations; the citation was dropped.",
            code=DROPPED,
            subject="citation",
            warnings=kwargs.get("warnings"),
        )
        return None

    @staticmethod
    def p_citation_to_ir(provider_citation: Any, **kwargs: Any) -> CitationPart:
        raise NotImplementedError("ACP lanes do not carry citations")


__all__ = ["AcpContentOps"]
