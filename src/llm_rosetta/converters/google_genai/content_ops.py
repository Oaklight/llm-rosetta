"""
LLM-Rosetta - Google GenAI Content Operations

Google GenAI API content conversion operations.
Handles bidirectional conversion of text, image, file, audio, reasoning,
and other content parts using Google's Part-based architecture.

Self-contained: does not depend on utils/FieldMapper or utils/ToolCallConverter.
"""

import base64
import logging
import mimetypes
import warnings
from typing import Any, cast

import httpx

from ...types.ir import (
    AudioData,
    AudioPart,
    CitationPart,
    FilePart,
    ImageData,
    ImagePart,
    ReasoningPart,
    RefusalPart,
    TextPart,
)
from ...types.ir.parts import ContentPart
from ..base import BaseContentOps


class GoogleGenAIContentOps(BaseContentOps):
    """Google GenAI content conversion operations.

    All methods are static and stateless. Handles TextPart, ImagePart,
    FilePart, AudioPart, and ReasoningPart bidirectional conversion.

    Google uses a Part-based architecture where all content is represented
    as Part objects containing text, inline_data, function_call,
    function_response, or thought fields.
    """

    # ==================== Text ====================

    @staticmethod
    def ir_text_to_p(ir_text: TextPart, **kwargs: Any) -> dict:
        """IR TextPart → Google GenAI text Part.

        Args:
            ir_text: IR text part.

        Returns:
            Google text Part dict: ``{"text": "..."}``
        """
        part: dict[str, Any] = {"text": ir_text["text"]}
        # Preserve thought_signature in provider_metadata
        provider_metadata = cast(dict, ir_text).get("provider_metadata")
        if provider_metadata:
            google_meta = provider_metadata.get("google", {})
            if "thought_signature" in google_meta:
                part["thoughtSignature"] = google_meta["thought_signature"]
        return part

    @staticmethod
    def p_text_to_ir(provider_text: Any, **kwargs: Any) -> TextPart:
        """Google GenAI text Part → IR TextPart.

        Args:
            provider_text: Google Part dict with ``text`` field.

        Returns:
            IR TextPart.
        """
        return TextPart(type="text", text=provider_text["text"])

    # ==================== Image ====================

    @staticmethod
    def ir_image_to_p(ir_image: ImagePart, **kwargs: Any) -> dict | None:
        """IR ImagePart → Google GenAI inline_data Part.

        Google uses inline_data for base64-encoded images. URL-based images
        are not directly supported and will emit a warning.

        Args:
            ir_image: IR image part.

        Returns:
            Google inline_data Part dict, or None if unsupported format.
        """
        if "image_data" in ir_image:
            image_data = ir_image["image_data"]
            return {
                "inlineData": {
                    "mimeType": image_data["media_type"],
                    "data": image_data["data"],
                }
            }

        # URL-based images: download and convert to inline base64
        url = ir_image.get("image_url") or ir_image.get("url")
        if url:
            # Handle data URIs directly
            if url.startswith("data:"):
                # data:image/png;base64,<data>
                try:
                    header, data = url.split(",", 1)
                    mime = header.split(":")[1].split(";")[0]
                    return {"inlineData": {"mimeType": mime, "data": data}}
                except (IndexError, ValueError):
                    pass
                return None

            try:
                import os

                proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
                headers = {
                    "User-Agent": "llm-rosetta/1.0 (image fetch)",
                }
                client_kwargs: dict[str, Any] = {
                    "timeout": 30,
                    "follow_redirects": True,
                    "headers": headers,
                }
                if proxy:
                    client_kwargs["proxy"] = proxy
                resp = httpx.get(url, **client_kwargs)
                resp.raise_for_status()
                mime = (
                    resp.headers.get("content-type", "").split(";")[0].strip()
                    or mimetypes.guess_type(url)[0]
                    or "image/jpeg"
                )
                b64 = base64.b64encode(resp.content).decode()
                return {"inlineData": {"mimeType": mime, "data": b64}}
            except Exception as exc:
                logging.getLogger("llm-rosetta").warning(
                    "Failed to download image URL for Google GenAI conversion: %s",
                    exc,
                )
                return None

        return None

    @staticmethod
    def p_image_to_ir(provider_image: Any, **kwargs: Any) -> ImagePart:
        """Google GenAI inline_data Part → IR ImagePart.

        Args:
            provider_image: Google Part dict with ``inline_data`` containing image.

        Returns:
            IR ImagePart.
        """
        inline_data = provider_image.get("inline_data") or provider_image.get(
            "inlineData", {}
        )
        return ImagePart(
            type="image",
            image_data=ImageData(
                data=inline_data.get("data", ""),
                media_type=inline_data.get(
                    "mime_type", inline_data.get("mimeType", "")
                ),
            ),
        )

    # ==================== File ====================

    @staticmethod
    def ir_file_to_p(ir_file: FilePart, **kwargs: Any) -> dict | None:
        """IR FilePart → Google GenAI inline_data Part.

        Args:
            ir_file: IR file part.

        Returns:
            Google inline_data Part dict, or None if unsupported format.
        """
        if "file_data" in ir_file:
            file_data = ir_file["file_data"]
            return {
                "inlineData": {
                    "mimeType": file_data["media_type"],
                    "data": file_data["data"],
                }
            }

        # URL-based files not supported
        if "file_url" in ir_file:
            warnings.warn("Google GenAI不直接支持文件URL，需要先上传文件。")
            return None

        return None

    @staticmethod
    def p_file_to_ir(provider_file: Any, **kwargs: Any) -> FilePart:
        """Google GenAI inline_data Part → IR FilePart.

        Args:
            provider_file: Google Part dict with ``inline_data`` containing file.

        Returns:
            IR FilePart.
        """
        inline_data = provider_file.get("inline_data") or provider_file.get(
            "inlineData", {}
        )
        return {
            "type": "file",
            "file_data": {
                "data": inline_data.get("data", ""),
                "media_type": inline_data.get(
                    "mime_type", inline_data.get("mimeType", "")
                ),
            },
        }

    # ==================== Audio ====================

    @staticmethod
    def ir_audio_to_p(ir_audio: AudioPart, **kwargs: Any) -> dict | None:
        """IR AudioPart → Google GenAI inline_data Part.

        Args:
            ir_audio: IR audio part.

        Returns:
            Google inline_data Part dict, or None if unsupported format.
        """
        if "audio_data" in ir_audio:
            audio_data = ir_audio["audio_data"]
            return {
                "inlineData": {
                    "mimeType": audio_data["media_type"],
                    "data": audio_data["data"],
                }
            }

        warnings.warn("不支持的音频格式")
        return None

    @staticmethod
    def p_audio_to_ir(provider_audio: Any, **kwargs: Any) -> AudioPart:
        """Google GenAI inline_data/file_data Part → IR AudioPart.

        Args:
            provider_audio: Google Part dict with audio data.

        Returns:
            IR AudioPart.
        """
        raw_inline = provider_audio.get("inline_data") or provider_audio.get(
            "inlineData"
        )
        if raw_inline:
            return AudioPart(
                type="audio",
                audio_data=AudioData(
                    data=raw_inline.get("data", ""),
                    media_type=raw_inline.get(
                        "mime_type", raw_inline.get("mimeType", "")
                    ),
                ),
            )
        raw_file = provider_audio.get("file_data") or provider_audio.get("fileData")
        if raw_file:
            return AudioPart(
                type="audio",
                url=raw_file.get("file_uri", raw_file.get("fileUri", "")),
            )
        raise ValueError("Audio part must have inline_data or file_data")

    # ==================== Reasoning (Thought) ====================

    @staticmethod
    def ir_reasoning_to_p(ir_reasoning: ReasoningPart, **kwargs: Any) -> dict:
        """IR ReasoningPart → Google GenAI thought Part.

        Google represents reasoning as a Part with ``thought=True`` and
        the reasoning text in the ``text`` field.

        Args:
            ir_reasoning: IR reasoning part.

        Returns:
            Google thought Part dict.
        """
        part: dict[str, Any] = {
            "thought": True,
            "text": ir_reasoning.get("reasoning", ""),
        }
        # Preserve thought_signature
        provider_metadata = cast(dict, ir_reasoning).get("provider_metadata")
        if provider_metadata:
            google_meta = provider_metadata.get("google", {})
            if "thought_signature" in google_meta:
                part["thoughtSignature"] = google_meta["thought_signature"]
        return part

    @staticmethod
    def p_reasoning_to_ir(provider_reasoning: Any, **kwargs: Any) -> ReasoningPart:
        """Google GenAI thought Part → IR ReasoningPart.

        Args:
            provider_reasoning: Google Part dict with ``thought=True``.

        Returns:
            IR ReasoningPart.
        """
        return ReasoningPart(
            type="reasoning", reasoning=provider_reasoning.get("text", "")
        )

    # ==================== Refusal (not natively supported) ====================

    @staticmethod
    def ir_refusal_to_p(ir_refusal: RefusalPart, **kwargs: Any) -> dict:
        """IR RefusalPart → Google GenAI text Part.

        Google does not have a native refusal type. Refusals are converted
        to text parts with a prefix.

        Args:
            ir_refusal: IR refusal part.

        Returns:
            Google text Part dict.
        """
        return {"text": f"[Refusal] {ir_refusal['refusal']}"}

    @staticmethod
    def p_refusal_to_ir(provider_refusal: Any, **kwargs: Any) -> RefusalPart:
        """Google GenAI text → IR RefusalPart.

        Raises:
            NotImplementedError: Google does not produce native refusal parts.
        """
        raise NotImplementedError(
            "Google GenAI does not produce native refusal content parts."
        )

    # ==================== Citation (not natively supported in parts) ====================

    @staticmethod
    def ir_citation_to_p(ir_citation: CitationPart, **kwargs: Any) -> dict | None:
        """IR CitationPart → Google GenAI format.

        Google handles citations at the response level (grounding_metadata),
        not as individual content parts.

        Args:
            ir_citation: IR citation part.

        Returns:
            None (citations are not supported as content parts).
        """
        warnings.warn(
            "Google GenAI does not support citations as content parts, ignored",
            stacklevel=2,
        )
        return None

    @staticmethod
    def p_citation_to_ir(provider_citation: Any, **kwargs: Any) -> CitationPart:
        """Google GenAI citation → IR CitationPart.

        Raises:
            NotImplementedError: Google citations are at response level.
        """
        raise NotImplementedError(
            "Google GenAI citations are at the response level, "
            "not individual content parts."
        )

    # ==================== Composite Part Dispatch ====================

    @staticmethod
    def p_part_to_ir(provider_part: Any) -> list[dict[str, Any]]:
        """Convert a single Google Part to IR content part(s).

        Handles all Part types: text, inline_data, file_data,
        function_call, function_response, thought, and thoughtSignature.

        Note: function_call and function_response are handled by ToolOps,
        so this method only handles content-related parts.

        Args:
            provider_part: Google Part dict.

        Returns:
            List of IR content parts.
        """
        ir_parts: list[ContentPart] = []

        # Handle text
        if (
            "text" in provider_part
            and provider_part["text"] is not None
            and provider_part["text"] != ""
        ):
            ir_parts.append(GoogleGenAIContentOps.p_text_to_ir(provider_part))

        # Handle inline_data / inlineData (image, audio, or file based on mime_type)
        raw_inline = provider_part.get("inline_data") or provider_part.get("inlineData")
        if raw_inline is not None:
            # Normalize camelCase keys inside inline_data
            inline_data = {
                "data": raw_inline.get("data", raw_inline.get("data", "")),
                "mime_type": raw_inline.get(
                    "mime_type", raw_inline.get("mimeType", "")
                ),
            }
            # Attach normalized inline_data so downstream p_image_to_ir etc. can read it
            provider_part = {**provider_part, "inline_data": inline_data}
            mime_type = inline_data.get("mime_type", "")

            if mime_type.startswith("image/"):
                ir_parts.append(GoogleGenAIContentOps.p_image_to_ir(provider_part))
            elif mime_type.startswith("audio/"):
                ir_parts.append(GoogleGenAIContentOps.p_audio_to_ir(provider_part))
            else:
                ir_parts.append(GoogleGenAIContentOps.p_file_to_ir(provider_part))

        # Handle file_data / fileData (URI-based)
        raw_file = provider_part.get("file_data") or provider_part.get("fileData")
        if raw_file is not None:
            file_data = {
                "mime_type": raw_file.get("mime_type", raw_file.get("mimeType", "")),
                "file_uri": raw_file.get("file_uri", raw_file.get("fileUri", "")),
            }
            mime_type = file_data.get("mime_type", "")

            if mime_type.startswith("image/"):
                ir_parts.append(
                    ImagePart(type="image", image_url=file_data["file_uri"])
                )
            elif mime_type.startswith("audio/"):
                ir_parts.append(
                    AudioPart(
                        type="audio",
                        url=file_data["file_uri"],
                    )
                )
            else:
                ir_parts.append(FilePart(type="file", file_url=file_data["file_uri"]))

        # Handle thoughtSignature on content parts
        thought_sig = provider_part.get("thoughtSignature") or provider_part.get(
            "thought_signature"
        )
        if thought_sig and ir_parts:
            last_part = cast(dict, ir_parts[-1])
            if "provider_metadata" not in last_part:
                last_part["provider_metadata"] = {}
            if "google" not in last_part["provider_metadata"]:
                last_part["provider_metadata"]["google"] = {}
            last_part["provider_metadata"]["google"]["thought_signature"] = thought_sig

        return cast(list[dict[str, Any]], ir_parts)
