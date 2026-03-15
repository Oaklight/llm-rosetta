"""
LLM-Rosetta - Base Content Operations
内容转换操作的抽象基类
Abstract base class for content conversion operations

处理所有类型的内容部分转换：
- 基础内容：文本、图像、文件、音频
- 特殊内容：推理、拒绝、引用
Handles all types of content part conversions:
- Basic content: text, image, file, audio
- Special content: reasoning, refusal, citation
"""

from abc import ABC, abstractmethod
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


class BaseContentOps(ABC):
    """内容转换操作的抽象基类
    Abstract base class for content conversion operations

    按内容类型组织转换方法，每种内容类型都有对应的双向转换方法。
    Organizes conversion methods by content type, with bidirectional conversion for each type.
    """

    # ==================== 基础内容转换 Basic content conversion ====================

    @staticmethod
    @abstractmethod
    def ir_text_to_p(ir_text: TextPart, **kwargs: Any) -> Any:
        """IR TextPart → Provider Text Content
        将IR文本部分转换为Provider文本内容

        Args:
            ir_text: IR格式的文本部分
            **kwargs: 额外参数

        Returns:
            Provider格式的文本内容
        """
        pass

    @staticmethod
    @abstractmethod
    def p_text_to_ir(provider_text: Any, **kwargs: Any) -> TextPart:
        """Provider Text Content → IR TextPart
        将Provider文本内容转换为IR文本部分

        Args:
            provider_text: Provider格式的文本内容
            **kwargs: 额外参数

        Returns:
            IR格式的文本部分
        """
        pass

    @staticmethod
    @abstractmethod
    def ir_image_to_p(ir_image: ImagePart, **kwargs: Any) -> Any:
        """IR ImagePart → Provider Image Content
        将IR图像部分转换为Provider图像内容

        Args:
            ir_image: IR格式的图像部分
            **kwargs: 额外参数

        Returns:
            Provider格式的图像内容
        """
        pass

    @staticmethod
    @abstractmethod
    def p_image_to_ir(provider_image: Any, **kwargs: Any) -> ImagePart:
        """Provider Image Content → IR ImagePart
        将Provider图像内容转换为IR图像部分

        Args:
            provider_image: Provider格式的图像内容
            **kwargs: 额外参数

        Returns:
            IR格式的图像部分
        """
        pass

    @staticmethod
    @abstractmethod
    def ir_file_to_p(ir_file: FilePart, **kwargs: Any) -> Any:
        """IR FilePart → Provider File Content
        将IR文件部分转换为Provider文件内容

        Args:
            ir_file: IR格式的文件部分
            **kwargs: 额外参数

        Returns:
            Provider格式的文件内容
        """
        pass

    @staticmethod
    @abstractmethod
    def p_file_to_ir(provider_file: Any, **kwargs: Any) -> FilePart:
        """Provider File Content → IR FilePart
        将Provider文件内容转换为IR文件部分

        Args:
            provider_file: Provider格式的文件内容
            **kwargs: 额外参数

        Returns:
            IR格式的文件部分
        """
        pass

    @staticmethod
    @abstractmethod
    def ir_audio_to_p(ir_audio: AudioPart, **kwargs: Any) -> Any:
        """IR AudioPart → Provider Audio Content
        将IR音频部分转换为Provider音频内容

        Args:
            ir_audio: IR格式的音频部分
            **kwargs: 额外参数

        Returns:
            Provider格式的音频内容
        """
        pass

    @staticmethod
    @abstractmethod
    def p_audio_to_ir(provider_audio: Any, **kwargs: Any) -> AudioPart:
        """Provider Audio Content → IR AudioPart
        将Provider音频内容转换为IR音频部分

        Args:
            provider_audio: Provider格式的音频内容
            **kwargs: 额外参数

        Returns:
            IR格式的音频部分
        """
        pass

    # ==================== 特殊内容转换 Special content conversion ====================

    @staticmethod
    @abstractmethod
    def ir_reasoning_to_p(ir_reasoning: ReasoningPart, **kwargs: Any) -> Any:
        """IR ReasoningPart → Provider Reasoning Content
        将IR推理部分转换为Provider推理内容

        用于处理模型的思考过程，如OpenAI的reasoning或Anthropic的thinking。
        Used to handle model's thought process, such as OpenAI's reasoning or Anthropic's thinking.

        Args:
            ir_reasoning: IR格式的推理部分
            **kwargs: 额外参数

        Returns:
            Provider格式的推理内容
        """
        pass

    @staticmethod
    @abstractmethod
    def p_reasoning_to_ir(
        provider_reasoning: Any, **kwargs: Any
    ) -> ReasoningPart | None:
        """Provider Reasoning Content → IR ReasoningPart
        将Provider推理内容转换为IR推理部分

        Args:
            provider_reasoning: Provider格式的推理内容
            **kwargs: 额外参数

        Returns:
            IR格式的推理部分
        """
        pass

    @staticmethod
    @abstractmethod
    def ir_refusal_to_p(ir_refusal: RefusalPart, **kwargs: Any) -> Any:
        """IR RefusalPart → Provider Refusal Content
        将IR拒绝部分转换为Provider拒绝内容

        用于处理模型拒绝回答的情况，如OpenAI的refusal。
        Used to handle cases where the model refuses to answer, such as OpenAI's refusal.

        Args:
            ir_refusal: IR格式的拒绝部分
            **kwargs: 额外参数

        Returns:
            Provider格式的拒绝内容
        """
        pass

    @staticmethod
    @abstractmethod
    def p_refusal_to_ir(provider_refusal: Any, **kwargs: Any) -> RefusalPart:
        """Provider Refusal Content → IR RefusalPart
        将Provider拒绝内容转换为IR拒绝部分

        Args:
            provider_refusal: Provider格式的拒绝内容
            **kwargs: 额外参数

        Returns:
            IR格式的拒绝部分
        """
        pass

    @staticmethod
    @abstractmethod
    def ir_citation_to_p(ir_citation: CitationPart, **kwargs: Any) -> Any:
        """IR CitationPart → Provider Citation Content
        将IR引用部分转换为Provider引用内容

        用于处理信息来源标注，如OpenAI的annotations或Anthropic的citations。
        Used to handle information source annotations, such as OpenAI's annotations or Anthropic's citations.

        Args:
            ir_citation: IR格式的引用部分
            **kwargs: 额外参数

        Returns:
            Provider格式的引用内容
        """
        pass

    @staticmethod
    @abstractmethod
    def p_citation_to_ir(provider_citation: Any, **kwargs: Any) -> CitationPart:
        """Provider Citation Content → IR CitationPart
        将Provider引用内容转换为IR引用部分

        Args:
            provider_citation: Provider格式的引用内容
            **kwargs: 额外参数

        Returns:
            IR格式的引用部分
        """
        pass
