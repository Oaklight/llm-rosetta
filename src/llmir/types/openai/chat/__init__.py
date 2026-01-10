"""OpenAI Chat Completion API Types.

This package contains TypedDict replicas of OpenAI SDK's chat completion types.
These types are used for type hints and validation in the LLMIR library.

The types are organized into three modules:
- request_types: Request parameter types
- message_types: Message and content part types
- response_types: Response and usage types
"""

# Request types
from .request_types import (
    ChatModel,
    ChatCompletionFunctionToolParam,
    ChatCompletionNamedToolChoiceFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    CompletionCreateParams,
    FunctionDefinition,
    FunctionParameters,
    Metadata,
    ReasoningEffort,
    ResponseFormat,
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
    ResponseFormatJSONSchemaSchema,
    ResponseFormatText,
)

# Message types
from .message_types import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallFunction,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
    FunctionCall,
    ImageURL,
    InputAudio,
)

# Response types
from .response_types import (
    Annotation,
    AnnotationURLCitation,
    ChatCompletion,
    ChatCompletionAudio,
    ChatCompletionMessage,
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCallUnion,
    ChatCompletionTokenLogprob,
    Choice,
    ChoiceLogprobs,
    CompletionTokensDetails,
    CompletionUsage,
    Function,
    FunctionCallResponse,
    PromptTokensDetails,
    TopLogprob,
)

__all__ = [
    # Request types
    "ChatModel",
    "ChatCompletionAudioParam",
    "ChatCompletionFunctionToolParam",
    "ChatCompletionNamedToolChoiceFunction",
    "ChatCompletionNamedToolChoiceParam",
    "ChatCompletionPredictionContentParam",
    "ChatCompletionStreamOptionsParam",
    "ChatCompletionToolChoiceOptionParam",
    "CompletionCreateParams",
    "FunctionDefinition",
    "FunctionParameters",
    "Metadata",
    "ReasoningEffort",
    "ResponseFormat",
    "ResponseFormatJSONObject",
    "ResponseFormatJSONSchema",
    "ResponseFormatJSONSchemaSchema",
    "ResponseFormatText",
    "WebSearchOptions",
    # Message types
    "ChatCompletionAssistantMessageParam",
    "ChatCompletionContentPartImageParam",
    "ChatCompletionContentPartInputAudioParam",
    "ChatCompletionContentPartParam",
    "ChatCompletionContentPartTextParam",
    "ChatCompletionDeveloperMessageParam",
    "ChatCompletionFunctionMessageParam",
    "ChatCompletionMessageParam",
    "ChatCompletionMessageToolCallFunction",
    "ChatCompletionMessageToolCallParam",
    "ChatCompletionSystemMessageParam",
    "ChatCompletionToolMessageParam",
    "ChatCompletionUserMessageParam",
    "FunctionCall",
    "ImageURL",
    "InputAudio",
    # Response types
    "Annotation",
    "AnnotationURLCitation",
    "ChatCompletion",
    "ChatCompletionAudio",
    "ChatCompletionMessage",
    "ChatCompletionMessageCustomToolCall",
    "ChatCompletionMessageFunctionToolCall",
    "ChatCompletionMessageToolCallUnion",
    "ChatCompletionTokenLogprob",
    "Choice",
    "ChoiceLogprobs",
    "CompletionTokensDetails",
    "CompletionUsage",
    "Function",
    "FunctionCallResponse",
    "PromptTokensDetails",
    "TopLogprob",
]
