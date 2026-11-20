"""Keep same-turn reasoning with text and tools through Responses conversion."""

import copy

import pytest

from llm_rosetta.pipeline import ConversionPipeline


def convert_history(items):
    body = {"model": "example-model", "input": items, "reasoning": {"effort": "high"}}
    original = copy.deepcopy(body)
    converted = ConversionPipeline(
        "openai--open_responses", "chat_completions"
    ).convert_request(body)
    assert body == original
    return converted


def tool_call(call_id="call_1"):
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": "lookup",
        "arguments": "{}",
    }


def tool_result(output="ok", call_id="call_1"):
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def reasoning(text="Think before looking up."):
    return {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": text}],
    }


def assistant_text(text="I will look it up.", **extra):
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
        **extra,
    }


def test_reasoning_text_and_tool_call_remain_in_the_same_assistant_turn():
    converted = convert_history(
        [
            {"role": "user", "content": "Look it up."},
            reasoning(),
            assistant_text(),
            tool_call(),
            tool_result(),
        ]
    )
    assistants = [m for m in converted["messages"] if m["role"] == "assistant"]
    assert len(assistants) == 1
    assert assistants[0]["reasoning_content"] == "Think before looking up."
    assert assistants[0]["content"] == "I will look it up."
    assert [c["id"] for c in assistants[0]["tool_calls"]] == ["call_1"]


def test_reasoning_identity_and_grouping_survive_responses_ir_roundtrip():
    items = [
        {"role": "user", "content": "Read."},
        {**reasoning(), "id": "rs_original"},
        assistant_text(),
        tool_call(),
        tool_result(),
    ]
    convert_history(items)
    from llm_rosetta import OpenAIResponsesConverter

    converter = OpenAIResponsesConverter()
    ir = converter.request_from_provider({"model": "demo", "input": items})
    responses, warnings = converter.request_to_provider(ir)
    assert not warnings
    reasoning_item = next(x for x in responses["input"] if x["type"] == "reasoning")
    assert reasoning_item["id"] == "rs_original"
    assert reasoning_item["summary"] == reasoning()["summary"]
    restored = convert_history(responses["input"])
    assistant = next(m for m in restored["messages"] if m.get("tool_calls"))
    assert assistant["reasoning_content"] == "Think before looking up."
    assert assistant["content"] == "I will look it up."
    assert assistant["tool_calls"][0]["id"] == "call_1"


def test_parallel_tools_and_later_turn_keep_their_own_reasoning():
    converted = convert_history(
        [
            {"role": "user", "content": "First question."},
            reasoning("First reasoning."),
            assistant_text("Checking two things."),
            tool_call("call_1"),
            tool_call("call_2"),
            tool_result("one", "call_1"),
            tool_result("two", "call_2"),
            reasoning("Continuation reasoning."),
            assistant_text("Checking the next thing."),
            tool_call("call_3"),
            tool_result("three", "call_3"),
            assistant_text("First answer."),
            {"role": "user", "content": "Second question."},
            reasoning("Second question reasoning."),
            assistant_text("Checking again."),
            tool_call("call_4"),
            tool_result("four", "call_4"),
        ]
    )
    assistants = [m for m in converted["messages"] if m["role"] == "assistant"]
    assert len(assistants) == 4
    assert assistants[0]["reasoning_content"] == "First reasoning."
    assert [c["id"] for c in assistants[0]["tool_calls"]] == ["call_1", "call_2"]
    assert assistants[1]["reasoning_content"] == "Continuation reasoning."
    assert [c["id"] for c in assistants[1]["tool_calls"]] == ["call_3"]
    assert not assistants[2].get("reasoning_content")
    assert not assistants[2].get("tool_calls")
    assert assistants[3]["reasoning_content"] == "Second question reasoning."
    assert [c["id"] for c in assistants[3]["tool_calls"]] == ["call_4"]
    results = [m for m in converted["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == [
        "call_1",
        "call_2",
        "call_3",
        "call_4",
    ]


@pytest.mark.parametrize("role", ["user", "system", "developer"])
def test_reasoning_does_not_leak_across_an_explicit_role_boundary(role):
    converted = convert_history(
        [
            {"role": "user", "content": "Earlier question."},
            reasoning("Old reasoning."),
            assistant_text("Earlier answer."),
            {"role": role, "content": "A new boundary."},
            assistant_text("A separate assistant turn."),
            tool_call(),
            tool_result(),
        ]
    )
    caller = next(m for m in converted["messages"] if m.get("tool_calls"))
    assert not caller.get("reasoning_content")


def test_explicit_response_message_phases_are_not_collapsed():
    converted = convert_history(
        [
            {"role": "user", "content": "Question."},
            reasoning(),
            assistant_text("Commentary.", phase="commentary", status="completed"),
            assistant_text("Final answer.", phase="final_answer", status="completed"),
        ]
    )
    # Chat has no phase field, but the initial conversion must not merge the
    # separate Responses messages and accidentally associate old reasoning.
    assistants = [m for m in converted["messages"] if m["role"] == "assistant"]
    assert len(assistants) == 2
    assert assistants[0]["content"] == "Commentary."
    assert assistants[0]["reasoning_content"] == "Think before looking up."
    assert assistants[1]["content"] == "Final answer."
    assert not assistants[1].get("reasoning_content")
