"""Generate the three inscription scripts via the REAL llm-rosetta converter.

Two canonical requests (the IR sources) are each converted to all three
provider dialects with the actual `convert()` tool — so the stone's three
script bands are genuine llm-rosetta output, not hand-written. Literal Rosetta
Stone: one decree, three scripts, produced by the translation engine itself.

Each band = its dialect's output for request 1 + request 2, hard-wrapped to a
fixed width so every line is full and the inscription packs edge-to-edge.

Writes scripts_data.py consumed by build_logo.py.
"""
from __future__ import annotations
import json
from pathlib import Path
from llm_rosetta import convert

HERE = Path(__file__).parent

# ONE self-referential request — a multi-turn conversation about llm-rosetta
# and the Rosetta Stone, with three tools. The whole decree, carried in three
# API dialects: exactly the Rosetta Stone metaphor (one text, three scripts).
# Rich enough to fill the stone top-to-bottom.
REQUEST = {
    "model": "gpt-5",
    "messages": [
        {"role": "system",
         "content": "You are llm-rosetta, a universal translator for large "
                    "language model APIs. You convert any request between the "
                    "OpenAI Chat Completions, OpenAI Responses, Anthropic "
                    "Messages, and Google GenAI formats through one neutral "
                    "intermediate representation, preserving every message, "
                    "system instruction, tool definition, tool call, image, "
                    "and reasoning block exactly as given. You never invent "
                    "fields and never drop information in translation."},
        {"role": "user",
         "content": "The Rosetta Stone of 196 BC bore a single decree of "
                    "Ptolemy V inscribed in three scripts — Egyptian "
                    "hieroglyphic, Demotic, and Ancient Greek — which let "
                    "scholars finally decipher the hieroglyphs. Carry this one "
                    "request across the modern API dialects in the same way."},
        {"role": "assistant",
         "content": "Understood. One intermediate representation, many provider "
                    "formats. I will keep the meaning identical across all of "
                    "them and only change the surface dialect, exactly as the "
                    "three scripts on the stone all say the same thing."},
        {"role": "user",
         "content": "Translate the greeting 'hello' into French, and if you are "
                    "unsure what the stone actually says, search the web and "
                    "fetch a reliable source before answering."},
    ],
    "tools": [
        {"type": "function", "function": {
            "name": "convert_request",
            "description": "translate an LLM API payload between providers",
            "parameters": {"type": "object",
                           "properties": {"source": {"type": "string"},
                                          "target": {"type": "string"},
                                          "preserve_tools": {"type": "boolean"}},
                           "required": ["source", "target"]}}},
        {"type": "function", "function": {
            "name": "transliterate",
            "description": "render ancient glyphs as readable text",
            "parameters": {"type": "object",
                           "properties": {"script": {"type": "string"},
                                          "lang": {"type": "string"}},
                           "required": ["script"]}}},
        {"type": "function", "function": {
            "name": "web_search",
            "description": "search the web for context on a query",
            "parameters": {"type": "object",
                           "properties": {"query": {"type": "string"},
                                          "max_results": {"type": "integer"}},
                           "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "web_fetch",
            "description": "fetch and extract the contents of a URL",
            "parameters": {"type": "object",
                           "properties": {"url": {"type": "string"},
                                          "max_chars": {"type": "integer"}},
                           "required": ["url"]}}},
    ],
    "tool_choice": "auto",
    "temperature": 0.2, "top_p": 0.9, "max_tokens": 512, "stream": True,
}

WRAP = 92  # overrun right edge at all heights, clipped to full rows


def hard_wrap(s: str, width: int) -> list[str]:
    return [s[i:i + width] for i in range(0, len(s), width)]


def to_lines(obj: dict) -> list[str]:
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return hard_wrap(s, WRAP)


def main() -> None:
    blocks = []
    for tgt in ("openai_chat", "anthropic", "google"):
        out = convert(REQUEST, tgt, "openai_chat")
        blocks.append((tgt, to_lines(out)))

    py = ['"""AUTO-GENERATED by gen_scripts.py — real convert() output. Do not edit."""',
          "", "SCRIPTS = ["]
    for name, lines in blocks:
        py.append(f"    ({name!r}, [")
        for ln in lines:
            py.append(f"        {ln!r},")
        py.append("    ]),")
    py.append("]")
    (HERE / "scripts_data.py").write_text("\n".join(py) + "\n")
    print("generated:", {n: len(l) for n, l in blocks},
          "total", sum(len(l) for _, l in blocks))


if __name__ == "__main__":
    main()
