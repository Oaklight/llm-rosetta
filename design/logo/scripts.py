"""The three 'scripts' of the LLM-Rosetta stone — real API payloads.

Like the real Rosetta Stone carries one decree in three scripts (hieroglyphic,
demotic, Greek), our stone carries the *same* request in three API dialects:
OpenAI Chat Completions, Anthropic Messages, and Google GenAI.

Rendered tiny: from afar it's carved-line texture; up close it's an easter
egg — three real, equivalent request bodies. Exactly what llm-rosetta does:
one intermediate request, three provider formats.

The payloads are deliberately full (system + tools + generation knobs) so the
inscription fills the stone densely, like the real artifact's edge-to-edge text.
"""

OPENAI_CHAT = [
    '{"model":"gpt-5","messages":[',
    '  {"role":"system","content":"You are a translator."},',
    '  {"role":"user","content":"Translate \'hello\' to French."}],',
    '"tools":[{"type":"function","function":{"name":"glossary",',
    '  "description":"look up a term","parameters":{"type":"object",',
    '  "properties":{"term":{"type":"string"}},"required":["term"]}}}],',
    '"tool_choice":"auto","response_format":{"type":"text"},',
    '"temperature":0.2,"top_p":0.9,"frequency_penalty":0,',
    '"max_completion_tokens":256,"stream":true}',
]

ANTHROPIC_MESSAGES = [
    '{"model":"claude-opus-4-6","max_tokens":256,',
    '"system":"You are a translator.","messages":[',
    '  {"role":"user","content":[{"type":"text",',
    '   "text":"Translate \'hello\' to French."}]}],',
    '"tools":[{"name":"glossary","description":"look up a term",',
    '  "input_schema":{"type":"object","properties":{',
    '   "term":{"type":"string"}},"required":["term"]}}],',
    '"tool_choice":{"type":"auto"},"temperature":0.2,"top_p":0.9,',
    '"thinking":{"type":"enabled","budget_tokens":1024},"stream":true}',
]

GOOGLE_GENAI = [
    '{"contents":[{"role":"user","parts":[',
    '  {"text":"Translate \'hello\' to French."}]}],',
    '"systemInstruction":{"parts":[{"text":"You are a translator."}]},',
    '"tools":[{"functionDeclarations":[{"name":"glossary",',
    '  "description":"look up a term","parameters":{"type":"OBJECT",',
    '  "properties":{"term":{"type":"STRING"}},"required":["term"]}}]}],',
    '"generationConfig":{"temperature":0.2,"topP":0.9,',
    '  "maxOutputTokens":256,"responseMimeType":"text/plain",',
    '  "thinkingConfig":{"thinkingBudget":1024}}}',
]

# Ordered top→bottom, mirroring the real artifact's three bands.
SCRIPTS = [
    ("openai_chat", OPENAI_CHAT),
    ("anthropic", ANTHROPIC_MESSAGES),
    ("google", GOOGLE_GENAI),
]
