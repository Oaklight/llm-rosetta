"""The three 'scripts' of the LLM-Rosetta stone — real API payloads.

Like the real Rosetta Stone carries one decree in three scripts (hieroglyphic,
demotic, Greek), our stone carries the *same* request in three API dialects:
OpenAI Chat Completions, Anthropic Messages, and Google GenAI.

Rendered at a tiny size: from afar it's carved-line texture; up close it's an
easter egg — three real, equivalent request bodies. This mirrors exactly what
llm-rosetta does: one intermediate request, three provider formats.
"""

# Same logical request — "Translate 'hello' to French" — in each dialect.
# Kept compact but syntactically real so a curious viewer sees genuine JSON.

OPENAI_CHAT = [
    '{"model":"gpt-5","messages":[',
    '  {"role":"system","content":"You translate text."},',
    '  {"role":"user","content":"Translate \'hello\' to French."}],',
    '"temperature":0.2,"max_completion_tokens":256,"stream":true}',
]

ANTHROPIC_MESSAGES = [
    '{"model":"claude-opus-4-6","max_tokens":256,',
    '"system":"You translate text.","messages":[',
    '  {"role":"user","content":[{"type":"text",',
    '   "text":"Translate \'hello\' to French."}]}],',
    '"thinking":{"type":"enabled","budget_tokens":1024}}',
]

GOOGLE_GENAI = [
    '{"contents":[{"role":"user","parts":[',
    '  {"text":"Translate \'hello\' to French."}]}],',
    '"systemInstruction":{"parts":[{"text":"You translate text."}]},',
    '"generationConfig":{"temperature":0.2,"maxOutputTokens":256,',
    '  "thinkingConfig":{"thinkingBudget":1024}}}',
]

# Ordered top→bottom on the stone, mirroring the real artifact's three bands.
SCRIPTS = [
    ("openai_chat", OPENAI_CHAT),
    ("anthropic", ANTHROPIC_MESSAGES),
    ("google", GOOGLE_GENAI),
]
