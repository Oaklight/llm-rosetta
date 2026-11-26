Live streams captured on 2026-11-26 (#1749): one `streamGenerateContent`, one
Interactions stream, one Anthropic `messages` stream (haiku-4-5) and one OpenAI
Chat stream with `include_usage`.  Ids and signatures are replaced by
placeholders; the text is only the numbers 1..40.  The usage objects are the
captured ones, byte for byte.  `tests/test_usage_live_streams.py` reads them.

Usage shapes: Google repeats the running `usageMetadata` on every chunk;
Interactions has `usage` only on `interaction.completed`; Anthropic's
`message_delta` repeats the `message_start` input beside the output so far
(24/1, then 24/83); OpenAI sends `usage` once, on the last chunk.
