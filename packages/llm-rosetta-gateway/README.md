# llm-rosetta-gateway

An HTTP proxy that accepts requests in any supported LLM API format and
forwards them to any configured upstream provider, converting on the fly
(including streaming), with key rotation, routing and an admin panel.

This is the **gateway distribution**. It depends on `llm-rosetta` for the
conversion core. See the [project README](../../README.md) and the
[documentation](https://llm-rosetta.readthedocs.io).
