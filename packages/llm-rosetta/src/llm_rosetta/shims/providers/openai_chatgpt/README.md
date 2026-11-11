# openai_chatgpt — ChatGPT plan lane

`openai_chatgpt--open_responses` serves the ChatGPT plan allowance
(Go/Plus/Pro), not Platform API billing. It speaks the same Open Responses
wire format as `openai--open_responses`; the credential, the model list and
the accepted request surface differ.

Full user documentation:
<https://llm-rosetta.readthedocs.io/en/latest/guide/providers/openai_chatgpt/>

## Credential

- No `api_key_env`: the credential is minted by `auth.py` beside this file.
- Sign in from **Providers → Auth → Start browser sign-in** in the admin
  panel (the gateway returns an authorization URL; the operator pastes the
  callback URL back), or produce a record with
  `llm-rosetta-gateway credential <provider> --out record.json` and install it.
- Starting the browser sign-in again while connected is a re-login: the new
  challenge supersedes a pending one and a completed callback replaces the
  record.
- Access tokens live 3600 s. `connection.token_refresh_interval` is 2880 s so
  the scheduled refresher renews twelve minutes early; a reactive 401 refresh
  covers the rest. Refresh is asynchronous (`asyncio`; blocking HTTP runs on a
  worker thread, commands use an async subprocess).

## Model list

The route answers `{models: [{slug, visibility, …}]}` and the shim keeps the
entries marked `visibility: "list"`. That is the signed-in account's
entitlement, not the Platform API catalogue: expect fewer models than
`openai--open_responses`, and a different set per plan or app grant. Fetch the
list again after a plan change.

## Request surface

`store: false` is forced; `previous_response_id` and the fields listed in
`llm/open_responses/transforms.py` are stripped. Streaming is required — a
non-streaming Responses client is refused with `400` naming `stream=true`.
Only the `web_search` hosted tool is declared. The plan allowance is in-band
only: the probe classifies refusals and is never polled.
