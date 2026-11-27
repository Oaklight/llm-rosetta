# ALCF

ALCF (Argonne Leadership Computing Facility) Inference Service is supported through three shim endpoints, one per compute cluster. All use the OpenAI Chat Completions compatible API.

!!! warning "Ships as a standalone package"
    Since 0.14.0 the three ALCF shims are no longer built into LLM-Rosetta; they
    ship in the `alcf` extension of
    [llm-rosetta-ext-argonne](https://github.com/Oaklight/llm-rosetta-ext-argonne).
    Declare it with `shims.install` and the names below keep working.
    **Gateway → Shims → Available** offers the same entry as a one-click install.

    ```jsonc
    "shims": { "install": ["argonne:alcf"] }
    ```

## Endpoints

| Shim Name | Base Type | Default Base URL | API Key Env | Hardware |
|:---|:---|:---|:---|:---|
| `alcf_metis--chat_completions` | `chat_completions` | `https://inference-api.alcf.anl.gov/resource_server/metis/api/v1` | (Globus sign-in) | SambaNova SN40L |
| `alcf_minerva--chat_completions` | `chat_completions` | `https://inference-api.alcf.anl.gov/resource_server/minerva/api/v1` | (Globus sign-in) | NVIDIA B200 |
| `alcf_sophia--chat_completions` | `chat_completions` | `https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1` | (Globus sign-in) | vLLM on A100 |

The three share one admin-panel top level: each declares `group: alcf`, so the
provider-type picker shows a single `alcf` group whose rows read
`metis · chat_completions`, `minerva · chat_completions` and
`sophia · chat_completions`. The older deployment-first spellings (`alcf--metis`, `alcf--minerva`,
`alcf--sophia`) are not registered by the standalone package; use the canonical
names above.

## Reasoning Support

Sophia models marked with **R** in the [ALCF model list](https://docs.alcf.anl.gov/services/inference-endpoints/#available-models) support reasoning (e.g. `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `google/gemma-4-31B-it`, `google/gemma-4-E4B-it`).

| Field | Value |
|:---|:---|
| Effort field | `reasoning_effort` |
| Effort range | Full IR ladder (`minimal` → `max`) |

Metis and Minerva do not currently have reasoning-capable models.

## Capability Flags

| Flag | `alcf_sophia--chat_completions` | `alcf_minerva--chat_completions` | `alcf_metis--chat_completions` |
|:---|:---:|:---:|:---:|
| Custom tools | ✅ | ✅ | — |

!!! note "Metis tool calling"
    SambaNova has [known tool-call sanitization issues](https://docs.alcf.anl.gov/services/inference-endpoints/#metis-tool-calling) that can cause incomplete function calls. Tool support is intentionally not declared for Metis.

## Transforms

All three clusters share a common set of transforms:

| Type | Transform | Purpose |
|:---|:---|:---|
| Post-IR | `strip_fields("logprobs", "top_logprobs")` | Removes unsupported request fields |
| Post-IR | `downgrade developer → system` | Converts developer role to system role |
| Post-IR | `default null content → ""` | Prevents null content errors |
| Post-IR | `default_tool_description()` | Fills empty tool descriptions to avoid upstream errors |
| IR | `hoist_late_system_messages()` | Moves late system messages to conversation start for cache stability |

!!! note "Metis additional transform"
    The Metis cluster additionally strips `parallel_tool_calls` from requests, as the SambaNova backend does not support parallel tool calling.

!!! note "Minerva response transforms"
    Minerva applies `rewrite_harmony_tool_calls()` on responses to fix intermittent cases where `inkling-bf16` emits tool calls as plain text instead of structured `tool_calls` fields.

All three clusters also export a `model_list_transform` that normalizes the model listing response.

## Authentication

ALCF uses Globus OAuth. The inference service takes a Globus access token minted from a grant that a person obtains once by signing in at `auth.globus.org`; the refresh token that comes back is spent on every later mint, without a browser. The refresh token expires after **30 days** because of ALCF's session policy; after that, an interactive sign-in is required.

The exchange ships **inside the extension**: the `alcf` extension carries `auth.py` beside its vendor manifest, so the three cluster shims authenticate alike. The provider is just its shim; there is no `token_command` to write and no `ALCF_API_KEY` to export. The gateway asks the exchange for a credential at startup, on the shim's interval and after an upstream 401.

### Sign in

Run the sign-in on a machine that has a browser, then hand the record to the gateway ([`credential <provider>`](../../gateway/cli.md#credential-provider)). The command prints a Globus authorization URL and takes the authorization code you paste back:

```bash
llm-rosetta gateway credential alcf-sophia --out record.json
```

In the admin panel, **Install a record** on the provider's Auth pane takes that record. The provider may be saved before it has one.

The grant is stored where the vendor's own tooling keeps it, `~/.globus/app/<client_id>/inference_app/tokens.json`, and written back in that layout. An installation that already signed in with `alcf-ai` therefore needs no second sign-in, and a container that mounts `~/.globus` is configured once for both.

### 30-day session expiry

!!! warning "Session policy limitation"
    ALCF's Globus session policy enforces a 30-day lifetime. When the session expires, the refresh token stops working and the gateway enters a 401 retry loop. **Interactive re-login is required**: this cannot be automated because ALCF requires browser-based institutional SSO.

    Watch for `consecutive_failures` escalating in the admin panel's token status, and sign in again when it happens.

    Tracking: [#687](https://github.com/Oaklight/llm-rosetta/issues/687)

## Gateway Configuration

```jsonc
{
  "providers": {
    "alcf-sophia": {
      "shim": "alcf_sophia--chat_completions"
    }
  }
}
```

To change the refresh interval, set `token_refresh_interval` on the provider. To pool several ALCF accounts, declare one provider per account: the gateway gives each provider its own state directory.

### Docker deployment

The exchange lives inside the extension, which the gateway installs from the `argonne` source, so only the token file needs mounting:

```yaml
# docker-compose.yaml
services:
  llm-rosetta-gateway:
    image: oaklight/llm-rosetta-gateway
    volumes:
      - ./config:/config
      - ~/.globus:/home/appuser/.globus              # Globus token file (rw for refresh)
```

The image keeps installed extensions in `/config/extensions` (`LLM_ROSETTA_EXTENSIONS_DIR`), so they survive the container. The container runs as `${PUID:-1000}` (Compose's `user:` directive, default the image's `appuser`, uid 1000), so the mounted `~/.globus` must be writable by that uid for the refresh token to be updated in place. If your host uid differs, set `PUID`/`PGID` in the compose file and make the `.globus` directory belong to the same uid.

See [`docker/docker-compose.yaml`](https://github.com/Oaklight/llm-rosetta/blob/master/docker/docker-compose.yaml) for the full example.

## Notes

- Each cluster runs different hardware and may host different models. Use the `models_path` in the shim configuration to query available models on each cluster.
