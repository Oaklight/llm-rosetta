# ALCF

ALCF (Argonne Leadership Computing Facility) Inference Service is supported through three shim endpoints, one per compute cluster. All use the OpenAI Chat Completions compatible API.

!!! warning "Ships as a standalone package"
    Since 0.14.0 the three ALCF shims are no longer built into LLM-Rosetta; they
    ship in the
    [llm-rosetta-shim-alcf](https://github.com/Oaklight/llm-rosetta-shim-alcf)
    package. Declare it with `shims.presets` (or the equivalent
    `shims.remotes` entry) and the names below keep working.
    **Settings → Shim Sources** offers the same entry as a one-click preset.

    ```jsonc
    "shims": { "presets": ["alcf"] }
    ```

## Endpoints

| Shim Name | Base Type | Default Base URL | API Key Env | Hardware |
|:---|:---|:---|:---|:---|
| `alcf_metis--chat_completions` | `chat_completions` | `https://inference-api.alcf.anl.gov/resource_server/metis/api/v1` | `ALCF_API_KEY` | SambaNova SN40L |
| `alcf_minerva--chat_completions` | `chat_completions` | `https://inference-api.alcf.anl.gov/resource_server/minerva/api/v1` | `ALCF_API_KEY` | NVIDIA B200 |
| `alcf_sophia--chat_completions` | `chat_completions` | `https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1` | `ALCF_API_KEY` | vLLM on A100 |

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

ALCF uses Globus OAuth for authentication. Access tokens are short-lived (~48 hours) and must be refreshed periodically. The refresh token itself expires after **30 days** due to ALCF's session policy — after that, interactive re-login is required.

The Globus exchange ships **inside the shim package**: `llm-rosetta-shim-alcf` carries the zero-dependency (stdlib only) helper at `llm_rosetta_shim_alcf.providers.alcf.token`, and every `alcf_<cluster>--chat_completions` shim declares it as its `connection.token_command`. There is nothing to download and no `token_command` to write in your config — the gateway runs the exchange at startup, on the shim's interval and after an upstream 401.

### Initial login

Run on any machine with a browser (or copy the URL to a machine that has one):

```bash
python3 -m llm_rosetta_shim_alcf.providers.alcf.token --login
```

This opens a Globus authorization URL, prompts you to paste the authorization code, and saves the token to `~/.globus/app/<client_id>/inference_app/tokens.json`.

### Token management

```bash
# Print a fresh access token (auto-refreshes if expired):
python3 -m llm_rosetta_shim_alcf.providers.alcf.token

# Check token status and remaining lifetime:
python3 -m llm_rosetta_shim_alcf.providers.alcf.token --status

# Verify the token works against a cluster:
python3 -m llm_rosetta_shim_alcf.providers.alcf.token --probe sophia

# Force a refresh even if not expired:
python3 -m llm_rosetta_shim_alcf.providers.alcf.token --force
```

### Multi-user token pooling

Multiple ALCF users can pool their tokens for round-robin key rotation. The gateway runs the shipped command without arguments, so pooling needs one override — point `token_command` at `--tokens-dir`:

```bash
# Each user logs in to their own file:
python3 -m llm_rosetta_shim_alcf.providers.alcf.token --login --token-file /shared/alcf-tokens/alice.json
python3 -m llm_rosetta_shim_alcf.providers.alcf.token --login --token-file /shared/alcf-tokens/bob.json
```

```jsonc
{
  "providers": {
    "alcf-sophia": {
      "shim": "alcf_sophia--chat_completions",
      "token_command": [
        "python3", "-m", "llm_rosetta_shim_alcf.providers.alcf.token",
        "--tokens-dir", "/shared/alcf-tokens/"
      ]
    }
  }
}
```

The command prints one comma-separated string for all files; the gateway uses it as a key ring.

### 30-day session expiry

!!! warning "Session policy limitation"
    ALCF's Globus session policy enforces a 30-day lifetime. When the session expires, the refresh token stops working and the gateway enters a 401 retry loop. **Interactive re-login is required** — this cannot be automated because ALCF requires browser-based institutional SSO.

    Watch for `consecutive_failures` escalating in the admin panel's token status, and re-run `--login` when it happens.

    Tracking: [#687](https://github.com/Oaklight/llm-rosetta/issues/687)

## Gateway Configuration

=== "Static API Key"

    Obtain a token manually and set it as an environment variable:

    ```bash
    export ALCF_API_KEY=$(python3 -m llm_rosetta_shim_alcf.providers.alcf.token)
    ```

    ```jsonc
    {
      "providers": {
        "alcf-sophia": {
          "shim": "alcf_sophia--chat_completions",
          "api_key": "${ALCF_API_KEY}"
        }
      }
    }
    ```

    This requires restarting the gateway when the token expires.

=== "Shipped exchange (recommended)"

    Nothing to wire up — the shim declares the exchange, so the provider is just its shim:

    ```jsonc
    {
      "providers": {
        "alcf-sophia": {
          "shim": "alcf_sophia--chat_completions"
        }
      }
    }
    ```

    The gateway runs the command at startup, on the shim's declared interval and after an upstream 401 response. To change the interval without replacing the command, set `token_refresh_interval` on the provider.

=== "Override the command"

    A provider that names its own `token_command` replaces the shipped one — the multi-user pooling setup above, or a script of your own:

    ```jsonc
    {
      "providers": {
        "alcf-sophia": {
          "shim": "alcf_sophia--chat_completions",
          "token_command": ["python3", "/opt/alcf/token-from-vault.py"],
          "token_refresh_interval": 1800
        }
      }
    }
    ```

### Docker deployment

The refresh script lives inside the package, which the image installs, so only the token file needs mounting:

```yaml
# docker-compose.yaml
services:
  llm-rosetta-gateway:
    image: oaklight/llm-rosetta-gateway
    volumes:
      - ./config:/config
      - ~/.globus:/home/appuser/.globus              # Globus token file (rw for refresh)
```

Setup steps:

1. **Login on the host** (one-time): `python3 -m llm_rosetta_shim_alcf.providers.alcf.token --login`
2. **Start the container** with the volume mount above
3. **Add the provider** through the admin panel or `config.jsonc` — no `token_command`

The container's `appuser` (uid 1000) needs read-write access to `~/.globus` so the refresh token can be updated in place. If your host uid differs, set `PUID`/`PGID` in the compose file or `chown 1000:1000` the `.globus` directory.

See [`docker/docker-compose.yaml`](https://github.com/Oaklight/llm-rosetta/blob/master/docker/docker-compose.yaml) for the full example.

## Notes

- Each cluster runs different hardware and may host different models. Use the `models_path` in the shim configuration to query available models on each cluster.
