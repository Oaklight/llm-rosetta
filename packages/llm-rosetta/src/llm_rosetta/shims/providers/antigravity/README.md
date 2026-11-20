# antigravity — Google Antigravity through the local `agy` CLI

`antigravity--acp` runs `agy acp` on the gateway host and speaks the Agent
Client Protocol to it. `agy` owns the Google sign-in; the gateway never sees a
token, and no credential file or keychain entry of `agy` is ever read.

## Credential

The credential is not a key. `auth.py` (beside this file) asks `agy` whether it
is signed in: it spawns `agy acp`, speaks `initialize` and `session/new`, and
reads the answer. A session that opens is signed in; a refusal that names a
sign-in step is signed out; any other failure is reported as a failure.
Signed-in is re-asked every `connection.token_refresh_interval` (900 s, one
`agy` process per check). A signed-out entry is not routed to and `/health/ready`
reports it not ready, like any provider that waits for a sign-in.

There is no static key and no `token_command` for this lane: it sends no
credential. The admin panel therefore offers one source card, the sign-in.

## Sign in from the admin panel

**Providers → Edit → Auth → Sign in on the gateway host** runs
`agy acp --login` on the machine that runs the gateway, as a subprocess the
panel observes:

- The browser step happens on the gateway host. The panel shows any `https`
  URL `agy` prints, so an operator on another machine can open it.
- Progress is polled. When `agy` exits 0 the exchange verifies as above; the
  gateway then re-seeds the provider and replaces the entry's agent processes,
  so the next request answers for the account that is signed in now.
- One sign-in runs at a time per entry, for at most 300 s. **Cancel sign-in**
  (or the deadline) kills the process group.
- The argv is fixed: the entry's `acp_argv` (default `agy acp`), `--login`, and
  the profile flag below. Output is clipped, token-shaped text is redacted, and
  a URL carrying a `code` or token parameter is never shown.

The same login on the command line: `agy acp --login`.

## One entry, one account

By default an entry uses the login of the gateway host, exactly as before:
two such entries are one upstream account (they report the same identity).

To use another account, add another provider entry and give it its own profile
in the **agy profile** field (stored as `auth_fields.agy_profile`):

```jsonc
"providers": {
  "agy":  { "type": "antigravity--acp" },                                    // the host's login
  "agy2": { "type": "antigravity--acp", "auth_fields": { "agy_profile": "separate" } }
}
```

A `separate` entry runs `agy acp --gemini-dir <data_dir>/shim-auth/<name>/agy`,
and signs in with `agy acp --login --gemini-dir …`, so its state lives under the
gateway data directory, in a `0700` directory named after the entry. Each entry
is a different agent process (the transport pool is keyed by provider name).
**Sign out** of a separate entry removes that directory; it revokes nothing
upstream. Sign-out of an entry on the host login is refused: that login belongs
to `agy` and to every other tool using it.

## What was and was not measured

Measured against `agy` 1.5.13 (no login performed):

- `--gemini-dir <dir>` relocates every file `agy` keeps (`config/`,
  `antigravity-cli/`); `--app-data-dir` takes a *relative* path (under
  `~/.gemini`) and rejects an absolute one.
- On the development host a fresh `--app-data-dir`, a fresh `HOME`, and a fresh
  `HOME` with no D-Bus session or keyring socket were all still authenticated:
  the host's login lives in a store that none of those flags reach. So
  `--app-data-dir` and `HOME` do **not** isolate a login there, and a signed-out
  `agy` could not be produced to observe its refusal.

Not measured (needs a human): where a login performed under `--gemini-dir`
lands, what `agy acp --login` prints, and the wording of the refusal a
signed-out `agy` gives. The exchange treats the wording the transport already
uses (`sign in`, `authenticate`, `login`, …) as the signal. This is why the
separate profile is opt-in.
