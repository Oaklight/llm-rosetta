---
title: Upgrading
---

# Upgrading the gateway

The gateway is its own distribution, `llm-rosetta-gateway`. It depends on the
conversion core, `llm-rosetta`. Both are built from one repository and share a
version number, so "which versions go together" has one answer: the same one.

## Version policy

- **Lockstep until 1.0.** A release `v0.15.0` publishes `llm-rosetta 0.15.0`
  and `llm-rosetta-gateway 0.15.0`. The release workflow refuses a tag whose
  two `__version__` values disagree.
- **Same-minor compatibility.** The gateway declares
  `llm-rosetta>=0.15.0.dev0,<0.16`: core patch releases are accepted without a
  new gateway, while a new core minor line requires a matching gateway release.
- **Dev builds** (`0.14.0.devN`) go to TestPyPI and the Docker `-dev` tags.
  They are for testing the release line, not for production.

## Upgrade with Docker

The image installs both packages at the same version, so upgrading the image
upgrades the core with it — there is no second step.

```bash
docker pull oaklight/llm-rosetta-gateway:0.15.0-python
docker compose pull && docker compose up -d
```

- Pin an exact tag in production. `:latest-python` follows the newest release;
  it is convenient for a test box and surprising in production.
- Never run the `-dev` tags in production: they move with the development
  branch, not with a release.

## Upgrade with pip

```bash
pip install -U "llm-rosetta[gateway]"    # gateway + its compatible core
pip install -U llm-rosetta-gateway       # gateway; core moves only if the range requires it
pip install -U llm-rosetta               # core only (library users)
```

`llm-rosetta[gateway]` is a one-release meta alias: it resolves the gateway
package and lets the gateway pull the core version it was tested against. The
idiomatic "upgrade the whole stack" command is therefore the first one.

If you installed the gateway on its own, upgrading it is enough: pip resolves
the newest core the gateway's range allows. Upgrading the core on its own is
safe while the gateway's range accepts it; if it does not, pip reports the
conflict instead of producing a pair that was never tested.

## Check what you run

```bash
llm-rosetta-gateway --version
# llm-rosetta-gateway 0.15.0 (llm-rosetta 0.15.0)
```

```bash
curl -s http://localhost:8765/health | jq .versions
# {"gateway": "0.15.0", "core": "0.15.0"}
```

The startup banner prints the same pair, and **Settings** in the admin panel
shows the gateway version — with `· core v…` appended when the two differ.

## Rolling back

- **pip:** install the pair explicitly, so the resolver cannot keep a newer
  core with an older gateway:

  ```bash
  pip install "llm-rosetta-gateway==0.14.0" "llm-rosetta==0.14.0"
  ```

- **Docker:** change the image tag back. The image carries both packages, so
  the rollback is atomic.

Configuration files are not versioned by the package: the gateway migrates an
older config on load (and keeps a `.pre-migration.bak` beside it), so a
rollback to a build that predates a migration is the case to check before
downgrading — keep the backup the upgrade left behind.
