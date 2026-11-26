# Shim template

This directory is the single source of truth for a provider shim's shape.
`scripts/dev/new_shim.py` renders these files (`llm-rosetta ext new` renders an
extension folder, whose tree has this shape); the loader that reads the
result lives in `llm_rosetta/shims/_loader/`. Nothing here is a shim itself:
the leading `_` keeps the directory out of every scan.

Read this before hand-writing a shim. If the template and a shim disagree, the
template is right and the shim is a bug.

## What a shim may contain

The loader opens exactly these names. Anything else has no effect of its own —
with one qualifier worth keeping straight: a helper module is inert until one of
these files imports it. `transforms.py` may, so `_helpers.py` beside it is a
working private module; `__init__.py` is the name that stays unread, and a
`raise` in one never runs.

| File | Where | Required | Read as |
|:--|:--|:--|:--|
| `provider.yaml` | vendor root | no | vendor manifest: `connection:`, `logo`, `group`, `group_label`, `fields:` |
| `provider.yaml` | leaf | yes | the shim's own declaration |
| `auth.py` | vendor root | no | `EXCHANGE`, the credential exchange for every lane |
| `auth.py` | leaf | no | that leaf's exchange; refused when the vendor has one |
| `quota.py` | vendor root or leaf | no | `PROBES`, allowance readers |
| `transforms.py` | leaf | no | field-level transform tuples |

`auth.py` and `quota.py` are executed as standalone modules: they cannot use
relative imports and must be self-contained (the standard library is always
available). `transforms.py` may use relative imports — the loader registers a
package chain for it — so a helper module beside it works.

## Where a leaf lives

```
<vendor>/                    provider.yaml is a manifest, never a leaf
├── auth.py                  optional
├── quota.py                 optional
└── [<lane>/]                optional: several named instances of one vendor
    └── <type>/              llm | embeddings | rerank | decisions
        └── [<format>/]      only when that type holds more than one lane
            ├── provider.yaml
            ├── transforms.py
            ├── auth.py
            └── quota.py
```

Three rules decide the path.  The loader also reads two older shapes — a
leaf directly under the vendor, and a leaf under `<vendor>/<format>/` — so an
existing package keeps loading; they are tolerated, not the contract, and
the tree check reports both.

1. **The vendor directory is never a leaf.** `provider.yaml` there is a
   manifest, and every leaf sits below it.  A manifest carries `connection:`,
   `logo`, `group`, `group_label`, `fields:` and nothing that identifies a
   shim — adding `name`, `model_type` or `format` turns the vendor into a
   leaf, and every shim below it then loads as zero with one "missing 'name'
   or 'format'" line.
2. **A type directory is always present**, even when it holds a single lane.
3. **The format directory appears only when a second lane of that type
   exists.** The first lane is written as `<vendor>/<type>/provider.yaml`; the
   second moves the first into `<vendor>/<type>/<its format>/` and puts itself
   beside it. The scaffold does that move for you.

A lane is a named instance of the same vendor — ALCF's `metis`, `minerva` and
`sophia` are three clusters of one account, sharing one manifest and one
credential. A lane has no `provider.yaml` of its own; the vendor manifest is
inherited.

## Declare the credential inputs

A shim whose credential needs more than the key declares the extra inputs in
`fields:` — on the vendor manifest when every lane shares them, on a leaf to
add one or reword one for that lane.  The admin provider form renders one input
per entry, and one Save writes the key and these values into the provider entry
(under `auth_fields`), which is what the shim's `auth.py` reads back:

```yaml
fields:
  - key: email            # the record key, a plain identifier
    label: Account email  # required
    placeholder: you@example.gov
    help: The address the AskSage account key was issued to.
    required: true
```

The block is data: changing the form for a vendor is a change to the YAML, not
to the gateway or the panel.  Entries are validated at load — a missing `key`
or `label`, a duplicate key, or a non-boolean `required` is refused with the
file named — and the tree check reports the same finding before the tree is
installed.  Nothing declared here is a secret: the values live on the provider
entry and are echoed to the authenticated panel, so a credential belongs in
`api_key` or in the exchange's own record.

## Create a shim

```bash
# inside a checkout, into the built-in tree
python scripts/dev/new_shim.py together --format chat_completions

# a lane under an existing vendor
python scripts/dev/new_shim.py alcf --lane metis --format chat_completions

# an installable extension folder (a tree of this shape plus plugin.yaml)
llm-rosetta ext new together
```

Every run loads its own output through the real loader before it reports
success, so a shape no gateway could read fails at creation rather than at
startup. Check an extension folder with `llm-rosetta ext check <folder>`, or a bare tree with:

```bash
python -c "import sys; from pathlib import Path; from llm_rosetta.api import run_scaffold_check; sys.exit(run_scaffold_check(Path(sys.argv[1])))" src/llm_rosetta/shims/builtin
```

## How the template directory maps to a shim

The directories under `tree/` use fixed names so an editor can open them:
`tree/vendor/` becomes `<root>/<vendor>/`, `tree/vendor/type/` becomes
`<vendor>/<type>/`, and `tree/vendor/type/format/` becomes
`<vendor>/<type>/<format>/` — or `<vendor>/<type>/` when that type holds a
single lane, in which case the format level is dropped.

Every file below `tree/` carries a `.tmpl` suffix and the scaffold strips it
when writing. Without the suffix, ruff and the loader would treat a placeholder
file as the real thing.

## Placeholders

The scaffold renders these names. A line whose placeholder renders empty is
dropped, so an optional field can stay on one line in the template.

`{{vendor}}`, `{{group}}`, `{{group_label}}`, `{{logo}}`, `{{api_key_env}}`,
`{{name}}`, `{{model_type}}`, `{{format}}`, `{{base_url}}`.
