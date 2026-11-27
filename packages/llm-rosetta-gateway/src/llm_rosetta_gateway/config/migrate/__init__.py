"""Config-file migrations: bring an existing config up to what the code expects.

- :mod:`.apply` — the load-time steps, recorded by ``config_version`` in the
  document itself.
- :mod:`.migrations` — the versioned config-schema migrations.
- :mod:`.names` — the guard that rejects a retired provider or shim name.
- :mod:`.migrate_0140` — the ``llm-rosetta gateway migrate`` tool, which edits a
  config file as text (:mod:`.jsonc_spans`) and installs the vendors that left
  core (:mod:`.migrate_presets`).

They read the config's own parser, so they live with it rather than with the
maintenance steps of the stores below the config layer.
"""
