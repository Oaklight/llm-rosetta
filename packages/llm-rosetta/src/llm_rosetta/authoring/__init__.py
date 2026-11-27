"""Extension authoring: the tools that create and check an extension folder.

``llm-rosetta ext new`` and ``llm-rosetta ext check`` come from
:mod:`~llm_rosetta.authoring.ext_scaffold`, which the CLI hands to the
extension package as :class:`~llm_rosetta.extensions.ExtVerb` records.
:mod:`~llm_rosetta.authoring.scaffold` and ``_shim_template`` are the shim-tree
scaffolder behind ``scripts/dev/new_shim.py``.  Internal, except
:func:`~llm_rosetta.api.run_ext_check`, which a generated extension's own test
calls through :mod:`llm_rosetta.api`.
"""
