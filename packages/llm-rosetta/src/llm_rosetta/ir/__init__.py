"""The IR layer: primitives, tools and passes around the typed IR.

``llm_rosetta.ir.model`` remains the one home for the typed IR and keeps its
public path.  This package holds the logic that operates on IR values:

====================  ===================================================
Sub-package           Holds
====================  ===================================================
``ir.primitives``     Body and IR transforms, plus the stateless helpers
                      they need (image limits, cache breakpoints, system
                      message hoisting, tool groups, truncation, parallel
                      call unwinding).
``ir.tools``          Stateless tool helpers: intrinsic-kind inspection,
                      call ids, error markers, orphan repair, batch ids,
                      allow-lists, strict mode.
``ir.passes``         Named passes that adapt an IR input to a model's
                      declared limits (intrinsic history, tool names,
                      description limits, request constraints).
====================  ===================================================

Dependency direction::

    types.ir  <-  ir.primitives  <-  ir.passes  <-  converters
                  ir.tools      <-

No module in ``ir`` imports anything from ``converters`` — the direction is
enforced by ``tests/test_ir_import_boundary.py``.  ``ConversionContext`` and
``StreamContext`` live in ``ir.context`` (``converters.base.context``
re-exports them).  A pass does not use the shim contract: the passes that read
a shim's declarations (reasoning, vision, custom and intrinsic tools) are
``pipeline.enforcement``, which sits above ``contracts``.

:mod:`llm_rosetta.contracts.transform` re-exports the transform types and
primitives a shim uses, and the modules that used to live under
``converters/base/helpers`` and ``converters/base/tools`` re-export from their
new home rather than keeping a copy.

This ``__init__`` deliberately imports nothing: ``llm_rosetta.shims`` imports
``llm_rosetta.contracts.transform``, and eager sub-package imports here would
make the ``ir`` import order matter to the contract layer.
"""

from __future__ import annotations
