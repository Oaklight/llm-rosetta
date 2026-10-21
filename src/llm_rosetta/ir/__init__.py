"""The IR layer: primitives, tools and passes around the typed IR.

``llm_rosetta.types.ir`` remains the one home for the typed IR and keeps its
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
``ir.passes``         Named capability passes that adapt an IR input to a
                      model's capabilities (vision, reasoning, custom
                      tools, intrinsic history, tool names, description
                      limits).
====================  ===================================================

Dependency direction::

    types.ir  <-  ir.primitives  <-  ir.passes  <-  converters
                  ir.tools      <-

A module in ``ir.primitives`` or ``ir.tools`` imports nothing from
``converters`` — the direction is enforced by ``tests/test_ir_import_boundary.py``.
A pass may use the converter *base* contracts (``ConversionContext``) and the
shim contract, because a pass is where a target's declared capabilities enter
the IR; it must not import a converter family.

``llm_rosetta.transforms`` and ``llm_rosetta.capabilities`` are facades over
this package, and the modules that used to live under
``converters/base/helpers`` and ``converters/base/tools`` re-export from their
new home rather than keeping a copy.

This ``__init__`` deliberately imports nothing: ``llm_rosetta.shims`` imports
``llm_rosetta.transforms``, and eager sub-package imports here would pull
``ir.passes`` (which imports the shim contract) back into that cycle.
"""

from __future__ import annotations
