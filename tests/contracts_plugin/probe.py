"""Run in a fresh interpreter by ``tests/test_contracts_partial_init.py``.

Prints what the extension registered, as JSON.
"""

import json
from typing import Any
import logging

failures = []


class _Collect(logging.Handler):
    def emit(self, record):
        message = record.getMessage()
        if record.levelno >= logging.WARNING and "c1657" in message:
            failures.append(message)


logging.getLogger().addHandler(_Collect())

import llm_rosetta  # noqa: E402
from llm_rosetta.shims import get_shim  # noqa: E402

shim = get_shim("c1657acme")
result: dict[str, Any] = dict(
    core=llm_rosetta.__file__, registered=shim is not None, warnings=failures
)
if shim is not None:
    result["pre"] = list(map(repr, shim.pre_ir_transforms))
    result["ir"] = len(shim.ir_transforms)
print(json.dumps(result))
