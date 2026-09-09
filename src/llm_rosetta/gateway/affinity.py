"""API key affinity — deterministic key selection for prompt cache locality.

Instead of round-robin, selects an upstream key based on
``hash(client_token + message_prefix) % num_keys`` so the same
conversation from the same client always hits the same upstream key.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def extract_prefix_from_ir(
    ir_request: dict[str, Any],
    max_messages: int = 2,
) -> str:
    """Extract a stable string prefix from an IR request.

    Builds a fingerprint from ``system_instruction`` plus the first
    *max_messages* entries in ``messages``.  Returns ``""`` when the
    IR is empty or extraction fails (e.g. passthrough mode produces
    ``{}``).
    """
    if not ir_request:
        return ""

    parts: list[str] = []

    # system_instruction is list[TextPart] in canonical IR
    sys_instr = ir_request.get("system_instruction")
    if sys_instr:
        parts.append(_stable_str(sys_instr))

    messages = ir_request.get("messages")
    if messages:
        for msg in messages[:max_messages]:
            parts.append(_stable_str(msg))

    return "\n".join(parts) if parts else ""


def compute_affinity_index(
    client_key_hash: str,
    message_prefix: str,
    num_keys: int,
) -> int | None:
    """Compute a deterministic key index from client identity + message prefix.

    Returns ``None`` when affinity cannot be determined (missing inputs
    or single key), signalling the caller to fall back to round-robin.
    """
    if num_keys <= 1 or not client_key_hash or not message_prefix:
        return None
    combined = f"{client_key_hash}\n{message_prefix}"
    h = hashlib.sha256(combined.encode()).digest()
    return int.from_bytes(h[:4], "big") % num_keys


def _stable_str(value: Any) -> str:
    """Convert a value to a stable string for hashing."""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)
