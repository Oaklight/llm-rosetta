"""Build real extension wheels for tests: zip files, no network, no build backend."""

from __future__ import annotations

import hashlib
import io
import textwrap
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION


def plugin_yaml(
    ext_id: str,
    version: str,
    shims: Sequence[str] = (),
    *,
    transports: Sequence[str] = (),
    requires_extensions: Sequence[str] = (),
    surface: int = SURFACE_VERSION,
    ir: int = IR_VERSION,
    maturity: str = "beta",
) -> str:
    """The text of a valid ``plugin.yaml``."""
    lines = [
        f"id: {ext_id}",
        f"name: {ext_id} test extension",
        f"version: {version}",
        f"maturity: {maturity}",
        "requires:",
        f"  surface: {surface}",
        f"  ir: {ir}",
    ]
    if requires_extensions:
        lines.append(f"  extensions: [{', '.join(requires_extensions)}]")
    lines.append("provides:")
    if shims:
        lines.append(f"  shims: [{', '.join(shims)}]")
    if transports:
        lines.append(f"  transports: [{', '.join(transports)}]")
    return "\n".join(lines) + "\n"


def shim_yaml(name: str, marker: str = "v", transport: str | None = None) -> str:
    """A minimal chat shim named *name* (``<id>--<format>``)."""
    lane = name.split("--", 1)[0]
    fmt = name.split("--", 1)[1] if "--" in name else "chat_completions"
    return textwrap.dedent(
        f"""\
        name: {name}
        format: {fmt}
        model_type: llm
        connection:
          base_url: https://{lane}.{marker}.example/v1
          api_key_env: ACME_API_KEY
        """
    ) + (f"  transport: {transport}\n" if transport else "")


def leaf_dir(ext_id: str, shim: str) -> str:
    """The folder of a shim's ``provider.yaml`` inside its extension folder.

    ``<id>--<format>`` lives at ``llm/<format>``; a lane shim
    ``<id>_<lane>--<format>`` at ``<lane>/llm/<format>``.
    """
    head, _, fmt = shim.partition("--")
    fmt = fmt or "chat_completions"
    if head == ext_id:
        return f"llm/{fmt}"
    return f"{head[len(ext_id) + 1 :]}/llm/{fmt}"


def build_wheel(
    ext_id: str,
    version: str,
    shims: Sequence[str],
    *,
    marker: str | None = None,
    members: Mapping[str, bytes | str] | None = None,
    transport_of: Mapping[str, str] | None = None,
    manifest: str | None = None,
    requires_python: str | None = None,
    **plugin: Any,
) -> bytes:
    """A pure-Python extension wheel holding ``llm_rosetta_ext/<id>/``.

    Every ``<id>--<format>`` shim gets ``llm/<format>/provider.yaml``.  *members*
    adds files under the extension folder (``transport.py``, ``auth.py``); a
    name starting with ``../`` is written at the wheel root instead.
    *transport_of* maps a shim name to the ``connection.transport`` it declares;
    *requires_python* is written as the wheel's ``Requires-Python``.
    """
    base = f"llm_rosetta_ext/{ext_id}"
    metadata = "Name: x\n" + (
        f"Requires-Python: {requires_python}\n" if requires_python else ""
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            f"{base}/plugin.yaml",
            manifest
            if manifest is not None
            else plugin_yaml(ext_id, version, shims, **plugin),
        )
        for shim in shims:
            zf.writestr(
                f"{base}/{leaf_dir(ext_id, shim)}/provider.yaml",
                shim_yaml(shim, marker or version, (transport_of or {}).get(shim)),
            )
        for name, payload in (members or {}).items():
            path = name[3:] if name.startswith("../") else f"{base}/{name}"
            zf.writestr(path, payload)
        zf.writestr(f"{ext_id}-{version}.dist-info/METADATA", metadata)
        zf.writestr(f"{ext_id}-{version}.dist-info/RECORD", f"{base}/plugin.yaml,,\n")
    return buf.getvalue()


def tree_wheel(
    ext_id: str,
    version: str,
    root: Path,
    shims: Sequence[str],
    **plugin: Any,
) -> bytes:
    """Pack the directory *root* as ``llm_rosetta_ext/<id>/`` and add a manifest."""
    base = f"llm_rosetta_ext/{ext_id}"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            f"{base}/plugin.yaml",
            plugin_yaml(ext_id, version, shims, **plugin),
        )
        for path in sorted(root.rglob("*")):
            if "__pycache__" in path.parts or not path.is_file():
                continue
            zf.writestr(
                f"{base}/{path.relative_to(root).as_posix()}", path.read_bytes()
            )
        zf.writestr(f"{ext_id}-{version}.dist-info/METADATA", "Name: x\n")
        zf.writestr(f"{ext_id}-{version}.dist-info/RECORD", f"{base}/plugin.yaml,,\n")
    return buf.getvalue()


def sha256(data: bytes) -> str:
    """The hex digest a manifest record carries."""
    return hashlib.sha256(data).hexdigest()
