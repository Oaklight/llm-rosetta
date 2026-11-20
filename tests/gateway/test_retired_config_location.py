"""A retired name in a config says which file, which key and what to run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from llm_rosetta import retired_names
from llm_rosetta_gateway.config.errors import RetiredConfigName
from llm_rosetta_gateway.config.loader import load_gateway_config

URL = "http://upstream.invalid/v1"
ROOT = Path(__file__).resolve().parent.parent.parent


def _doc(**sections: Any) -> dict[str, Any]:
    document: dict[str, Any] = dict(
        config_version=5,
        providers=dict(p=dict(type="chat_completions", base_url=URL, api_key="k")),
        models=dict(),
    )
    document.update(sections)
    return document


def _write(tmp_path: Path, document: dict[str, Any]) -> Path:
    path = tmp_path / "config.jsonc"
    path.write_text(json.dumps(document))
    return path


def _error(tmp_path: Path, document: dict[str, Any]) -> RetiredConfigName:
    path = _write(tmp_path, document)
    with pytest.raises(RetiredConfigName) as info:
        load_gateway_config(str(path))
    return info.value


def test_a_retired_capability_names_file_model_and_index(tmp_path: Path) -> None:
    doc = _doc(models=dict(m=dict(provider="p", capabilities=["embedding"])))
    error = _error(tmp_path, doc)
    message = str(error)
    assert isinstance(error, ValueError)
    assert message.startswith(f"{tmp_path / 'config.jsonc'}: ")
    assert "models.m.capabilities[0]" in message
    assert "'embedding' was renamed to 'embeddings'" in message
    assert message.endswith(retired_names.MIGRATE_HINT)
    assert error.where == "models.m.capabilities[0]"
    assert error.path == str(tmp_path / "config.jsonc")


def test_a_retired_model_type_names_the_type_key(tmp_path: Path) -> None:
    doc = _doc(models=dict(m=dict(provider="p", type="decision")))
    error = _error(tmp_path, doc)
    assert error.where == "models.m.type"
    assert "'decision' was renamed to 'decisions'" in str(error)


def test_a_retired_provider_type_names_the_provider_and_slot(tmp_path: Path) -> None:
    doc = _doc(providers=dict(acme=dict(type="openai_chat", base_url=URL)))
    message = str(_error(tmp_path, doc))
    assert "providers.acme.type" in message
    assert "'openai_chat' was renamed to 'chat_completions'" in message
    assert message.endswith(retired_names.MIGRATE_HINT)


def test_a_retired_shim_names_the_shim_slot(tmp_path: Path) -> None:
    doc = _doc(providers=dict(acme=dict(shim="volcengine--openai_chat")))
    assert "providers.acme.shim" in str(_error(tmp_path, doc))


def test_a_provider_key_read_as_a_name_is_located(tmp_path: Path) -> None:
    doc = _doc(providers=dict([("openai_chat", dict(base_url=URL))]))
    assert _error(tmp_path, doc).where == "providers.openai_chat"


def test_a_retired_provider_key_names_the_provider_entry(tmp_path: Path) -> None:
    provider = dict(type="chat_completions", base_url=URL, embedding_format="openai")
    error = _error(tmp_path, _doc(providers=dict(p=provider)))
    assert error.where == "providers.p.embedding_format"
    assert "'embedding_format' was renamed to 'embeddings_format'" in str(error)


def test_a_retired_top_level_section_names_the_key(tmp_path: Path) -> None:
    error = _error(tmp_path, _doc(embedding_providers=dict()))
    assert error.where == "embedding_providers"
    assert str(error).endswith(retired_names.MIGRATE_HINT)


@pytest.mark.parametrize("value", ["openai", "openai_chat"])
def test_a_retired_face_shim_is_refused_with_its_location(
    tmp_path: Path, value: str
) -> None:
    provider = dict(type="chat_completions", base_url=URL, embeddings_shim=value)
    error = _error(tmp_path, _doc(providers=dict(p=provider)))
    assert error.where == "providers.p.embeddings_shim"
    assert f"'{value}' was renamed" in str(error)
    assert str(error).endswith(retired_names.MIGRATE_HINT)


def test_a_face_shim_typo_gets_no_rename_hint(tmp_path: Path) -> None:
    provider = dict(type="chat_completions", base_url=URL, embeddings_shim="opnai")
    path = _write(tmp_path, _doc(providers=dict(p=provider)))
    # The typo disables the endpoint (a warning) and is not a retired name.
    assert load_gateway_config(str(path)).providers is not None


def test_a_current_config_still_loads(tmp_path: Path) -> None:
    doc = _doc(models=dict(m=dict(provider="p", capabilities=["text"])))
    assert "m" in load_gateway_config(str(_write(tmp_path, doc))).models


def test_located_keeps_what_is_already_known() -> None:
    error = RetiredConfigName("x", where="a.b").located(where="c", path="f")
    assert (error.where, error.path) == ("a.b", "f")
    assert str(error) == "f: a.b: x"


def test_the_cli_prints_one_line_and_exits_1(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _doc(models=dict(m=dict(provider="p", capabilities=["embedding"]))),
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT / "packages/llm-rosetta/src"),
            str(ROOT / "packages/llm-rosetta-gateway/src"),
        ]
    )
    result = subprocess.run(
        [sys.executable, "-m", "llm_rosetta_gateway", "--config", str(path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 1, result.stderr
    assert "Traceback" not in result.stderr
    lines = [line for line in result.stderr.splitlines() if line.strip()]
    assert len(lines) == 1, result.stderr
    line = lines[0]
    assert line.startswith("llm-rosetta-gateway: error: ")
    assert str(path) in line
    assert "models.m.capabilities[0]" in line
    assert line.endswith(retired_names.MIGRATE_HINT)
