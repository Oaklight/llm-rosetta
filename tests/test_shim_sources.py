"""Tests for declarative external shim sources (directories + remote archives)."""

from __future__ import annotations

import hashlib
import io
import json
import socket
import tarfile
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from llm_rosetta.shims import get_shim, list_shims, register_shim
from llm_rosetta_gateway._vendor.httpclient import HTTPError
from llm_rosetta_gateway.shims import (
    ShimSourceError,
    rebuild_shims,
    resolve_remote,
)
from llm_rosetta_gateway.shims import sources as sources_mod
from llm_rosetta.shims.provider_shim import (
    ConnectionConfig,
    ProviderShim,
    _reset_registry,
)

PROVIDER_YAML = (
    "name: {name}\n"
    "base: chat_completions\n"
    "connection:\n"
    "  base_url: https://{slug}.example/v1\n"
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the shim registry before and after each test."""
    _reset_registry()
    sources_mod._source_shims.clear()
    sources_mod._chosen_legs.clear()
    yield
    _reset_registry()
    sources_mod._source_shims.clear()
    sources_mod._chosen_legs.clear()


def _names() -> set[str]:
    return {s.name for s in list_shims()}


def _write_provider(root: Path, *parts: str, name: str | None = None) -> None:
    """Write ``root/<parts>/provider.yaml`` with a valid shim."""
    directory = root.joinpath(*parts)
    directory.mkdir(parents=True, exist_ok=True)
    slug = (name or parts[-1]).split("--")[0]
    (directory / "provider.yaml").write_text(
        PROVIDER_YAML.format(name=name or parts[-1], slug=slug)
    )


def _tar_bytes(files: dict[str, bytes], root: str | None = "repo-abc123") -> bytes:
    """Build an in-memory tar.gz; members are nested under *root* if given."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for path, content in files.items():
            full = f"{root}/{path}" if root else path
            info = tarfile.TarInfo(full)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


_API_SHA = "d" * 40


def _http(data: bytes, sha: str = _API_SHA):
    """Serve *data* for archive URLs and a commit SHA for GitHub's API."""

    def get(url: str, **_kwargs):
        if url.startswith("https://api.github.com/"):
            return json.dumps({"sha": sha}).encode()
        return data

    return get


class TestLocalDirs:
    def test_flat_and_grouped_layouts(self, tmp_path: Path):
        root = tmp_path / "shims"
        _write_provider(root, "solo", name="solo--chat_completions")
        _write_provider(root, "grp", "chat_completions", name="grp--chat_completions")
        _write_provider(root, "grp", "messages", name="grp--messages")

        report = rebuild_shims(extra_dirs=[root])

        assert report.failures() == []
        assert report.sources[0].shim_count == 3
        assert {
            "solo--chat_completions",
            "grp--chat_completions",
            "grp--messages",
        } <= _names()

    def test_model_typed_layout(self, tmp_path: Path):
        """A declared source may group by model type, like the built-ins."""
        root = tmp_path / "shims"
        _write_provider(
            root, "acme", "llm", "chat_completions", name="acme--chat_completions"
        )
        embedding = root / "acme" / "embeddings"
        embedding.mkdir(parents=True)
        (embedding / "provider.yaml").write_text(
            "name: acme--embedding\n"
            "model_type: embeddings\n"
            "format: openai\n"
            "connection:\n"
            "  base_url: https://acme.example/v1\n"
        )

        report = rebuild_shims(extra_dirs=[root])

        assert report.failures() == []
        assert report.sources[0].shim_count == 2
        assert {"acme--chat_completions", "acme--embedding"} <= _names()

    def test_missing_dir_is_reported_not_raised(self, tmp_path: Path):
        report = rebuild_shims(extra_dirs=[tmp_path / "nope"])
        assert report.total_shims > 0  # built-ins still load
        failures = report.failures()
        assert len(failures) == 1
        assert failures[0].ok is False
        assert "not a directory" in (failures[0].error or "")

    def test_bad_yaml_skipped_others_still_load(self, tmp_path: Path):
        root = tmp_path / "shims"
        _write_provider(root, "good", name="good--chat_completions")
        broken = root / "broken"
        broken.mkdir(parents=True)
        (broken / "provider.yaml").write_text("name: [oops\nbase: chat_completions\n")

        report = rebuild_shims(extra_dirs=[root])

        assert report.sources[0].ok is True
        assert report.sources[0].shim_names == ("good--chat_completions",)
        assert "good--chat_completions" in _names()

    def test_strict_raises(self, tmp_path: Path):
        with pytest.raises(ShimSourceError):
            rebuild_shims(extra_dirs=[tmp_path / "nope"], strict=True)


class TestRebuild:
    def test_idempotent(self, tmp_path: Path):
        root = tmp_path / "shims"
        _write_provider(root, "acme", name="acme--chat_completions")
        rebuild_shims(extra_dirs=[root])
        first = _names()
        rebuild_shims(extra_dirs=[root])
        assert _names() == first

    def test_removing_source_drops_shims(self, tmp_path: Path):
        root = tmp_path / "shims"
        _write_provider(root, "acme", name="acme--chat_completions")
        rebuild_shims(extra_dirs=[root])
        assert "acme--chat_completions" in _names()

        report = rebuild_shims()
        assert "acme--chat_completions" not in _names()
        assert report.total_shims > 0  # built-ins intact
        assert "chat_completions" in {s.base for s in list_shims()}

    def test_programmatic_without_config(self, tmp_path: Path):
        """rebuild_shims is usable with no config file at all."""
        root = tmp_path / "shims"
        _write_provider(root, "acme", name="acme--chat_completions")
        report = rebuild_shims(extra_dirs=[root])
        assert report.sources[0].ok is True
        assert "acme--chat_completions" in _names()

    def test_report_is_json_safe(self, tmp_path: Path):
        report = rebuild_shims(extra_dirs=[tmp_path / "nope"])
        json.dumps(report.to_dict())  # must not raise


class TestOutOfBandShims:
    """In-process ``register_shim`` calls must survive a rebuild."""

    @staticmethod
    def _register(name: str) -> None:
        register_shim(ProviderShim(name=name, base="chat_completions"))

    def test_survives_a_no_op_rebuild(self):
        """The CLI runs apply_shim_sources() at startup even with no `shims`."""
        self._register("embedder--x")
        report = rebuild_shims()
        assert report.sources == []
        assert get_shim("embedder--x") is not None

    def test_survives_a_rebuild_with_sources(self, tmp_path: Path):
        root = tmp_path / "shims"
        _write_provider(root, "acme", name="acme--chat_completions")
        self._register("embedder--x")

        rebuild_shims(extra_dirs=[root])

        assert get_shim("embedder--x") is not None
        assert get_shim("acme--chat_completions") is not None

    def test_source_shims_are_still_dropped(self, tmp_path: Path):
        root = tmp_path / "shims"
        _write_provider(root, "acme", name="acme--chat_completions")
        self._register("embedder--x")
        rebuild_shims(extra_dirs=[root])

        rebuild_shims()  # source removed

        assert get_shim("acme--chat_completions") is None
        assert get_shim("embedder--x") is not None


class TestEntryPointTier:
    """Entry-point plugins survive alongside dirs/remotes (precedence tier)."""

    def test_plugin_shim_and_dir_shim_coexist(self, tmp_path: Path, monkeypatch):
        import llm_rosetta.shims.providers as providers_mod

        def register_shims():
            register_shim(ProviderShim(name="plugin--x", base="chat_completions"))

        class _EP:
            name = "fake-plugin"

            @staticmethod
            def load():
                return register_shims

        class _EPs:
            @staticmethod
            def select(group):
                return [_EP()] if group == "llm_rosetta.shim_providers" else []

        monkeypatch.setattr(providers_mod, "entry_points", lambda: _EPs())

        root = tmp_path / "shims"
        _write_provider(root, "acme", name="acme--chat_completions")
        rebuild_shims(extra_dirs=[root])

        assert get_shim("plugin--x") is not None
        assert get_shim("acme--chat_completions") is not None

    def test_source_can_override_a_plugin_shim(self, tmp_path: Path, monkeypatch):
        import llm_rosetta.shims.providers as providers_mod

        def register_shims():
            register_shim(
                ProviderShim(
                    name="dup--x",
                    base="chat_completions",
                    connection=ConnectionConfig(base_url="https://plugin.example/v1"),
                )
            )

        class _EP:
            name = "fake-plugin"

            @staticmethod
            def load():
                return register_shims

        class _EPs:
            @staticmethod
            def select(group):
                return [_EP()] if group == "llm_rosetta.shim_providers" else []

        monkeypatch.setattr(providers_mod, "entry_points", lambda: _EPs())

        root = tmp_path / "shims"
        directory = root / "dup"
        directory.mkdir(parents=True)
        (directory / "provider.yaml").write_text(
            "name: dup--x\n"
            "base: chat_completions\n"
            "connection:\n"
            "  base_url: https://source.example/v1\n"
        )
        rebuild_shims(extra_dirs=[root])

        shim = get_shim("dup--x")
        assert shim is not None
        assert shim.connection.base_url == "https://source.example/v1"


class TestRemote:
    def _spec(self, **over):
        spec = {
            "url": "https://github.com/acme/repo",
            "ref": "v1.0.0",
            "path": "providers",
        }
        spec.update(over)
        return spec

    def test_fetch_extract_and_cache(self, tmp_path: Path):
        data = _tar_bytes(
            {
                "providers/rem/provider.yaml": PROVIDER_YAML.format(
                    name="rem--chat_completions", slug="rem"
                ).encode()
            }
        )
        calls: list[str] = []
        serve = _http(data)

        def http_get(url, *, headers=None, timeout=None, proxy=None):
            calls.append(url)
            return serve(url)

        cache = tmp_path / "cache"
        report = rebuild_shims(
            extra_remotes=[self._spec()], cache_root=cache, http_get=http_get
        )
        src = report.sources[0]
        assert src.ok, src.error
        assert src.shim_names == ("rem--chat_completions",)
        assert src.from_cache is False
        # A declared ref is resolved to its commit, and the archive is fetched
        # at that commit — the stamp records a commit, not the tag name.
        assert calls[0].endswith("/commits/v1.0.0")
        assert src.resolved_ref == _API_SHA
        assert calls[1] == (f"https://codeload.github.com/acme/repo/tar.gz/{_API_SHA}")

        # Second call within the TTL is served from cache — no refetch.
        report2 = rebuild_shims(
            extra_remotes=[self._spec()], cache_root=cache, http_get=http_get
        )
        assert len(calls) == 2
        assert report2.sources[0].from_cache is True

    def test_sha256_mismatch_fails_without_extracting(self, tmp_path: Path):
        cache = tmp_path / "cache"
        report = rebuild_shims(
            extra_remotes=[self._spec(sha256="0" * 64)],
            cache_root=cache,
            http_get=_http(b"not-the-archive"),
        )
        src = report.sources[0]
        assert src.ok is False
        assert "sha256 mismatch" in (src.error or "")
        # No stamp is written and nothing is extracted.
        assert not list(cache.glob("*.stamp.json"))

    def test_sha256_mismatch_detected_on_warm_cache(self, tmp_path: Path):
        """A pin must be re-checked against the *cached* archive, not just once."""
        data = _tar_bytes(
            {
                "providers/rem/provider.yaml": PROVIDER_YAML.format(
                    name="rem--chat_completions", slug="rem"
                ).encode()
            }
        )
        cache = tmp_path / "cache"
        good = hashlib.sha256(data).hexdigest()
        rebuild_shims(
            extra_remotes=[self._spec(sha256=good)],
            cache_root=cache,
            http_get=_http(data),
        )

        # Same (warm) cache, but the pin now names different bytes.
        report = rebuild_shims(
            extra_remotes=[self._spec(sha256="0" * 64)],
            cache_root=cache,
            http_get=_http(data),
        )
        src = report.sources[0]
        assert src.ok is False
        assert "sha256 mismatch" in (src.error or "")
        assert src.stale is False  # a mismatch must not degrade to the cache

    def test_sha256_mismatch_raises_under_strict(self, tmp_path: Path):
        data = _tar_bytes(
            {
                "providers/rem/provider.yaml": PROVIDER_YAML.format(
                    name="rem--chat_completions", slug="rem"
                ).encode()
            }
        )
        cache = tmp_path / "cache"
        rebuild_shims(
            extra_remotes=[self._spec()],
            cache_root=cache,
            http_get=_http(data),
        )
        with pytest.raises(ShimSourceError):
            rebuild_shims(
                extra_remotes=[self._spec(sha256="0" * 64)],
                cache_root=cache,
                http_get=_http(data),
                strict=True,
            )

    def test_declared_path_wins_over_root_strip(self, tmp_path: Path):
        """A declared `path` names the content — don't strip it away as a root."""
        data = _tar_bytes(
            {
                "providers/rem/provider.yaml": PROVIDER_YAML.format(
                    name="rem--chat_completions", slug="rem"
                ).encode()
            },
            root=None,
        )
        report = rebuild_shims(
            extra_remotes=[self._spec(path="providers")],
            cache_root=tmp_path / "cache",
            http_get=_http(data),
        )
        src = report.sources[0]
        assert src.ok, src.error
        assert src.shim_names == ("rem--chat_completions",)

    def test_registry_stays_populated_during_fetch(self, tmp_path: Path):
        """A rebuild must never expose a partially-populated registry.

        Sampled from another thread across the *entire* rebuild — the fetch and
        the registry swap — because the proxy resolves each request's shim by
        name against this registry and degrades silently to "no transforms"
        rather than failing when a lookup misses.
        """
        data = _tar_bytes(
            {
                "providers/rem/provider.yaml": PROVIDER_YAML.format(
                    name="rem--chat_completions", slug="rem"
                ).encode()
            }
        )
        cache = tmp_path / "cache"
        spec = self._spec()
        rebuild_shims(extra_remotes=[spec], cache_root=cache, http_get=_http(data))

        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        samples: list[bool] = []

        def slow_get(url, **kw):
            started.set()
            release.wait(10)
            return data

        def rebuild():
            rebuild_shims(
                extra_remotes=[self._spec(refresh="always")],
                cache_root=cache,
                http_get=slow_get,
            )
            finished.set()

        def sample():
            started.wait(10)
            while not finished.is_set():
                samples.append(get_shim("rem--chat_completions") is not None)
                time.sleep(0.002)

        sampler = threading.Thread(target=sample)
        worker = threading.Thread(target=rebuild)
        sampler.start()
        worker.start()
        time.sleep(0.2)
        release.set()
        worker.join(10)
        sampler.join(10)

        assert samples, "sampler produced no samples"
        assert all(samples), "registry was not fully populated during the rebuild"

    def test_stale_cache_used_when_fetch_fails(self, tmp_path: Path):
        data = _tar_bytes(
            {
                "providers/rem/provider.yaml": PROVIDER_YAML.format(
                    name="rem--chat_completions", slug="rem"
                ).encode()
            }
        )
        cache = tmp_path / "cache"
        rebuild_shims(
            extra_remotes=[self._spec()],
            cache_root=cache,
            http_get=_http(data),
        )

        def boom(url, **kw):
            raise OSError("network down")

        # refresh=always forces a fetch, which fails; the cache is used.
        report = rebuild_shims(
            extra_remotes=[self._spec(refresh="always")],
            cache_root=cache,
            http_get=boom,
        )
        src = report.sources[0]
        assert src.ok is True
        assert src.stale is True
        assert src.shim_names == ("rem--chat_completions",)

    def test_refresh_never_no_fetch_when_cached(self, tmp_path: Path):
        data = _tar_bytes(
            {
                "providers/rem/provider.yaml": PROVIDER_YAML.format(
                    name="rem--chat_completions", slug="rem"
                ).encode()
            }
        )
        cache = tmp_path / "cache"
        rebuild_shims(
            extra_remotes=[self._spec()],
            cache_root=cache,
            http_get=_http(data),
        )
        calls: list[str] = []

        def http_get(url, **kw):
            calls.append(url)
            return data

        rebuild_shims(
            extra_remotes=[self._spec(refresh="never")],
            cache_root=cache,
            http_get=http_get,
        )
        assert calls == []

    def test_head_resolved_and_pinned(self, tmp_path: Path):
        sha = "a" * 40
        data = _tar_bytes(
            {
                "providers/rem/provider.yaml": PROVIDER_YAML.format(
                    name="rem--chat_completions", slug="rem"
                ).encode()
            }
        )
        calls: list[str] = []

        def http_get(url, *, headers=None, timeout=None, proxy=None):
            calls.append(url)
            if url.startswith("https://api.github.com/"):
                return json.dumps({"sha": sha}).encode()
            return data

        cache = tmp_path / "cache"
        report = rebuild_shims(
            extra_remotes=[
                {"url": "https://github.com/acme/repo", "path": "providers"}
            ],
            cache_root=cache,
            http_get=http_get,
        )
        assert report.sources[0].ok, report.sources[0].error
        assert report.sources[0].resolved_ref == sha
        assert "api.github.com" in calls[0]
        assert calls[1] == f"https://codeload.github.com/acme/repo/tar.gz/{sha}"

        # The resolved ref is reused from the cache — no second API call.
        before = len(calls)
        rebuild_shims(
            extra_remotes=[
                {"url": "https://github.com/acme/repo", "path": "providers"}
            ],
            cache_root=cache,
            http_get=http_get,
        )
        assert len(calls) == before

    def test_a_declared_ref_is_pinned_to_its_commit(self, tmp_path: Path):
        """A tag or branch in the declaration is not the pin: its commit is."""
        sha = "b" * 40
        data = _tar_bytes(
            {
                "providers/rem/provider.yaml": PROVIDER_YAML.format(
                    name="rem--chat_completions", slug="rem"
                ).encode()
            }
        )
        calls: list[str] = []

        def http_get(url, *, headers=None, timeout=None, proxy=None):
            calls.append(url)
            if url.startswith("https://api.github.com/"):
                return json.dumps({"sha": sha}).encode()
            return data

        report = rebuild_shims(
            extra_remotes=[
                {
                    "url": "https://github.com/acme/repo",
                    "ref": "v1.0.0",
                    "path": "providers",
                }
            ],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        )
        assert report.sources[0].ok, report.sources[0].error
        assert calls[0].endswith("/commits/v1.0.0")
        assert report.sources[0].resolved_ref == sha
        # The declared ref survives the scan that turns the fetch into a status;
        # the update check asks about it instead of the default branch.
        assert report.sources[0].ref == "v1.0.0"
        assert calls[1] == f"https://codeload.github.com/acme/repo/tar.gz/{sha}"

    def test_an_unreachable_api_still_installs_the_source(self, tmp_path: Path):
        """The commit is for the update check, not a precondition for loading."""
        data = _tar_bytes(
            {
                "providers/rem/provider.yaml": PROVIDER_YAML.format(
                    name="rem--chat_completions", slug="rem"
                ).encode()
            }
        )
        calls: list[str] = []

        def http_get(url, *, headers=None, timeout=None, proxy=None):
            calls.append(url)
            if url.startswith("https://api.github.com/"):
                raise OSError("HTTP 403 rate limit exceeded")
            return data

        report = rebuild_shims(
            extra_remotes=[
                {
                    "url": "https://github.com/acme/repo",
                    "ref": "main",
                    "path": "providers",
                }
            ],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        )
        src = report.sources[0]
        assert src.ok, src.error
        assert src.shim_names == ("rem--chat_completions",)
        # The stamp keeps the ref; the next fetch re-pins it.
        assert src.resolved_ref == "main"
        assert calls[-1] == "https://codeload.github.com/acme/repo/tar.gz/main"

    @pytest.mark.parametrize(
        ("extra", "expect_ref", "archive"),
        [
            ({"ref": "main"}, "main", "main"),
            ({}, None, "HEAD"),
        ],
    )
    def test_the_outage_fallback_covers_both_shapes(
        self, tmp_path: Path, extra: dict, expect_ref: str | None, archive: str
    ):
        data = _tar_bytes(
            {
                "providers/rem/provider.yaml": PROVIDER_YAML.format(
                    name="rem--chat_completions", slug="rem"
                ).encode()
            }
        )

        calls: list[str] = []

        def http_get(url, *, headers=None, timeout=None, proxy=None):
            calls.append(url)
            if url.startswith("https://api.github.com/"):
                raise OSError("HTTP 403 rate limit exceeded")
            return data

        report = rebuild_shims(
            extra_remotes=[
                {"url": "https://github.com/acme/repo", "path": "providers", **extra}
            ],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        )
        src = report.sources[0]
        assert src.ok, src.error
        assert src.ref == extra.get("ref")
        assert src.resolved_ref == expect_ref
        assert calls[-1] == f"https://codeload.github.com/acme/repo/tar.gz/{archive}"

    def test_a_digest_pin_keeps_the_warm_cache_warm(self, tmp_path: Path):
        """A literal-ref stamp is not re-pinned when the bytes are pinned.

        The mirror exists for a network where ``github.com`` cannot be reached.
        There the API never answers, so the stamp can never hold a commit, and
        re-pinning on every apply made each reload pay one API timeout — a warm
        cache that was not warm (#1433).
        """
        data = _tar_bytes(
            {
                "providers/rem/provider.yaml": PROVIDER_YAML.format(
                    name="rem--chat_completions", slug="rem"
                ).encode()
            }
        )
        cache = tmp_path / "cache"
        calls: list[str] = []

        def http_get(url, *, headers=None, timeout=None, proxy=None):
            calls.append(url)
            if url.startswith("https://api.github.com/"):
                raise TimeoutError("blackholed")
            if "codeload" in url:
                raise ConnectionRefusedError("blocked")
            return data

        spec = [
            {
                "url": "https://github.com/acme/repo",
                "ref": "v1",
                "path": "src/pkg/providers",
                "mirror": {
                    "url_template": "https://cdn.example/gh/acme/repo@dist/{ref}/r.tar.gz",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "path": "providers",
                },
            }
        ]
        first = rebuild_shims(
            extra_remotes=spec, cache_root=cache, http_get=http_get
        ).sources[0]
        assert first.ok, first.error
        assert first.leg == "mirror"
        assert first.resolved_ref == "v1"

        calls.clear()
        second = rebuild_shims(
            extra_remotes=spec, cache_root=cache, http_get=http_get
        ).sources[0]
        assert calls == []
        assert second.ok, second.error
        assert second.from_cache is True

    def test_a_stamp_holding_a_branch_name_is_re_pinned(self, tmp_path: Path):
        """Caches written before refs were resolved carry a name, not a commit."""
        sha = "c" * 40
        data = _tar_bytes(
            {
                "providers/rem/provider.yaml": PROVIDER_YAML.format(
                    name="rem--chat_completions", slug="rem"
                ).encode()
            }
        )
        cache = tmp_path / "cache"
        calls: list[str] = []

        def http_get(url, *, headers=None, timeout=None, proxy=None):
            calls.append(url)
            if url.startswith("https://api.github.com/"):
                return json.dumps({"sha": sha}).encode()
            return data

        spec = [
            {"url": "https://github.com/acme/repo", "ref": "main", "path": "providers"}
        ]
        rebuild_shims(extra_remotes=spec, cache_root=cache, http_get=http_get)

        # Rewrite the stamp the way the old loader wrote it: the literal ref.
        stamp_path = next(cache.glob("*.stamp.json"))
        stamp = json.loads(stamp_path.read_text())
        stamp["resolved_ref"] = "main"
        stamp_path.write_text(json.dumps(stamp))

        before = len(calls)
        report = rebuild_shims(extra_remotes=spec, cache_root=cache, http_get=http_get)
        assert len(calls) > before
        assert report.sources[0].resolved_ref == sha

    def test_refresh_never_keeps_a_branch_name_stamp(self, tmp_path: Path):
        """The operator's schedule wins: the re-pin waits for an allowed fetch."""
        sha = "e" * 40
        data = _tar_bytes(
            {
                "providers/rem/provider.yaml": PROVIDER_YAML.format(
                    name="rem--chat_completions", slug="rem"
                ).encode()
            }
        )
        cache = tmp_path / "cache"
        calls: list[str] = []

        def http_get(url, *, headers=None, timeout=None, proxy=None):
            calls.append(url)
            if url.startswith("https://api.github.com/"):
                return json.dumps({"sha": sha}).encode()
            return data

        spec = [
            {"url": "https://github.com/acme/repo", "ref": "main", "path": "providers"}
        ]
        rebuild_shims(extra_remotes=spec, cache_root=cache, http_get=http_get)
        stamp_path = next(cache.glob("*.stamp.json"))
        stamp = json.loads(stamp_path.read_text())
        stamp["resolved_ref"] = "main"
        stamp_path.write_text(json.dumps(stamp))

        before = len(calls)
        report = rebuild_shims(
            extra_remotes=[dict(spec[0], refresh="never")],
            cache_root=cache,
            http_get=http_get,
        )
        assert calls == [] or len(calls) == before
        assert report.sources[0].resolved_ref == "main"

    def test_the_status_carries_the_token_variable(self, tmp_path: Path, monkeypatch):
        """The check needs the variable's name; the fetch proves it is honoured."""
        monkeypatch.setenv("ACME_SHIM_TOKEN", "sekret")
        data = _tar_bytes(
            {
                "providers/rem/provider.yaml": PROVIDER_YAML.format(
                    name="rem--chat_completions", slug="rem"
                ).encode()
            }
        )
        seen: dict[str, str] = {}

        def http_get(url, *, headers=None, timeout=None, proxy=None):
            seen.update(headers or {})
            if url.startswith("https://api.github.com/"):
                return json.dumps({"sha": "f" * 40}).encode()
            return data

        report = rebuild_shims(
            extra_remotes=[
                {
                    "url": "https://github.com/acme/repo",
                    "path": "providers",
                    "token_env": "ACME_SHIM_TOKEN",
                }
            ],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        )
        assert report.sources[0].ok, report.sources[0].error
        assert report.sources[0].token_env == "ACME_SHIM_TOKEN"
        assert seen["Authorization"] == "Bearer sekret"

    def test_a_forced_refetch_re_pins_under_refresh_never(self, tmp_path: Path):
        """An explicit refresh is not the schedule ``never`` governs."""
        sha = "1" * 40
        data = _tar_bytes(
            {
                "providers/rem/provider.yaml": PROVIDER_YAML.format(
                    name="rem--chat_completions", slug="rem"
                ).encode()
            }
        )

        def http_get(url, *, headers=None, timeout=None, proxy=None):
            if url.startswith("https://api.github.com/"):
                return json.dumps({"sha": sha}).encode()
            return data

        cache = tmp_path / "cache"
        spec = {
            "url": "https://github.com/acme/repo",
            "ref": "main",
            "path": "providers",
        }
        rebuild_shims(extra_remotes=[spec], cache_root=cache, http_get=http_get)
        stamp_path = next(cache.glob("*.stamp.json"))
        stamp = json.loads(stamp_path.read_text())
        stamp["resolved_ref"] = "main"
        stamp_path.write_text(json.dumps(stamp))

        report = rebuild_shims(
            extra_remotes=[dict(spec, refresh="always")],
            cache_root=cache,
            http_get=http_get,
        )
        assert report.sources[0].resolved_ref == sha

    def test_non_github_archive_url(self, tmp_path: Path):
        data = _tar_bytes(
            {
                "providers/rem/provider.yaml": PROVIDER_YAML.format(
                    name="rem--chat_completions", slug="rem"
                ).encode()
            }
        )
        calls: list[str] = []

        def http_get(url, **kw):
            calls.append(url)
            return data

        report = rebuild_shims(
            extra_remotes=[
                {
                    "url": "https://gitlab.example/api/v4/x/archive/{ref}.tar.gz",
                    "ref": "main",
                }
            ],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        )
        assert report.sources[0].ok, report.sources[0].error
        assert calls == ["https://gitlab.example/api/v4/x/archive/main.tar.gz"]

    def test_plain_url_without_ref_or_archive_is_rejected(self, tmp_path: Path):
        report = rebuild_shims(
            extra_remotes=[{"url": "https://gitlab.example/x/y"}],
            cache_root=tmp_path / "cache",
            http_get=lambda url, **kw: b"",
        )
        assert report.sources[0].ok is False

    def test_http_scheme_rejected_except_loopback(self, tmp_path: Path):
        st = resolve_remote(
            {"url": "http://evil.example/archive.tar.gz"},
            cache_root=tmp_path,
            http_get=lambda url, **kw: b"",
        )
        assert st.ok is False
        assert "https" in (st.error or "")

    def test_zip_archive(self, tmp_path: Path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                "repo-abc/providers/rem/provider.yaml",
                PROVIDER_YAML.format(name="rem--chat_completions", slug="rem"),
            )
        report = rebuild_shims(
            extra_remotes=[self._spec(archive="zip")],
            cache_root=tmp_path / "cache",
            http_get=_http(buf.getvalue()),
        )
        assert report.sources[0].ok, report.sources[0].error
        assert report.sources[0].shim_names == ("rem--chat_completions",)


class TestSafety:
    def _extract(self, data: bytes, tmp_path: Path, subpath: str = "."):
        return rebuild_shims(
            extra_remotes=[
                {"url": "https://github.com/acme/repo", "ref": "v1", "path": subpath}
            ],
            cache_root=tmp_path / "cache",
            http_get=_http(data),
        )

    def test_traversal_member_rejected(self, tmp_path: Path):
        data = _tar_bytes({"../evil.yaml": b"x"}, root="repo-abc")
        report = self._extract(data, tmp_path)
        assert report.sources[0].ok is False
        assert "traversal" in (report.sources[0].error or "")

    def test_backslash_traversal_rejected(self, tmp_path: Path):
        """``..\\`` must not slip past the traversal check on a POSIX host."""
        data = _tar_bytes({"..\\..\\evil.yaml": b"x"}, root="repo-abc")
        report = self._extract(data, tmp_path)
        assert report.sources[0].ok is False
        assert "traversal" in (report.sources[0].error or "")

    def test_drive_prefixed_member_rejected_before_strip(self, tmp_path: Path):
        """A lone ``C:/x`` member must not be laundered into ``x`` by the strip."""
        data = _tar_bytes({"C:/evil.yaml": b"x"}, root=None)
        report = self._extract(data, tmp_path)
        assert report.sources[0].ok is False
        assert "absolute path" in (report.sources[0].error or "")

    def test_symlink_member_rejected(self, tmp_path: Path):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            info = tarfile.TarInfo("repo-abc/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tf.addfile(info)
        report = self._extract(buf.getvalue(), tmp_path)
        assert report.sources[0].ok is False
        assert "link" in (report.sources[0].error or "")

    def test_missing_subpath_reports_failure(self, tmp_path: Path):
        data = _tar_bytes({"other/readme.md": b"hi"})
        report = self._extract(data, tmp_path, subpath="providers")
        assert report.sources[0].ok is False
        assert "not a directory" in (report.sources[0].error or "")


def _closed_port() -> int:
    """Bind and release a loopback port, so a connect to it is refused."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _LoopbackArchive:
    """A loopback server serving whatever ``body`` holds at request time.

    Used where a test must exercise the **default** getter — its streaming
    read and the cap that aborts it — rather than an injected stand-in.  The
    test rewrites ``body`` between fetches to move from a warm, valid cache to
    an oversized response on the same URL.
    """

    def __init__(self) -> None:
        self.body = b""
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - the base class names it
                payload = outer.body
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/shims.tar.gz"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> _LoopbackArchive:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _provider_archive() -> bytes:
    """A one-shim tar.gz in the shape every remote test expects."""
    return _tar_bytes(
        {
            "providers/rem/provider.yaml": PROVIDER_YAML.format(
                name="rem--chat_completions", slug="rem"
            ).encode()
        }
    )


class TestMirrorLayout:
    """The two legs may lay their archives out differently (#1411 Phase 4).

    The published ``dist`` archives hold ``providers/`` at the root — they are
    built from the providers directory — while the forge tarball keeps the
    repository's ``src/<package>/providers``.  One declared path cannot address
    both, and before ``mirror.path`` existed the mirror leg extracted fine and
    then failed to scan: ``not a directory: <cache>/<src>/src/<pkg>/providers``.
    """

    _SHIM = "rem--chat_completions"

    def _legs(self) -> tuple[bytes, bytes]:
        body = PROVIDER_YAML.format(name=self._SHIM, slug="rem").encode()
        # The forge tarball: one <repo>-<ref> root, then the package tree.
        primary = _tar_bytes({"src/pkg/providers/rem/provider.yaml": body})
        # The published dist archive: providers/ is the root.
        mirror = _tar_bytes({"providers/rem/provider.yaml": body}, root=None)
        return primary, mirror

    def _load(self, tmp_path: Path, *, mirror_extra: dict | None = None):
        primary, mirror = self._legs()

        def http_get(url: str, **_kw):
            if url.startswith("https://api.github.com/"):
                return json.dumps({"sha": _API_SHA}).encode()
            if "codeload" in url:
                raise ConnectionRefusedError("github is blocked here")
            return mirror

        spec = {
            "url": "https://github.com/acme/repo",
            "ref": "v1",
            "path": "src/pkg/providers",
            "mirror": {
                "url_template": "https://cdn.example/gh/acme/repo@dist/{ref}/rem.tar.gz",
                "sha256": hashlib.sha256(mirror).hexdigest(),
                **(mirror_extra or {}),
            },
        }
        return rebuild_shims(
            extra_remotes=[spec], cache_root=tmp_path / "cache", http_get=http_get
        ).sources[0]

    def test_the_mirror_scans_its_own_root(self, tmp_path: Path):
        """``path: "."`` names the extraction root the dist archive has."""
        src = self._load(tmp_path, mirror_extra={"path": "."})
        assert src.ok, src.error
        assert src.leg == "mirror"
        assert src.shim_names == (self._SHIM,)

    def test_without_a_mirror_path_the_source_path_is_inherited(self, tmp_path: Path):
        """The default is the source's path, which the dist archive lacks."""
        src = self._load(tmp_path)
        assert src.leg == "mirror"
        assert src.ok is False
        assert "not a directory" in (src.error or "")

    def test_both_legs_yield_the_same_shim(self, tmp_path: Path):
        """Each leg scans its own tree and registers the same shim."""
        primary, mirror = self._legs()
        by_leg = {}
        for primary_fails in (False, True):

            def http_get(url: str, **_kw):
                if url.startswith("https://api.github.com/"):
                    return json.dumps({"sha": _API_SHA}).encode()
                if "codeload" in url:
                    if primary_fails:
                        raise ConnectionRefusedError("blocked")
                    return primary
                return mirror

            sources_mod._chosen_legs.clear()
            src = rebuild_shims(
                extra_remotes=[
                    {
                        "url": "https://github.com/acme/repo",
                        "ref": "v1",
                        "path": "src/pkg/providers",
                        "mirror": {
                            "url_template": (
                                "https://cdn.example/gh/acme/repo@dist/{ref}/rem.tar.gz"
                            ),
                            "sha256": hashlib.sha256(mirror).hexdigest(),
                            "path": ".",
                        },
                    }
                ],
                cache_root=tmp_path / f"cache-{primary_fails}",
                http_get=http_get,
            ).sources[0]
            assert src.ok, src.error
            by_leg[src.leg] = src.shim_names
        assert by_leg == {"primary": (self._SHIM,), "mirror": (self._SHIM,)}

    def test_the_mirror_is_addressed_by_the_declared_tag(self, tmp_path: Path):
        """The forge's commit resolution must not leak into the mirror URL.

        A mirror publishes one directory per release; substituting the commit
        the tag resolved to asks for a directory that does not exist, and an
        api.github.com rate limit would decide whether the fallback works.
        """
        _primary, mirror = self._legs()

        def http_get(url: str, **_kw):
            if url.startswith("https://api.github.com/"):
                return json.dumps({"sha": _API_SHA}).encode()
            if "codeload" in url:
                raise ConnectionRefusedError("github is blocked here")
            return mirror

        sources_mod._chosen_legs.clear()
        src = rebuild_shims(
            extra_remotes=[
                {
                    "url": "https://github.com/acme/repo",
                    "ref": "v1",
                    "path": "src/pkg/providers",
                    "mirror": {
                        "url_template": (
                            "https://cdn.example/gh/acme/repo@dist/{ref}/rem.tar.gz"
                        ),
                        "sha256": hashlib.sha256(mirror).hexdigest(),
                        "path": ".",
                    },
                }
            ],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        ).sources[0]
        assert src.ok, src.error
        assert src.leg == "mirror"
        assert src.archive_url == "https://cdn.example/gh/acme/repo@dist/v1/rem.tar.gz"
        assert _API_SHA not in src.archive_url


class TestMirror:
    """The mirror leg: selection, integrity and the remembered outcome."""

    def test_unreachable_primary_loads_via_mirror(self, tmp_path: Path):
        """The primary's port is closed, so the mirror must serve the shims."""
        port = _closed_port()
        data = _provider_archive()
        real_get = sources_mod._default_http_get
        calls: list[str] = []

        def http_get(url, *, headers=None, timeout=None, proxy=None):
            calls.append(url)
            if url.startswith(f"http://127.0.0.1:{port}/"):
                # Exercise the real client's connection path, not a mock.
                return real_get(url, headers=headers, timeout=2.0, proxy=proxy)
            return data

        spec = {
            "url": f"http://127.0.0.1:{port}/shims.tar.gz",
            "mirror": "https://mirror.example/shims.tar.gz",
        }
        report = rebuild_shims(
            extra_remotes=[spec], cache_root=tmp_path / "cache", http_get=http_get
        )
        src = report.sources[0]
        assert src.ok, src.error
        assert src.leg == "mirror"
        assert src.archive_url == "https://mirror.example/shims.tar.gz"
        assert src.shim_names == ("rem--chat_completions",)
        assert calls == [
            f"http://127.0.0.1:{port}/shims.tar.gz",
            "https://mirror.example/shims.tar.gz",
        ]
        assert src.to_dict()["leg"] == "mirror"

    def test_a_timeout_also_falls_back(self, tmp_path: Path):
        """A blackholed route times out instead of refusing; it still falls back."""
        calls: list[str] = []
        data = _provider_archive()

        def http_get(url, **_kw):
            calls.append(url)
            if "primary" in url:
                raise TimeoutError("timed out")
            return data

        report = rebuild_shims(
            extra_remotes=[
                {
                    "url": "https://primary.example/shims.tar.gz",
                    "mirror": "https://mirror.example/shims.tar.gz",
                }
            ],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        )
        assert report.sources[0].ok, report.sources[0].error
        assert report.sources[0].leg == "mirror"

    def test_404_from_the_primary_does_not_fall_back(self, tmp_path: Path):
        """A server answer is the source's verdict, not a network symptom."""
        calls: list[str] = []

        def http_get(url, **_kw):
            calls.append(url)
            if "primary" in url:
                raise HTTPError(404, "", url)
            return _provider_archive()

        report = rebuild_shims(
            extra_remotes=[
                {
                    "url": "https://primary.example/shims.tar.gz",
                    "mirror": "https://mirror.example/shims.tar.gz",
                }
            ],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        )
        src = report.sources[0]
        assert src.ok is False
        assert "404" in (src.error or "")
        assert calls == ["https://primary.example/shims.tar.gz"]
        assert src.unreachable is False

    def test_a_corrupt_primary_archive_does_not_fall_back(self, tmp_path: Path):
        """Bytes that arrive but do not extract are a hard error."""
        calls: list[str] = []

        def http_get(url, **_kw):
            calls.append(url)
            if "primary" in url:
                return b"definitely not a tar archive"
            return _provider_archive()

        report = rebuild_shims(
            extra_remotes=[
                {
                    "url": "https://primary.example/shims.tar.gz",
                    "mirror": "https://mirror.example/shims.tar.gz",
                }
            ],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        )
        src = report.sources[0]
        assert src.ok is False
        assert "archive" in (src.error or "")
        assert calls == ["https://primary.example/shims.tar.gz"]

    def test_mirror_hash_mismatch_names_the_source_and_both_digests(
        self, tmp_path: Path
    ):
        data = _provider_archive()
        actual = hashlib.sha256(data).hexdigest()
        expected = "0" * 64
        calls: list[str] = []

        def http_get(url, **_kw):
            calls.append(url)
            if "primary" in url:
                raise ConnectionRefusedError("no route")
            return data

        spec = {
            "url": "https://primary.example/shims.tar.gz",
            "mirror": {
                "url_template": "https://mirror.example/shims.tar.gz",
                "sha256": expected,
            },
        }
        report = rebuild_shims(
            extra_remotes=[spec], cache_root=tmp_path / "cache", http_get=http_get
        )
        src = report.sources[0]
        assert src.ok is False
        error = src.error or ""
        assert "sha256 mismatch" in error
        assert spec["url"] in error
        assert "via mirror" in error
        assert expected in error
        assert actual in error
        # A pin mismatch is final: the primary it replaced is never asked.
        assert calls == [
            "https://primary.example/shims.tar.gz",
            "https://mirror.example/shims.tar.gz",
        ]

        with pytest.raises(ShimSourceError):
            rebuild_shims(
                extra_remotes=[spec],
                cache_root=tmp_path / "cache2",
                http_get=http_get,
                strict=True,
            )

    def test_no_mirror_is_unchanged(self, tmp_path: Path):
        calls: list[str] = []

        def http_get(url, **_kw):
            calls.append(url)
            return _provider_archive()

        report = rebuild_shims(
            extra_remotes=[{"url": "https://primary.example/shims.tar.gz"}],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        )
        src = report.sources[0]
        assert src.ok, src.error
        assert src.leg == "primary"
        assert calls == ["https://primary.example/shims.tar.gz"]

    def test_the_chosen_leg_is_remembered(self, tmp_path: Path):
        """A working mirror is not re-probed behind a failing primary."""
        port = _closed_port()
        data = _provider_archive()
        real_get = sources_mod._default_http_get
        primary_calls = 0
        calls: list[str] = []

        def http_get(url, *, headers=None, timeout=None, proxy=None):
            nonlocal primary_calls
            calls.append(url)
            if url.startswith(f"http://127.0.0.1:{port}/"):
                primary_calls += 1
                return real_get(url, headers=headers, timeout=2.0, proxy=proxy)
            return data

        spec = {
            "url": f"http://127.0.0.1:{port}/shims.tar.gz",
            "mirror": "https://mirror.example/shims.tar.gz",
            "refresh": "always",
        }
        cache = tmp_path / "cache"
        first = rebuild_shims(extra_remotes=[spec], cache_root=cache, http_get=http_get)
        assert first.sources[0].leg == "mirror"
        assert primary_calls == 1

        calls.clear()
        second = rebuild_shims(
            extra_remotes=[spec], cache_root=cache, http_get=http_get
        )
        assert second.sources[0].leg == "mirror"
        # The remembered mirror is tried first; the primary is never contacted.
        assert calls == ["https://mirror.example/shims.tar.gz"]
        assert primary_calls == 1

    def test_the_remembered_leg_can_fall_back(self, tmp_path: Path):
        """The mirror may go down while the primary recovers."""
        data = _provider_archive()
        state = {"primary": "refuse", "mirror": "serve"}

        def http_get(url, **_kw):
            leg = "primary" if "primary" in url else "mirror"
            if state[leg] == "refuse":
                raise ConnectionRefusedError(f"{leg} down")
            return data

        spec = {
            "url": "https://primary.example/shims.tar.gz",
            "mirror": "https://mirror.example/shims.tar.gz",
            "refresh": "always",
        }
        cache = tmp_path / "cache"
        first = rebuild_shims(extra_remotes=[spec], cache_root=cache, http_get=http_get)
        assert first.sources[0].leg == "mirror"

        state.update(primary="serve", mirror="refuse")
        calls: list[str] = []

        def recording_get(url, **kw):
            calls.append(url)
            return http_get(url, **kw)

        second = rebuild_shims(
            extra_remotes=[spec], cache_root=cache, http_get=recording_get
        )
        assert second.sources[0].ok, second.sources[0].error
        assert second.sources[0].leg == "primary"
        assert calls[:2] == [
            "https://mirror.example/shims.tar.gz",
            "https://primary.example/shims.tar.gz",
        ]

    def test_a_mirror_pin_is_checked_once_not_every_fetch(self, tmp_path: Path):
        """The mirror pin satisfies a warm cache, so no refetch is forced."""
        data = _provider_archive()
        digest = hashlib.sha256(data).hexdigest()
        calls: list[str] = []

        def http_get(url, **_kw):
            calls.append(url)
            if "primary" in url:
                raise ConnectionRefusedError("no route")
            return data

        spec = {
            "url": "https://primary.example/shims.tar.gz",
            "mirror": {
                "url_template": "https://mirror.example/shims.tar.gz",
                "sha256": digest,
            },
        }
        cache = tmp_path / "cache"
        first = rebuild_shims(extra_remotes=[spec], cache_root=cache, http_get=http_get)
        assert first.sources[0].ok, first.sources[0].error
        assert len(calls) == 2

        second = rebuild_shims(
            extra_remotes=[spec], cache_root=cache, http_get=http_get
        )
        assert second.sources[0].from_cache is True
        assert second.sources[0].leg == "mirror"
        assert len(calls) == 2  # served from the cache

    def test_a_rotated_mirror_pin_forces_a_refetch(self, tmp_path: Path):
        data = _provider_archive()
        digest = hashlib.sha256(data).hexdigest()

        def http_get(url, **_kw):
            if "primary" in url:
                raise ConnectionRefusedError("no route")
            return data

        cache = tmp_path / "cache"
        rebuild_shims(
            extra_remotes=[
                {
                    "url": "https://primary.example/shims.tar.gz",
                    "mirror": {
                        "url_template": "https://mirror.example/shims.tar.gz",
                        "sha256": digest,
                    },
                }
            ],
            cache_root=cache,
            http_get=http_get,
        )
        rotated = {
            "url": "https://primary.example/shims.tar.gz",
            "mirror": {
                "url_template": "https://mirror.example/shims.tar.gz",
                "sha256": "0" * 64,
            },
        }
        report = rebuild_shims(
            extra_remotes=[rotated], cache_root=cache, http_get=http_get
        )
        src = report.sources[0]
        assert src.ok is False
        assert "sha256 mismatch" in (src.error or "")
        assert src.stale is False  # a mismatch never degrades to the cache

    def test_a_primary_pin_does_not_judge_a_mirror_cache(self, tmp_path: Path):
        """The two legs serve different bytes, so each pin judges its own leg."""
        data = _provider_archive()
        digest = hashlib.sha256(data).hexdigest()
        calls: list[str] = []

        def http_get(url, **_kw):
            calls.append(url)
            if "primary" in url:
                raise ConnectionRefusedError("no route")
            return data

        spec = {
            "url": "https://primary.example/shims.tar.gz",
            "sha256": "1" * 64,  # the *primary's* pin, not the mirror's
            "mirror": {
                "url_template": "https://mirror.example/shims.tar.gz",
                "sha256": digest,
            },
        }
        cache = tmp_path / "cache"
        first = rebuild_shims(extra_remotes=[spec], cache_root=cache, http_get=http_get)
        assert first.sources[0].ok, first.sources[0].error
        before = len(calls)

        second = rebuild_shims(
            extra_remotes=[spec], cache_root=cache, http_get=http_get
        )
        assert second.sources[0].ok, second.sources[0].error
        assert len(calls) == before  # no refetch forced by the other leg's pin

    def test_the_primary_token_is_not_sent_to_the_mirror(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A bearer token belongs to the origin that declared it."""
        monkeypatch.setenv("PRIMARY_TOKEN", "SECRET-CANARY-TOKEN")
        seen: dict[str, dict[str, str]] = {}

        def http_get(url, *, headers=None, **_kw):
            seen[url] = dict(headers or {})
            if "primary" in url:
                raise ConnectionRefusedError("no route")
            return _provider_archive()

        report = rebuild_shims(
            extra_remotes=[
                {
                    "url": "https://primary.example/shims.tar.gz",
                    "token_env": "PRIMARY_TOKEN",
                    "mirror": "https://mirror.example/shims.tar.gz",
                }
            ],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        )
        src = report.sources[0]
        assert src.ok, src.error
        assert src.leg == "mirror"
        # The half that was already right, pinned again: the primary gets it.
        assert (
            seen["https://primary.example/shims.tar.gz"]["Authorization"]
            == "Bearer SECRET-CANARY-TOKEN"
        )
        # The mirror is another origin, so it never sees the credential.
        assert "Authorization" not in seen["https://mirror.example/shims.tar.gz"]

    def test_a_mirror_on_the_primary_origin_keeps_the_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The same host is the same trust boundary, so the token still rides."""
        monkeypatch.setenv("PRIMARY_TOKEN", "sekret")
        seen: dict[str, dict[str, str]] = {}

        def http_get(url, *, headers=None, **_kw):
            seen[url] = dict(headers or {})
            if "/repo/" in url:
                raise ConnectionRefusedError("no route")
            return _provider_archive()

        report = rebuild_shims(
            extra_remotes=[
                {
                    "url": "https://primary.example/repo/shims.tar.gz",
                    "token_env": "PRIMARY_TOKEN",
                    "mirror": "https://primary.example/mirror/shims.tar.gz",
                }
            ],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        )
        assert report.sources[0].ok, report.sources[0].error
        assert (
            seen["https://primary.example/mirror/shims.tar.gz"]["Authorization"]
            == "Bearer sekret"
        )

    @pytest.mark.parametrize(
        "first, second, same",
        [
            ("https://h/p", "https://h:443/m", True),
            ("https://h:443/p", "https://h/m", True),
            ("http://h/p", "http://h:80/m", True),
            ("https://h/p", "https://h:8443/m", False),
            ("https://h/p", "http://h:443/m", False),
            ("https://h/p", "https://other/m", False),
            # A host is compared case-insensitively, and an IPv6 literal keeps
            # its brackets out of the comparison.
            ("https://H/p", "https://h:443/m", True),
            ("https://[::1]/p", "https://[::1]:443/m", True),
            ("https://[::1]:8443/p", "https://[::1]/m", False),
        ],
    )
    def test_an_origin_compares_the_effective_port(self, first, second, same):
        """An explicit default port is the same origin, not another one."""
        assert sources_mod._same_origin(first, second) is same

    def test_a_mirror_on_the_default_port_keeps_the_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Writing :443 explicitly must not cost the credential."""
        monkeypatch.setenv("PRIMARY_TOKEN", "sekret")
        seen: dict[str, dict[str, str]] = {}

        def http_get(url, *, headers=None, **_kw):
            seen[url] = dict(headers or {})
            if "/repo/" in url:
                raise ConnectionRefusedError("no route")
            return _provider_archive()

        report = rebuild_shims(
            extra_remotes=[
                {
                    "url": "https://primary.example/repo/shims.tar.gz",
                    "token_env": "PRIMARY_TOKEN",
                    "mirror": "https://primary.example:443/mirror/shims.tar.gz",
                }
            ],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        )
        assert report.sources[0].ok, report.sources[0].error
        assert (
            seen["https://primary.example:443/mirror/shims.tar.gz"]["Authorization"]
            == "Bearer sekret"
        )

    def test_a_mirror_can_carry_its_own_credential(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A mirror that needs a credential declares its own token_env."""
        monkeypatch.setenv("PRIMARY_TOKEN", "primary-secret")
        monkeypatch.setenv("MIRROR_TOKEN", "mirror-secret")
        seen: dict[str, dict[str, str]] = {}

        def http_get(url, *, headers=None, **_kw):
            seen[url] = dict(headers or {})
            if "primary" in url:
                raise ConnectionRefusedError("no route")
            return _provider_archive()

        report = rebuild_shims(
            extra_remotes=[
                {
                    "url": "https://primary.example/shims.tar.gz",
                    "token_env": "PRIMARY_TOKEN",
                    "mirror": {
                        "url_template": "https://mirror.example/shims.tar.gz",
                        "token_env": "MIRROR_TOKEN",
                    },
                }
            ],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        )
        assert report.sources[0].ok, report.sources[0].error
        assert (
            seen["https://primary.example/shims.tar.gz"]["Authorization"]
            == "Bearer primary-secret"
        )
        assert (
            seen["https://mirror.example/shims.tar.gz"]["Authorization"]
            == "Bearer mirror-secret"
        )

    def test_a_pinned_primary_refuses_an_unpinned_mirror(self, tmp_path: Path):
        """An unpinned mirror would serve bytes the primary's pin never approved."""

        def http_get(url, **_kw):
            if "primary" in url:
                raise ConnectionRefusedError("no route")
            return _provider_archive()

        report = rebuild_shims(
            extra_remotes=[
                {
                    "url": "https://primary.example/shims.tar.gz",
                    "sha256": "1" * 64,
                    "mirror": "https://mirror.example/shims.tar.gz",
                }
            ],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        )
        src = report.sources[0]
        assert src.ok is False
        error = src.error or ""
        assert "mirror" in error
        assert "sha256" in error

    def test_a_mirror_http_error_falls_through_to_the_primary(self, tmp_path: Path):
        """A remembered mirror's 404 must not outrank a primary nobody asked."""
        data = _provider_archive()
        state = {"primary": "refuse", "mirror": "serve"}

        def http_get(url, **_kw):
            leg = "primary" if "primary" in url else "mirror"
            if state[leg] == "refuse":
                raise ConnectionRefusedError(f"{leg} down")
            if state[leg] == "404":
                raise HTTPError(404, "", url)
            return data

        spec = {
            "url": "https://primary.example/shims.tar.gz",
            "mirror": "https://mirror.example/shims.tar.gz",
            "refresh": "always",
        }
        cache = tmp_path / "cache"
        first = rebuild_shims(extra_remotes=[spec], cache_root=cache, http_get=http_get)
        assert first.sources[0].leg == "mirror"

        state.update(primary="serve", mirror="404")
        calls: list[str] = []

        def recording_get(url, **kw):
            calls.append(url)
            return http_get(url, **kw)

        second = rebuild_shims(
            extra_remotes=[spec], cache_root=cache, http_get=recording_get
        )
        src = second.sources[0]
        assert src.ok, src.error
        assert src.leg == "primary"
        assert calls == [
            "https://mirror.example/shims.tar.gz",
            "https://primary.example/shims.tar.gz",
        ]

    def test_no_usable_leg_names_both(self, tmp_path: Path):
        """A mirror error on top of an unreachable primary says both."""

        def http_get(url, **_kw):
            if "mirror" in url:
                raise HTTPError(404, "", url)
            raise ConnectionRefusedError("no route")

        report = rebuild_shims(
            extra_remotes=[
                {
                    "url": "https://primary.example/shims.tar.gz",
                    "mirror": "https://mirror.example/shims.tar.gz",
                }
            ],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        )
        src = report.sources[0]
        assert src.ok is False
        error = src.error or ""
        assert "no usable leg" in error
        assert "https://mirror.example/shims.tar.gz" in error
        assert "https://primary.example/shims.tar.gz" in error
        assert "ConnectionRefusedError" in error

    def test_a_transport_failure_is_still_masked_by_the_cache(self, tmp_path: Path):
        """stale covers "cannot reach": the last good bytes are still served."""
        state = {"up": True}

        def http_get(url, **_kw):
            if not state["up"]:
                raise ConnectionRefusedError("no route")
            return _provider_archive()

        spec = {
            "url": "https://primary.example/shims.tar.gz",
            "mirror": "https://mirror.example/shims.tar.gz",
            "refresh": "always",
        }
        cache = tmp_path / "cache"
        first = rebuild_shims(extra_remotes=[spec], cache_root=cache, http_get=http_get)
        assert first.sources[0].ok, first.sources[0].error

        state["up"] = False
        second = rebuild_shims(
            extra_remotes=[spec], cache_root=cache, http_get=http_get
        )
        src = second.sources[0]
        assert src.ok is True
        assert src.stale is True
        assert src.from_cache is True

    def test_an_http_error_is_still_masked_by_the_cache(self, tmp_path: Path):
        """An HTTP status may be a publish window: stale is still the answer."""
        state = {"status": "ok"}

        def http_get(url, **_kw):
            if state["status"] == "404":
                raise HTTPError(404, "", url)
            return _provider_archive()

        spec = {
            "url": "https://primary.example/shims.tar.gz",
            "refresh": "always",
        }
        cache = tmp_path / "cache"
        first = rebuild_shims(extra_remotes=[spec], cache_root=cache, http_get=http_get)
        assert first.sources[0].ok, first.sources[0].error

        state["status"] = "404"
        second = rebuild_shims(
            extra_remotes=[spec], cache_root=cache, http_get=http_get
        )
        src = second.sources[0]
        assert src.ok is True
        assert src.stale is True

    def test_a_junk_mirror_is_not_masked_by_a_stale_cache(self, tmp_path: Path):
        """Warm cache or not, a body that is not the source is a hard failure."""
        data = _provider_archive()
        state = {"primary": "refuse", "mirror": "serve"}

        def http_get(url, **_kw):
            leg = "primary" if "primary" in url else "mirror"
            if state[leg] == "refuse":
                raise ConnectionRefusedError(f"{leg} down")
            if state[leg] == "junk":
                return b"<html>nope</html>"
            return data

        spec = {
            "url": "https://primary.example/shims.tar.gz",
            "mirror": "https://mirror.example/shims.tar.gz",
            "refresh": "always",
        }
        cache = tmp_path / "cache"
        first = rebuild_shims(extra_remotes=[spec], cache_root=cache, http_get=http_get)
        assert first.sources[0].leg == "mirror"

        state.update(mirror="junk")
        second = rebuild_shims(
            extra_remotes=[spec], cache_root=cache, http_get=http_get
        )
        src = second.sources[0]
        # Not `ok=True stale=True`: the cache must not paper over a bad body.
        assert src.ok is False
        assert "no usable leg" in (src.error or "")
        # And the class survives: a plain source error would be masked again.
        assert (src.error or "").startswith("ShimArchiveUnusable")
        # Each attempt extracts into its own directory and removes it, even
        # when the bytes never became an archive.
        residue = cache / ".tmp"
        assert not residue.exists() or not any(residue.iterdir())

    def test_a_corrupt_body_is_not_masked_by_the_cache(self, tmp_path: Path):
        """The same rule on the primary: corruption is not a network symptom."""
        state = {"body": _provider_archive()}

        def http_get(url, **_kw):
            return state["body"]

        spec = {
            "url": "https://primary.example/shims.tar.gz",
            "mirror": "https://mirror.example/shims.tar.gz",
            "refresh": "always",
        }
        cache = tmp_path / "cache"
        first = rebuild_shims(extra_remotes=[spec], cache_root=cache, http_get=http_get)
        assert first.sources[0].ok, first.sources[0].error

        # The cache is warm and usable: without a refetch it serves from it.
        warm = rebuild_shims(
            extra_remotes=[{**spec, "refresh": "never"}],
            cache_root=cache,
            http_get=http_get,
        )
        assert warm.sources[0].from_cache is True
        assert warm.sources[0].stale is False

        state["body"] = b"<html>nope</html>"
        second = rebuild_shims(
            extra_remotes=[spec], cache_root=cache, http_get=http_get
        )
        src = second.sources[0]
        assert src.ok is False
        assert src.stale is False
        assert "archive" in (src.error or "")

    def test_a_pinned_mirror_fails_even_when_a_leg_follows_it(self, tmp_path: Path):
        """The pin guard holds wherever the mirror sits in the leg order."""
        junk = b"<html>nope</html>"
        pin = hashlib.sha256(junk).hexdigest()
        calls: list[str] = []

        def http_get(url, **_kw):
            calls.append(url)
            return _provider_archive() if "primary" in url else junk

        parsed = sources_mod._parse_remote_spec(
            {
                "url": "https://primary.example/shims.tar.gz",
                "mirror": {
                    "url_template": "https://mirror.example/shims.tar.gz",
                    "sha256": pin,
                },
            },
            tmp_path / "cache",
        )
        # The mirror is first, as a remembered leg would be, and each leg
        # carries its own pin.
        legs = [
            ("mirror", "https://mirror.example/shims.tar.gz", pin),
            ("primary", "https://primary.example/shims.tar.gz", None),
        ]
        with pytest.raises(ShimSourceError) as excinfo:
            sources_mod._fetch_first_usable(
                legs, parsed, headers={}, http_get=http_get, proxy=None
            )
        assert "archive" in str(excinfo.value)
        # Pinned bytes are the operator's declaration: no fallback, ever.
        assert calls == ["https://mirror.example/shims.tar.gz"]

    def test_a_remembered_mirror_with_an_unusable_body_falls_through(
        self, tmp_path: Path
    ):
        """A 200 that is not an archive is a bad replica, not a verdict."""
        data = _provider_archive()
        state = {"primary": "refuse", "mirror": "serve"}

        def http_get(url, **_kw):
            leg = "primary" if "primary" in url else "mirror"
            if state[leg] == "refuse":
                raise ConnectionRefusedError(f"{leg} down")
            if state[leg] == "junk":
                return b"<html>your session has expired</html>"
            return data

        spec = {
            "url": "https://primary.example/shims.tar.gz",
            "mirror": "https://mirror.example/shims.tar.gz",
            "refresh": "always",
        }
        cache = tmp_path / "cache"
        first = rebuild_shims(extra_remotes=[spec], cache_root=cache, http_get=http_get)
        assert first.sources[0].leg == "mirror"

        state.update(primary="serve", mirror="junk")
        calls: list[str] = []

        def recording_get(url, **kw):
            calls.append(url)
            return http_get(url, **kw)

        second = rebuild_shims(
            extra_remotes=[spec], cache_root=cache, http_get=recording_get
        )
        src = second.sources[0]
        assert src.ok, src.error
        assert src.leg == "primary"
        assert src.stale is False
        assert src.from_cache is False
        assert calls == [
            "https://mirror.example/shims.tar.gz",
            "https://primary.example/shims.tar.gz",
        ]

    def test_a_cold_fetch_names_both_legs_when_the_mirror_serves_junk(
        self, tmp_path: Path
    ):
        """No cache to fall back on: the error still names the refused primary."""

        def http_get(url, **_kw):
            if "primary" in url:
                raise ConnectionRefusedError("no route")
            return b"<html>nope</html>"

        report = rebuild_shims(
            extra_remotes=[
                {
                    "url": "https://primary.example/shims.tar.gz",
                    "mirror": "https://mirror.example/shims.tar.gz",
                }
            ],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        )
        src = report.sources[0]
        assert src.ok is False
        error = src.error or ""
        assert "no usable leg" in error
        assert "https://mirror.example/shims.tar.gz" in error
        # The primary's own failure, not just its URL (which is the source URL):
        # "ConnectionRefusedError" only comes from the other leg's attempt.
        assert "ConnectionRefusedError" in error
        assert error.startswith("ShimArchiveUnusable")

    def test_a_pinned_mirror_with_an_unusable_body_fails_hard(self, tmp_path: Path):
        """Pinned bytes are the operator's declaration: no fallback, ever."""
        junk = b"<html>nope</html>"
        calls: list[str] = []

        def http_get(url, **_kw):
            calls.append(url)
            if "primary" in url:
                raise ConnectionRefusedError("no route")
            return junk

        report = rebuild_shims(
            extra_remotes=[
                {
                    "url": "https://primary.example/shims.tar.gz",
                    "mirror": {
                        "url_template": "https://mirror.example/shims.tar.gz",
                        "sha256": hashlib.sha256(junk).hexdigest(),
                    },
                }
            ],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        )
        src = report.sources[0]
        assert src.ok is False
        assert "archive" in (src.error or "")
        # Primary first for transport, then the mirror — and no third attempt:
        # a pinned body that will not extract is final.
        assert calls == [
            "https://primary.example/shims.tar.gz",
            "https://mirror.example/shims.tar.gz",
        ]

    def test_a_local_oserror_is_not_a_transport_failure(self):
        """Only transport failures fall back; a local error is not one."""
        assert sources_mod._is_unreachable(PermissionError("denied")) is False
        assert sources_mod._is_unreachable(FileNotFoundError("gone")) is False

    def test_transport_errors_are_recognized(self):
        assert sources_mod._is_unreachable(ConnectionRefusedError("no route")) is True
        assert sources_mod._is_unreachable(TimeoutError("timed out")) is True


class TestOversizeArchive:
    """The size cap is one rule, whichever layer notices it first.

    A body past :data:`MAX_ARCHIVE_BYTES` is the source answering with
    something that cannot be installed, not a network symptom.  Both layers
    that enforce the cap — the default getter, which aborts the read, and
    :func:`_install_archive`, which measures the finished body — must raise
    :class:`ShimArchiveUnusable`, and a warm cache must not mask either.
    """

    @staticmethod
    def _spec(url: str = "https://primary.example/shims.tar.gz") -> dict:
        return {"url": url, "refresh": "always"}

    @staticmethod
    def _oversized() -> bytes:
        return b"x" * 2048

    @staticmethod
    def _class_of(src) -> str:
        """The exception class the failed status was built from."""
        return (src.error or "").split(":", 1)[0]

    def _warm(self, tmp_path: Path, http_get) -> None:
        """Install a valid archive first, so the next fetch has a cache."""
        first = rebuild_shims(
            extra_remotes=[self._spec()],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        )
        assert first.sources[0].ok, first.sources[0].error

    # -- the default getter aborts the stream ---------------------------

    def test_the_stream_abort_is_unusable(self, tmp_path, monkeypatch):
        """Cold: the reader stops at the cap and the class says why."""
        monkeypatch.setattr(sources_mod, "MAX_ARCHIVE_BYTES", 1024)
        with _LoopbackArchive() as server:
            server.body = self._oversized()
            src = rebuild_shims(
                extra_remotes=[self._spec(server.url)],
                cache_root=tmp_path / "cache",
            ).sources[0]
        assert src.ok is False
        assert src.stale is False
        assert src.from_cache is False
        assert self._class_of(src) == "ShimArchiveUnusable"

    def test_the_stream_abort_is_not_masked_by_a_warm_cache(
        self, tmp_path, monkeypatch
    ):
        """Warm: the cache holds good bytes and still must not answer."""
        monkeypatch.setattr(sources_mod, "MAX_ARCHIVE_BYTES", 1024)
        with _LoopbackArchive() as server:
            server.body = _provider_archive()
            # The warm-up archive is itself over the lowered cap, so warm the
            # cache at the real cap and only then lower it.
            monkeypatch.setattr(sources_mod, "MAX_ARCHIVE_BYTES", 50 * 1024 * 1024)
            first = rebuild_shims(
                extra_remotes=[self._spec(server.url)],
                cache_root=tmp_path / "cache",
            )
            assert first.sources[0].ok, first.sources[0].error

            monkeypatch.setattr(sources_mod, "MAX_ARCHIVE_BYTES", 1024)
            server.body = self._oversized()
            src = rebuild_shims(
                extra_remotes=[self._spec(server.url)],
                cache_root=tmp_path / "cache",
            ).sources[0]
        assert src.ok is False
        assert src.stale is False
        assert src.from_cache is False
        assert self._class_of(src) == "ShimArchiveUnusable"

    # -- an injected getter hands back the oversized bytes --------------

    def test_an_oversized_body_is_unusable(self, tmp_path, monkeypatch):
        """Cold: the install step measures the body and refuses it."""
        monkeypatch.setattr(sources_mod, "MAX_ARCHIVE_BYTES", 1024)
        body = self._oversized()
        src = rebuild_shims(
            extra_remotes=[self._spec()],
            cache_root=tmp_path / "cache",
            http_get=lambda url, **_kw: body,
        ).sources[0]
        assert src.ok is False
        assert src.stale is False
        assert self._class_of(src) == "ShimArchiveUnusable"

    def test_an_oversized_body_is_not_masked_by_a_warm_cache(
        self, tmp_path, monkeypatch
    ):
        """Warm: same class, same decision — the cache does not answer."""
        state = {"body": _provider_archive()}

        def http_get(url, **_kw):
            return state["body"]

        self._warm(tmp_path, http_get)
        monkeypatch.setattr(sources_mod, "MAX_ARCHIVE_BYTES", 1024)
        state["body"] = self._oversized()
        src = rebuild_shims(
            extra_remotes=[self._spec()],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        ).sources[0]
        assert src.ok is False
        assert src.stale is False
        assert self._class_of(src) == "ShimArchiveUnusable"

    def test_both_layers_raise_the_same_class(self, tmp_path, monkeypatch):
        """The issue in one assertion: which layer notices must not matter."""
        monkeypatch.setattr(sources_mod, "MAX_ARCHIVE_BYTES", 1024)
        body = self._oversized()
        through_install = rebuild_shims(
            extra_remotes=[self._spec()],
            cache_root=tmp_path / "injected",
            http_get=lambda url, **_kw: body,
        ).sources[0]
        with _LoopbackArchive() as server:
            server.body = body
            through_stream = rebuild_shims(
                extra_remotes=[self._spec(server.url)],
                cache_root=tmp_path / "streamed",
            ).sources[0]
        assert self._class_of(through_install) == self._class_of(through_stream)
        assert self._class_of(through_install) == "ShimArchiveUnusable"

    # -- what a getter's own exception means ----------------------------

    def test_a_getter_raising_the_unusable_class_is_not_masked(
        self, tmp_path, monkeypatch
    ):
        """A custom getter may reach the verdict itself; the class carries it."""
        state = {"body": _provider_archive()}

        def http_get(url, **_kw):
            if state["body"] is None:
                raise sources_mod.ShimArchiveUnusable("oversized, says the getter")
            return state["body"]

        self._warm(tmp_path, http_get)
        state["body"] = None
        src = rebuild_shims(
            extra_remotes=[self._spec()],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        ).sources[0]
        assert src.ok is False
        assert src.stale is False
        assert self._class_of(src) == "ShimArchiveUnusable"

    def test_a_plain_source_error_from_a_getter_is_still_masked(
        self, tmp_path, monkeypatch
    ):
        """Only a verdict about the body is final; a plain error is not one.

        The class is the rule (see :func:`_exhausted_error`): a getter that
        raises the base class says "this fetch failed", not "these bytes are
        not the source", so a warm cache may still answer — the same way an
        HTTP status does.
        """
        state = {"fail": False}

        def http_get(url, **_kw):
            if state["fail"]:
                raise sources_mod.ShimSourceError("getter gave up")
            return _provider_archive()

        self._warm(tmp_path, http_get)
        state["fail"] = True
        src = rebuild_shims(
            extra_remotes=[self._spec()],
            cache_root=tmp_path / "cache",
            http_get=http_get,
        ).sources[0]
        assert src.ok is True
        assert src.stale is True
        assert src.from_cache is True


class TestLocalInstallFailure:
    """A host that cannot write is not a source that served the wrong bytes.

    ENOSPC, EPERM and EROFS say nothing about the body: the bytes may be
    perfect and this machine still cannot install them.  Such a failure must
    degrade to the stale cache like a transport failure, keep a plain
    :class:`ShimSourceError` class when there is nothing to degrade to — never
    :class:`ShimArchiveUnusable`, which the cache must not soften — and never
    try another leg, because the other leg needs the same disk.
    """

    @staticmethod
    def _warnings(monkeypatch) -> list[str]:
        seen: list[str] = []
        monkeypatch.setattr(
            sources_mod.logger, "warning", lambda msg, *args: seen.append(msg % args)
        )
        return seen

    @staticmethod
    def _http_get(calls: list[str] | None = None):
        def get(url, **_kw):
            if calls is not None:
                calls.append(url)
            return _provider_archive()

        return get

    def test_a_full_disk_degrades_to_the_stale_cache(self, tmp_path, monkeypatch):
        """Warm cache + ENOSPC: the old bytes are still the answer."""
        state = {"fail": False}
        real_replace = sources_mod._atomic_replace

        def replace(src, dest):
            if state["fail"]:
                raise OSError(28, "No space left on device")
            real_replace(src, dest)

        monkeypatch.setattr(sources_mod, "_atomic_replace", replace)
        warnings = self._warnings(monkeypatch)
        spec = {"url": "https://primary.example/shims.tar.gz", "refresh": "always"}
        cache = tmp_path / "cache"
        first = rebuild_shims(
            extra_remotes=[spec], cache_root=cache, http_get=self._http_get()
        )
        assert first.sources[0].ok, first.sources[0].error

        state["fail"] = True
        src = rebuild_shims(
            extra_remotes=[spec], cache_root=cache, http_get=self._http_get()
        ).sources[0]
        assert src.ok is True
        assert src.stale is True
        assert src.from_cache is True
        assert any("using stale cache" in w and "[Errno 28]" in w for w in warnings)

    def test_a_cold_full_disk_is_a_plain_source_error(self, tmp_path, monkeypatch):
        """No cache to soften it: the class must stay distinguishable."""

        def replace(src, dest):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(sources_mod, "_atomic_replace", replace)
        src = rebuild_shims(
            extra_remotes=[{"url": "https://primary.example/shims.tar.gz"}],
            cache_root=tmp_path / "cache",
            http_get=self._http_get(),
        ).sources[0]
        assert src.ok is False
        assert (src.error or "").startswith("ShimSourceError:")
        assert "ShimArchiveUnusable" not in (src.error or "")
        assert "No space left on device" in (src.error or "")
        assert src.unreachable is False

    def test_a_write_denied_during_extraction_also_degrades(
        self, tmp_path, monkeypatch
    ):
        """EPERM while writing a member is the same local wall as ENOSPC."""
        state = {"fail": False}
        real_extract = sources_mod._extract_archive

        def extract(*args, **kwargs):
            if state["fail"]:
                raise PermissionError(13, "Permission denied")
            return real_extract(*args, **kwargs)

        monkeypatch.setattr(sources_mod, "_extract_archive", extract)
        warnings = self._warnings(monkeypatch)
        spec = {"url": "https://primary.example/shims.tar.gz", "refresh": "always"}
        cache = tmp_path / "cache"
        first = rebuild_shims(
            extra_remotes=[spec], cache_root=cache, http_get=self._http_get()
        )
        assert first.sources[0].ok, first.sources[0].error

        state["fail"] = True
        src = rebuild_shims(
            extra_remotes=[spec], cache_root=cache, http_get=self._http_get()
        ).sources[0]
        assert src.ok is True
        assert src.stale is True
        assert any("[Errno 13]" in w for w in warnings)

    def test_a_local_failure_does_not_request_the_other_leg(
        self, tmp_path, monkeypatch
    ):
        """The remembered mirror goes first; a local wall must stop there."""
        state = {"primary_up": False, "fail": False}
        calls: list[str] = []
        real_replace = sources_mod._atomic_replace

        def http_get(url, **_kw):
            calls.append(url)
            if "primary" in url and not state["primary_up"]:
                raise ConnectionRefusedError("no route")
            return _provider_archive()

        def replace(src, dest):
            if state["fail"]:
                raise OSError(28, "No space left on device")
            real_replace(src, dest)

        monkeypatch.setattr(sources_mod, "_atomic_replace", replace)
        spec = {
            "url": "https://primary.example/shims.tar.gz",
            "mirror": "https://mirror.example/shims.tar.gz",
            "refresh": "always",
        }
        cache = tmp_path / "cache"
        first = rebuild_shims(extra_remotes=[spec], cache_root=cache, http_get=http_get)
        assert first.sources[0].ok, first.sources[0].error
        assert first.sources[0].leg == "mirror"

        state["primary_up"] = True
        state["fail"] = True
        calls.clear()
        src = rebuild_shims(
            extra_remotes=[spec], cache_root=cache, http_get=http_get
        ).sources[0]
        assert src.ok is True
        assert src.stale is True
        assert calls == ["https://mirror.example/shims.tar.gz"]


class TestMirrorValidation:
    def test_a_template_string_is_accepted(self):
        assert sources_mod.normalize_mirror("https://m.example/x.tar.gz") == (
            "https://m.example/x.tar.gz",
            None,
            None,
            None,
        )

    def test_an_object_carries_its_own_pin(self):
        pin = "a" * 64
        assert sources_mod.normalize_mirror(
            {"url_template": "https://m.example/x@{ref}.tar.gz", "sha256": pin}
        ) == ("https://m.example/x@{ref}.tar.gz", pin, None, None)

    def test_an_object_can_carry_its_own_credential(self):
        assert sources_mod.normalize_mirror(
            {
                "url_template": "https://m.example/x.tar.gz",
                "token_env": "MIRROR_TOKEN",
            }
        ) == ("https://m.example/x.tar.gz", None, "MIRROR_TOKEN", None)

    @pytest.mark.parametrize(
        "declared, expected", [(".", ""), ("", ""), ("./", ""), ("a/./b", "a/b")]
    )
    def test_a_relative_path_is_normalized(self, declared, expected):
        normalized = sources_mod.normalize_mirror(
            {"url_template": "https://m.example/x.tar.gz", "path": declared}
        )
        assert normalized is not None
        assert normalized[3] == expected

    def test_an_object_can_carry_its_own_scan_path(self):
        normalized = sources_mod.normalize_mirror(
            {"url_template": "https://m.example/x.tar.gz", "path": "providers"}
        )
        assert normalized is not None
        assert normalized[3] == "providers"

    @pytest.mark.parametrize("path", ["/etc", "../x", "a/../../b", "a\\b", 5, ".."])
    def test_a_path_that_can_escape_is_refused(self, path):
        with pytest.raises(ValueError):
            sources_mod.normalize_mirror(
                {"url_template": "https://m.example/x.tar.gz", "path": path}
            )

    def test_absent_is_none(self):
        assert sources_mod.normalize_mirror(None) is None

    @pytest.mark.parametrize(
        "value",
        [
            5,
            "",
            {},
            {"sha256": "a" * 64},
            {"url_template": "https://m.example/not-an-archive"},
            {"url_template": "https://m.example/x.tar.gz", "sha256": "nope"},
            {"url_template": "https://m.example/x.tar.gz", "url": "x"},
            {"url_template": "http://m.example/x.tar.gz"},
            {"url_template": "https://m.example/x.tar.gz", "token_env": ""},
            {"url_template": "https://m.example/x.tar.gz", "token_env": 5},
        ],
    )
    def test_bad_shapes_are_rejected(self, value):
        with pytest.raises(ValueError):
            sources_mod.normalize_mirror(value)

    def test_a_bad_mirror_fails_the_source_not_the_process(self, tmp_path: Path):
        """The loader path reports it; only ``strict`` turns it into an error."""
        report = rebuild_shims(
            extra_remotes=[{"url": "https://primary.example/x.tar.gz", "mirror": 5}],
            cache_root=tmp_path / "cache",
            http_get=lambda url, **_kw: _provider_archive(),
        )
        src = report.sources[0]
        assert src.ok is False
        assert "mirror" in (src.error or "")
