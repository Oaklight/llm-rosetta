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
from pathlib import Path

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


def _provider_archive() -> bytes:
    """A one-shim tar.gz in the shape every remote test expects."""
    return _tar_bytes(
        {
            "providers/rem/provider.yaml": PROVIDER_YAML.format(
                name="rem--chat_completions", slug="rem"
            ).encode()
        }
    )


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

        def http_get(url, **_kw):
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


class TestMirrorValidation:
    def test_a_template_string_is_accepted(self):
        assert sources_mod.normalize_mirror("https://m.example/x.tar.gz") == (
            "https://m.example/x.tar.gz",
            None,
        )

    def test_an_object_carries_its_own_pin(self):
        pin = "a" * 64
        assert sources_mod.normalize_mirror(
            {"url_template": "https://m.example/x@{ref}.tar.gz", "sha256": pin}
        ) == ("https://m.example/x@{ref}.tar.gz", pin)

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
