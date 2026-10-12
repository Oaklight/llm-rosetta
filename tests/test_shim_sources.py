"""Tests for declarative external shim sources (directories + remote archives)."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import threading
import time
import zipfile
from pathlib import Path

import pytest

from llm_rosetta.shims import (
    ShimSourceError,
    get_shim,
    list_shims,
    rebuild_shims,
    register_shim,
    resolve_remote,
)
from llm_rosetta.shims import sources as sources_mod
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
    yield
    _reset_registry()
    sources_mod._source_shims.clear()


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

        def http_get(url, *, headers=None, timeout=None, proxy=None):
            calls.append(url)
            return data

        cache = tmp_path / "cache"
        report = rebuild_shims(
            extra_remotes=[self._spec()], cache_root=cache, http_get=http_get
        )
        src = report.sources[0]
        assert src.ok, src.error
        assert src.shim_names == ("rem--chat_completions",)
        assert src.from_cache is False
        assert calls == ["https://codeload.github.com/acme/repo/tar.gz/v1.0.0"]

        # Second call within the TTL is served from cache — no refetch.
        report2 = rebuild_shims(
            extra_remotes=[self._spec()], cache_root=cache, http_get=http_get
        )
        assert len(calls) == 1
        assert report2.sources[0].from_cache is True

    def test_sha256_mismatch_fails_without_extracting(self, tmp_path: Path):
        cache = tmp_path / "cache"
        report = rebuild_shims(
            extra_remotes=[self._spec(sha256="0" * 64)],
            cache_root=cache,
            http_get=lambda url, **kw: b"not-the-archive",
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
            http_get=lambda url, **kw: data,
        )

        # Same (warm) cache, but the pin now names different bytes.
        report = rebuild_shims(
            extra_remotes=[self._spec(sha256="0" * 64)],
            cache_root=cache,
            http_get=lambda url, **kw: data,
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
            http_get=lambda url, **kw: data,
        )
        with pytest.raises(ShimSourceError):
            rebuild_shims(
                extra_remotes=[self._spec(sha256="0" * 64)],
                cache_root=cache,
                http_get=lambda url, **kw: data,
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
            http_get=lambda url, **kw: data,
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
        rebuild_shims(
            extra_remotes=[spec], cache_root=cache, http_get=lambda url, **kw: data
        )

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
            http_get=lambda url, **kw: data,
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
            http_get=lambda url, **kw: data,
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
            http_get=lambda url, **kw: buf.getvalue(),
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
            http_get=lambda url, **kw: data,
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
