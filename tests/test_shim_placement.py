"""The placement rule for core shims, as a test (#1660 S4).

A vendor ships in core only when it is the model maker's own API and it
authenticates with a static API key.  Two of those facts can be read off the
tree, so they are checked here:

1. **No credential exchange.**  No file named ``auth.py`` anywhere under a core
   vendor directory, and no built-in shim with an exchange or a
   ``token_command`` attached: a lane that mints or refreshes its credential
   (OAuth, a subscription plan) lives in contrib.

2. **A quota probe reads only that key.**  Every probe a core ``quota.py``
   exports declares (or defaults to) the ``api_key`` credential role, *and*
   is named by ``quota.probe`` on at least one shim of its own vendor.  The
   role alone is not enough: the ChatGPT plan probe (``openai_codex``)
   declared the default role while it read an OAuth access token from the key
   slot of another vendor's lane.  What gave it away is that no ``openai``
   shim selected it — it lived in ``openai/quota.py`` only for
   ``openai_chatgpt``.  A probe that its own static-key shims select reads
   their key, which is what the rule asks.

There is no allowlist: since the contrib move nothing in core breaks the rule,
and a vendor that needs an exception belongs in contrib.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_rosetta.contracts.quota import PROBE_ROLE_API_KEY, probe_credential_role
from llm_rosetta.shims import providers as providers_pkg
from llm_rosetta.shims.providers import load_providers

PROVIDERS = Path(providers_pkg.__file__).parent

#: The core vendor directories: every non-underscore directory under
#: ``shims/providers`` (``__pycache__`` is not a vendor).
VENDORS = sorted(
    p for p in PROVIDERS.iterdir() if p.is_dir() and not p.name.startswith("_")
)


def _core_shims():
    """The built-in shims, keyed by vendor directory name."""
    by_vendor: dict[str, list] = {}
    for shim in load_providers():
        assert shim.manifest is not None, shim.name
        vendor_dir = Path(shim.manifest.path).parent
        assert PROVIDERS in vendor_dir.parents or vendor_dir.parent == PROVIDERS
        by_vendor.setdefault(vendor_dir.name, []).append(shim)
    return by_vendor


def test_there_are_vendors_to_check():
    assert {"openai", "anthropic", "google", "deepseek"} <= {v.name for v in VENDORS}


@pytest.mark.parametrize("vendor", VENDORS, ids=lambda p: p.name)
def test_a_core_vendor_ships_no_auth_module(vendor: Path):
    found = sorted(str(p.relative_to(PROVIDERS)) for p in vendor.rglob("auth.py"))
    assert found == [], (
        "a core vendor authenticates with a static API key only; a credential "
        "exchange belongs in the contrib shim repository"
    )


def test_no_built_in_shim_mints_its_credential():
    minting = [
        shim.name
        for shims in _core_shims().values()
        for shim in shims
        if shim.auth is not None
        or shim.connection.token_command
        or shim.connection.token_command_env
    ]
    assert minting == []


def _probe_problems(vendor: str, shims) -> list[str]:
    """What breaks clause 2 among *shims* (one vendor's shims); empty when nothing."""
    probes = {probe.id: probe for shim in shims for probe in shim.quota_probes}
    selected = {shim.quota.probe for shim in shims if shim.quota is not None}
    problems = []
    for probe_id, probe in sorted(probes.items()):
        if probe_credential_role(probe) != PROBE_ROLE_API_KEY:
            problems.append(f"probe {probe_id!r} reads a second credential")
        if probe_id not in selected:
            problems.append(
                f"probe {probe_id!r} is not selected by any {vendor!r} shim, so "
                "it reads some other lane's credential; it belongs with that lane"
            )
    return problems


def test_there_is_a_probe_to_check():
    assert any(shim.quota_probes for shims in _core_shims().values() for shim in shims)


@pytest.mark.parametrize(
    "vendor", sorted(p.name for p in VENDORS if (p / "quota.py").is_file())
)
def test_a_core_probe_reads_only_its_vendors_api_key(vendor: str):
    assert _probe_problems(vendor, _core_shims()[vendor]) == []


def test_the_rule_catches_the_probe_that_moved(tmp_path: Path):
    """Clause 2, applied to the layout core had before #1660.

    ``openai/quota.py`` exported ``openai_codex`` with the default role while no
    ``openai`` shim selected it; the selection clause is what reports it.
    """
    from _moved_vendors import MOVED_VENDORS_DIR
    from llm_rosetta.api import load_providers_from_dir

    probe_source = (MOVED_VENDORS_DIR / "openai_chatgpt" / "quota.py").read_text()
    vendor = tmp_path / "openai"
    lane = vendor / "llm" / "chat_completions"
    lane.mkdir(parents=True)
    (vendor / "provider.yaml").write_text(
        "connection:\n  base_url: https://api.openai.com/v1\n"
        "  api_key_env: OPENAI_API_KEY\n"
    )
    (vendor / "quota.py").write_text(probe_source)
    (lane / "provider.yaml").write_text(
        "name: openai--chat_completions\nformat: chat_completions\nmodel_type: llm\n"
    )

    shims = load_providers_from_dir(tmp_path)

    assert _probe_problems("openai", shims) == [
        "probe 'openai_codex' is not selected by any 'openai' shim, so it reads "
        "some other lane's credential; it belongs with that lane"
    ]
