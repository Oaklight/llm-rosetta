"""``llm_rosetta.shims.providers`` keeps the names 0.13.0 exported (#1660).

0.13.0 held the shim loader in that module, and shim packages import
``get_model_list_transform`` and ``load_providers`` from it.  The loader moved
to the internal ``llm_rosetta.shims._loader``; the old path forwards each name,
as the same object, with a ``DeprecationWarning`` that points at the caller,
until 0.15.0.
"""

from __future__ import annotations

import importlib
import warnings

import pytest

import llm_rosetta.api as api
import llm_rosetta.contracts.shim as contract_shim
import llm_rosetta.shims as shims
from llm_rosetta.shims import _loader, provider_shim

#: 0.13.0's public names in ``llm_rosetta.shims.providers`` and where each
#: lives now.  The module had no ``__all__``; these are its public defs and
#: its re-exports from ``llm_rosetta``.
V0_13_0_NAMES = {
    "ProviderShim": contract_shim,
    "ReasoningCapability": contract_shim,
    "register_shim": provider_shim,
    "load_providers": _loader,
    "load_providers_from_dir": _loader,
    "get_model_list_transform": _loader,
    "register_model_list_transform": _loader,
    "ModelListTransform": _loader,
}


@pytest.fixture(autouse=True)
def _unbind_forwarded_names():
    """Start each test with no forwarded name bound, so the first access warns."""
    providers = importlib.import_module("llm_rosetta.shims.providers")
    for name in V0_13_0_NAMES:
        providers.__dict__.pop(name, None)
    yield
    for name in V0_13_0_NAMES:
        providers.__dict__.pop(name, None)


def test_a_from_import_warns_exactly_once_and_a_second_access_not_at_all():
    providers = importlib.import_module("llm_rosetta.shims.providers")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from llm_rosetta.shims.providers import get_model_list_transform  # noqa: F401
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, [str(w.message) for w in deprecations]
    assert deprecations[0].filename == __file__
    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter("always")
        from llm_rosetta.shims.providers import get_model_list_transform  # noqa: F401,F811

        _ = providers.get_model_list_transform
    assert [w for w in again if issubclass(w.category, DeprecationWarning)] == []


def test_the_producer_import_works_and_warns():
    providers = importlib.import_module("llm_rosetta.shims.providers")
    with pytest.warns(DeprecationWarning, match="removed in 0.15.0") as caught:
        from llm_rosetta.shims.providers import get_model_list_transform
    assert get_model_list_transform is api.get_model_list_transform
    # stacklevel points at the importing line, not at the forwarder.
    assert caught[0].filename == __file__
    assert "llm_rosetta.api" in str(caught[0].message)
    assert providers.__all__ == sorted(V0_13_0_NAMES)


@pytest.mark.parametrize("name", sorted(V0_13_0_NAMES))
def test_every_0_13_0_name_is_the_same_object_and_warns(name):
    providers = importlib.import_module("llm_rosetta.shims.providers")
    with pytest.warns(DeprecationWarning, match=f"shims.providers.{name} is"):
        value = getattr(providers, name)
    assert value is getattr(V0_13_0_NAMES[name], name)


def test_the_plugin_entry_point_resolves_to_the_shims_export():
    providers = importlib.import_module("llm_rosetta.shims.providers")
    with pytest.warns(DeprecationWarning, match="use llm_rosetta.shims$"):
        value = providers.load_providers_from_dir
    assert value is shims.load_providers_from_dir


def test_importing_the_package_does_not_warn():
    # Core imports this package on every start-up (the sidecars load under it),
    # so only an access to a forwarded name may warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        importlib.reload(importlib.import_module("llm_rosetta.shims.providers"))


def test_an_unknown_name_is_an_attribute_error():
    providers = importlib.import_module("llm_rosetta.shims.providers")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(AttributeError):
            _ = providers.no_such_name
        assert not hasattr(providers, "_leaf_connection")
