"""Names the old flat modules exposed survive the package split.

``routing`` and ``fidelity`` had no ``__all__``, and ``profiling`` declared only
``DeepProfiler``, so callers could also import names those modules merely
re-exported.  The packages keep every one of those names reachable while
declaring the owned surface in ``__all__``.

``pipeline`` is deliberately absent: it declared a nine-name ``__all__``, which
the package keeps as a strict superset, so the helper names it merely imported
were never its declared surface.
"""

from __future__ import annotations

from llm_rosetta import fidelity, profiling, routing
from llm_rosetta.fidelity import DEFAULT_SEVERITY_BY_KIND, canonical_provider_name
from llm_rosetta.profiling import Profiler, TracingProfiler
from llm_rosetta.routing import ProviderType, SoftErrorPattern


def test_routing_preserves_flat_module_names() -> None:
    assert routing.ProviderType is ProviderType
    assert routing.SoftErrorPattern is SoftErrorPattern
    assert "ProviderType" in routing.__all__
    assert "SoftErrorPattern" in routing.__all__


def test_fidelity_preserves_flat_module_names() -> None:
    assert fidelity.canonical_provider_name is canonical_provider_name
    assert fidelity.DEFAULT_SEVERITY_BY_KIND is DEFAULT_SEVERITY_BY_KIND
    # The constant is owned by the module, so it is declared API.
    assert "DEFAULT_SEVERITY_BY_KIND" in fidelity.__all__


def test_profiling_preserves_flat_module_names() -> None:
    assert profiling.Profiler is Profiler
    assert profiling.TracingProfiler is TracingProfiler
    # The old module's declared surface was DeepProfiler only.
    assert profiling.__all__ == ["DeepProfiler"]
