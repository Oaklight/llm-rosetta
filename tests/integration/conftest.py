"""Integration conftest — skip modules whose prerequisites are absent.

Every module here is dual-purpose: a pytest module *and* a script you can
run directly (``python tests/integration/test_anthropic_rest_e2e.py``).  Two
prerequisites are checked as the module imports, and neither being met is a
supported configuration rather than a fault:

*Credentials.*  The scripts check them at import and ``sys.exit(1)`` when one
is missing, which is right for a script and wrong for a collector.
``SystemExit`` does not inherit from ``Exception``, so pytest does not treat
it as a failed import; it escapes collection entirely and aborts the run with
an INTERNALERROR, taking every other module with it.  A bare ``pytest`` in a
checkout without credentials looked catastrophically broken rather than
merely skipped.

*Provider SDKs.*  ``anthropic``, ``openai`` and ``google-genai`` are optional
extras in ``pyproject.toml``, and ``agentabi`` is not installable from here at
all (the ``gateway`` extra is empty), so an install without them is expected.

Catching both here keeps the scripts scripts and reports the modules as
skipped.  The exemption is deliberately narrow: only an exit or an import
failure raised while *importing* is caught, a test that fails once running is
left alone, and only the packages named below are forgiven — an ImportError
from ``llm_rosetta`` itself is real breakage and still fails the run.
"""

import pytest

# Top-level packages whose absence is a supported install, not a defect.
_OPTIONAL_PACKAGES = frozenset({"agentabi", "anthropic", "google", "openai"})


class _ScriptModule(pytest.Module):
    """A module that may decline to import for want of a prerequisite."""

    def collect(self):
        try:
            return super().collect()
        except SystemExit as exc:
            pytest.skip(
                f"{self.path.name} exited with status {exc.code} at import, "
                "which for these scripts means a credential or endpoint is "
                "not configured; set it in .env to run this module",
                allow_module_level=True,
            )
        except pytest.Collector.CollectError as exc:
            # pytest catches an ImportError raised by the module and re-raises
            # it as CollectError, keeping the original on ``__cause__``.  That
            # is the only kind of CollectError we are willing to forgive.
            cause = exc.__cause__
            if not isinstance(cause, ImportError):
                raise
            package = (cause.name or "").split(".")[0]
            if package not in _OPTIONAL_PACKAGES:
                raise
            pytest.skip(
                f"{self.path.name} needs the optional package {package!r}, "
                "which is not installed",
                allow_module_level=True,
            )


def pytest_pycollect_makemodule(module_path, parent):
    return _ScriptModule.from_parent(parent, path=module_path)
