"""The extension contracts: everything an outside party implements or receives.

A shim author, a host such as the gateway, or a router built on core imports
from this package and from :mod:`llm_rosetta.api`, and from nothing else under
``llm_rosetta``.  Every other module is internal and may move in any release
(:mod:`llm_rosetta._tiers` states the tier of each module).

One module per contract:

- :mod:`llm_rosetta.contracts.shim` — the shim data classes
  (:class:`~llm_rosetta.contracts.shim.ProviderShim` and the connection, tools,
  reasoning, quota and batch declarations);
- :mod:`llm_rosetta.contracts.command` — a subcommand a distribution adds to
  the ``llm-rosetta`` dispatcher (a spec, an entry point group, registration);
- :mod:`llm_rosetta.contracts.extension` — the manifest (``plugin.yaml``) an
  extension folder carries, the compatibility rule, and the transport scope;
- :mod:`llm_rosetta.contracts.transform` — the transform hook types and the
  primitives a shim's ``transforms.py`` builds from;
- :mod:`llm_rosetta.contracts.auth` — the credential exchange a shim ships in
  ``auth.py``;
- :mod:`llm_rosetta.contracts.quota` — the allowance probe a shim ships in
  ``quota.py``;
- :mod:`llm_rosetta.contracts.cli_login` — the vendor CLI sign-in driver an
  ``auth.py`` sidecar uses (standard library only);
- :mod:`llm_rosetta.contracts.routing` — the resolved route and the router
  protocol;
- :mod:`llm_rosetta.contracts.transport` — the upstream transport protocol,
  its response types and the provider view a lane reads;
- :mod:`llm_rosetta.contracts.usage` — usage metering semantics and the report
  (the package, never one of its submodules).

This package itself holds only constants, so importing it loads no converter
and no shim registry.  The sidecar hook names below are the names the shim
loader reads; the loader takes them from here, so they are stated once.

Stability: within one minor release the contracts only gain names, and
``SURFACE_VERSION`` stays put; it rises on a removal or a changed meaning,
never on an addition.  ``IR_VERSION`` rises when the IR data model changes in a
way an IR value written at the old level cannot be read at the new one.
"""

from __future__ import annotations

from llm_rosetta.ir.model import IR_VERSION

#: The extension surface's compatibility level (this package and
#: :mod:`llm_rosetta.api`).  It rises only when a name is removed or changes
#: meaning, never when one is added, so a host asserts a *minimum*: a gateway
#: built for level N runs on any core at level N or above.  A host that needs a
#: name added in a later minor pins that minor instead.
#: ``scripts/dev/public_api_diff.py`` enforces the rise: a breaking finding in
#: this package or in :mod:`llm_rosetta.api` against the last release fails the
#: check unless this level is above the release's.
SURFACE_VERSION = 1

# ---------------------------------------------------------------------------
# Sidecar modules beside a provider.yaml, and the names the loader reads
# ---------------------------------------------------------------------------

#: File name of a shim's transforms sidecar.
TRANSFORMS_MODULE = "transforms.py"
#: File name of a shim's credential-exchange sidecar.
AUTH_MODULE = "auth.py"
#: File name of a shim's allowance-probe sidecar.
QUOTA_MODULE = "quota.py"

#: ``transforms.py``: body transforms applied before the request enters the IR.
PRE_IR_TRANSFORMS = "pre_ir_transforms"
#: ``transforms.py``: body transforms applied after the IR is rendered.
POST_IR_TRANSFORMS = "post_ir_transforms"
#: ``transforms.py``: IR-level transforms, applied to the format-neutral request.
IR_TRANSFORMS = "ir_transforms"
#: ``transforms.py``: body transforms applied to the upstream response body.
RESPONSE_BODY_TRANSFORMS = "response_body_transforms"
#: ``transforms.py``: the callable that rewrites the upstream model listing.
MODEL_LIST_TRANSFORM = "model_list_transform"
#: ``quota.py``: the probes the shim ships (a tuple, or a mapping of id to probe).
QUOTA_PROBES = "PROBES"
#: ``auth.py``: the credential exchange the shim ships.
AUTH_EXCHANGE = "EXCHANGE"

__all__ = [
    "AUTH_EXCHANGE",
    "AUTH_MODULE",
    "IR_TRANSFORMS",
    "IR_VERSION",
    "MODEL_LIST_TRANSFORM",
    "POST_IR_TRANSFORMS",
    "PRE_IR_TRANSFORMS",
    "QUOTA_MODULE",
    "QUOTA_PROBES",
    "RESPONSE_BODY_TRANSFORMS",
    "SURFACE_VERSION",
    "TRANSFORMS_MODULE",
]
