"""The plan lane's allowance probe — the same vendor reading as ``openai``.

The plan route meters the same account windows the Codex backend reports, so
the probe is the group-level implementation rather than a second copy of it:
one vendor fact, one place.  This shim declares it ``in_band: true``, which
means the probe is loaded for its refusal classifier and never polled — there
is no usage endpoint the application may read.
"""

from llm_rosetta.shims.quota_probe import QuotaProbe
from llm_rosetta.shims.providers.openai.quota import PROBES as _OPENAI_PROBES

PROBES: tuple[QuotaProbe, ...] = _OPENAI_PROBES
