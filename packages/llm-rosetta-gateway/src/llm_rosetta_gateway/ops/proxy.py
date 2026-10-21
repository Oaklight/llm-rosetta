"""Hot-path proxy request operation.

:class:`OpsProxyRequest` replaces ``_record_telemetry()`` in
``gateway/app.py``.  It overrides :meth:`_record` to write to
``request_log`` + ``metrics`` instead of ``ops_log`` — proxy requests
are far too frequent for the audit log.

Every model type reaches the ledger through this one operation, chat and
non-chat alike: a second write path that filled a row differently is what
made the per-key panel and the dashboard disagree before #1014.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace as _dc_replace
from typing import Any, ClassVar

from llm_rosetta.observability.request_log import RequestLogEntry
from llm_rosetta.observability.usage import usage_columns
from llm_rosetta.observability.usage.normalize import DEFAULT_MODEL_TYPE

from ..middleware.auth import api_key_context_var
from ..middleware.request_context import request_context_var
from .base import OpsBase, OpsContext


class OpsProxyRequest(OpsBase):
    """Record a completed proxy request in request_log + metrics.

    This is a record-only operation — the actual proxy handling happens
    outside this class.  :meth:`_run` is a no-op; all logic lives in
    the overridden :meth:`_record`.

    Uses ``__slots__`` for zero-overhead object creation on the hot path.
    """

    event_type: ClassVar[str] = "proxy_request"

    __slots__ = (
        "model",
        "source_provider",
        "target_provider",
        "provider_name",
        "is_stream",
        "status_code",
        "duration_ms",
        "error_detail",
        "profile",
        "model_type",
        "usage",
        "entry_id_override",
        "_entry_id",
    )

    def __init__(
        self,
        ctx: OpsContext,
        *,
        model: str,
        source_provider: str,
        target_provider: str,
        provider_name: str,
        is_stream: bool,
        status_code: int,
        duration_ms: float,
        error_detail: str | None,
        profile: dict[str, Any] | None = None,
        model_type: str = DEFAULT_MODEL_TYPE,
        usage: Mapping[str, Any] | None = None,
        entry_id_override: str | None = None,
    ) -> None:
        super().__init__(ctx)
        self.model = model
        self.source_provider = source_provider
        self.target_provider = target_provider
        self.provider_name = provider_name
        self.is_stream = is_stream
        self.status_code = status_code
        self.duration_ms = duration_ms
        self.error_detail = error_detail
        self.profile = profile
        self.model_type = model_type
        self.usage = usage
        self.entry_id_override = entry_id_override
        self._entry_id: str | None = None

    @property
    def entry_id(self) -> str | None:
        """The request log entry ID, available after :meth:`execute`."""
        return self._entry_id

    async def _run(self) -> None:
        return None

    async def _record(self, result: Any, *, error: BaseException | None = None) -> None:
        metrics = self._ctx.metrics
        if self.is_stream and metrics:
            metrics.active_streams -= 1

        # The chat leg fills ``profile["usage"]``; a non-chat handler holds its
        # IR usage already and passes it in, which is the more direct answer of
        # the two and therefore the one that wins.
        if self.usage is not None:
            _usage: Mapping[str, Any] | None = self.usage
        else:
            _usage = (self.profile or {}).get("usage") if not self.is_stream else None
        # The ledger columns, total included — a row that reports any token
        # count is never NULL in that column: every view sums it, and a NULL row
        # counts in none of them.
        _columns = usage_columns(_usage, self.model_type)
        _input_tokens = _columns.get("input_tokens")
        _output_tokens = _columns.get("output_tokens")

        if metrics:
            metrics.record_request(
                model=self.model,
                source=self.source_provider,
                target=self.target_provider,
                status_code=self.status_code,
                duration_ms=self.duration_ms,
                is_stream=self.is_stream,
                provider_name=self.provider_name,
                error_detail=self.error_detail,
                **_columns,
            )

        request_log = self._ctx.request_log
        if request_log is not None:
            _kctx = api_key_context_var.get()
            entry = RequestLogEntry.create(
                model=self.model,
                source_provider=self.source_provider,
                target_provider=self.target_provider,
                target_provider_name=self.provider_name,
                is_stream=self.is_stream,
                status_code=self.status_code,
                duration_ms=self.duration_ms,
                error_detail=self.error_detail,
                api_key_label=_kctx.label if _kctx else None,
                api_key_bucket=_kctx.usage_bucket if _kctx else None,
                client_ip=(
                    _rctx.client_ip if (_rctx := request_context_var.get()) else None
                ),
                profile=self.profile,
                model_type=self.model_type,
                **_columns,
            )
            if self.entry_id_override:
                entry = _dc_replace(entry, id=self.entry_id_override)
            await request_log.add(entry)
            self._entry_id = entry.id

    def _message(self, result: Any) -> str:
        return ""  # not used — _record is overridden

    def _details(self, result: Any) -> dict[str, Any]:
        return {}  # not used — _record is overridden


async def record_proxy_request(
    app: Any,
    *,
    model: str,
    source_provider: str,
    target_provider: str,
    provider_name: str,
    is_stream: bool,
    status_code: int,
    duration_ms: float,
    error_detail: str | None,
    model_type: str = DEFAULT_MODEL_TYPE,
    profile: dict[str, Any] | None = None,
    usage: Mapping[str, Any] | None = None,
    entry_id_override: str | None = None,
) -> None:
    """Record one proxy request against the app's ops context.

    Both the chat handler and the non-chat pipelines record through here, so a
    row is built in exactly one place.  An app without an ops context — the
    in-memory test setups, a gateway started without persistence — records
    nothing rather than failing the request it was serving.

    Args:
        app: The application holding ``ops_ctx``.
        usage: The response's IR usage, when the caller has it; it takes
            precedence over the profile's, which only the chat leg fills.
    """
    ctx = getattr(app, "ops_ctx", None)
    if ctx is None:
        return
    op = OpsProxyRequest(
        ctx,
        model=model,
        source_provider=source_provider,
        target_provider=target_provider,
        provider_name=provider_name,
        is_stream=is_stream,
        status_code=status_code,
        duration_ms=duration_ms,
        error_detail=error_detail,
        profile=profile,
        model_type=model_type,
        usage=usage,
        entry_id_override=entry_id_override,
    )
    await op.execute()
