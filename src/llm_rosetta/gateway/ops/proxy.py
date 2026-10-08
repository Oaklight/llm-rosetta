"""Hot-path proxy request operation.

:class:`OpsProxyRequest` replaces ``_record_telemetry()`` in
``gateway/app.py``.  It overrides :meth:`_record` to write to
``request_log`` + ``metrics`` instead of ``ops_log`` — proxy requests
are far too frequent for the audit log.
"""

from __future__ import annotations

from dataclasses import replace as _dc_replace
from typing import Any, ClassVar

from llm_rosetta.observability.request_log import RequestLogEntry

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

        _usage = (self.profile or {}).get("usage") if not self.is_stream else None
        _input_tokens = _usage.get("prompt_tokens") if _usage else None
        _output_tokens = _usage.get("completion_tokens") if _usage else None
        _total_tokens = _usage.get("total_tokens") if _usage else None
        _cache_read_tokens = _usage.get("cache_read_tokens") if _usage else None
        _cache_creation_tokens = _usage.get("cache_creation_tokens") if _usage else None
        _reasoning_tokens = _usage.get("reasoning_tokens") if _usage else None

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
                input_tokens=_input_tokens,
                output_tokens=_output_tokens,
                cache_read_tokens=_cache_read_tokens,
                cache_creation_tokens=_cache_creation_tokens,
                reasoning_tokens=_reasoning_tokens,
            )

        request_log = self._ctx.request_log
        if request_log is not None:
            entry = RequestLogEntry.create(
                model=self.model,
                source_provider=self.source_provider,
                target_provider=self.target_provider,
                target_provider_name=self.provider_name,
                is_stream=self.is_stream,
                status_code=self.status_code,
                duration_ms=self.duration_ms,
                error_detail=self.error_detail,
                api_key_label=(
                    _kctx.label if (_kctx := api_key_context_var.get()) else None
                ),
                client_ip=(
                    _rctx.client_ip if (_rctx := request_context_var.get()) else None
                ),
                profile=self.profile,
                input_tokens=_input_tokens,
                output_tokens=_output_tokens,
                total_tokens=_total_tokens,
                cache_read_tokens=_cache_read_tokens,
                cache_creation_tokens=_cache_creation_tokens,
                reasoning_tokens=_reasoning_tokens,
            )
            if self.entry_id_override:
                entry = _dc_replace(entry, id=self.entry_id_override)
            await request_log.add(entry)
            self._entry_id = entry.id

    def _message(self, result: Any) -> str:
        return ""  # not used — _record is overridden

    def _details(self, result: Any) -> dict[str, Any]:
        return {}  # not used — _record is overridden
