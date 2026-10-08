"""API key management operations."""

from __future__ import annotations

from typing import Any, ClassVar

from llm_rosetta.observability.ops_log import (
    EVENT_KEY_CREATE,
    EVENT_KEY_DELETE,
    EVENT_KEY_ROTATE,
    EVENT_KEY_UPDATE,
    SOURCE_KEYS,
)

from .base import OpsBase, OpsContext


class _OpsKeyBase(OpsBase):
    """Shared base for key ops — record-only, mutation done in route handler."""

    source: ClassVar[str] = SOURCE_KEYS

    async def _run(self) -> None:
        return None


class OpsKeyCreate(_OpsKeyBase):
    event_type: ClassVar[str] = EVENT_KEY_CREATE

    __slots__ = ("_key_id", "_label")

    def __init__(self, ctx: OpsContext, *, key_id: str, label: str) -> None:
        super().__init__(ctx)
        self._key_id = key_id
        self._label = label

    def _message(self, result: Any) -> str:
        return f"API key created: {self._label or '(no label)'}"

    def _details(self, result: Any) -> dict[str, Any]:
        return {"key_id": self._key_id, "label": self._label}


class OpsKeyUpdate(_OpsKeyBase):
    event_type: ClassVar[str] = EVENT_KEY_UPDATE

    __slots__ = ("_key_id", "_changed_fields")

    def __init__(
        self, ctx: OpsContext, *, key_id: str, changed_fields: list[str]
    ) -> None:
        super().__init__(ctx)
        self._key_id = key_id
        self._changed_fields = changed_fields

    def _message(self, result: Any) -> str:
        return f"API key updated: {self._key_id}"

    def _details(self, result: Any) -> dict[str, Any]:
        return {"key_id": self._key_id, "changed_fields": self._changed_fields}


class OpsKeyDelete(_OpsKeyBase):
    event_type: ClassVar[str] = EVENT_KEY_DELETE

    __slots__ = ("_key_id", "_label")

    def __init__(self, ctx: OpsContext, *, key_id: str, label: str | None) -> None:
        super().__init__(ctx)
        self._key_id = key_id
        self._label = label

    def _message(self, result: Any) -> str:
        return f"API key deleted: {self._label or self._key_id}"

    def _details(self, result: Any) -> dict[str, Any]:
        return {"key_id": self._key_id, "label": self._label}


class OpsKeyRotate(_OpsKeyBase):
    event_type: ClassVar[str] = EVENT_KEY_ROTATE

    __slots__ = ("_key_id", "_label")

    def __init__(self, ctx: OpsContext, *, key_id: str, label: str | None) -> None:
        super().__init__(ctx)
        self._key_id = key_id
        self._label = label

    def _message(self, result: Any) -> str:
        return f"API key rotated: {self._label or self._key_id}"

    def _details(self, result: Any) -> dict[str, Any]:
        return {"key_id": self._key_id, "label": self._label}
