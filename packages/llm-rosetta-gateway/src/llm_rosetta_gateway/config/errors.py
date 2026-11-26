"""The error a config raises for a retired name, with where it was found."""

from __future__ import annotations


class RetiredConfigName(ValueError):
    """A config document spells a name that was retired.

    A :class:`ValueError` carrying the retired-names message (it ends with the
    ``llm-rosetta gateway migrate`` command), plus where the name sits: the config file
    and the JSON path inside it (``models.m.capabilities[0]``).  The message
    reads ``<file>: <json path>: <retired-names message>``; either location part
    is left out until it is known.

    Attributes:
        detail: The retired-names message alone.
        where: The JSON path of the offending value, or ``None``.
        path: The config file, or ``None`` before the loader has added it.
    """

    def __init__(
        self, detail: str, *, where: str | None = None, path: str | None = None
    ) -> None:
        self.detail = detail
        self.where = where
        self.path = path
        super().__init__(": ".join(part for part in (path, where, detail) if part))

    def located(
        self, *, where: str | None = None, path: str | None = None
    ) -> RetiredConfigName:
        """A copy with *where* and *path* filled in where they were not known."""
        return RetiredConfigName(
            self.detail,
            where=self.where or where,
            path=self.path or path,
        )
