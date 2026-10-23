"""One sink for what a conversion loses, and one place it is logged.

A conversion can lose something on every leg: reading a request, writing it,
reading a response, writing it back, and both directions of a stream.  A part
the target has no form for, a field its spec does not define, a tool it cannot
declare, a block whose translation raised.

Those losses used to reach the caller through three unrelated channels — a
list of free strings on the request context, a ``logger.warning`` inside
whichever converter noticed, and stdlib :mod:`warnings`.  A caller that
inspected only the list silently missed whatever went to the other two, and
the stream leg had no route to the list at all: each ``StreamContext`` carried
a list of its own, so a drop seen while converting a chunk was gone with the
chunk.

This module gives them one route:

* :class:`ConversionWarning` is one loss, with its provenance.  It is a
  ``str``, so every existing consumer — ``"; ".join(...)``, a log format, a
  JSON body, a test comparing against the message — keeps working.
* :class:`WarningSink` is the list they accumulate in: one per conversion,
  shared by its legs, and the one place the log line is written.
* :func:`report_drop` is how a converter reports one.  It uses an explicit
  sink when it has one, then the conversion in progress, and only when
  neither exists — a library caller reaching a helper on its own — writes
  the log line itself.

Recording is one act, so a converter that reports does not also log.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextvars import Token

logger = logging.getLogger("llm_rosetta")

#: The target has no form for the item; it is gone.
DROPPED = "dropped"
#: The item was carried, in a weaker form than the source expressed.
DOWNGRADED = "downgraded"
#: The item's translation raised, and the item was skipped.
FAILED = "failed"


class ConversionWarning(str):
    """One conversion loss: its message, plus where it came from.

    A ``str`` subclass on purpose.  Every consumer of a warning predates this
    type and treats it as the message — joining the list, formatting it into a
    log line, putting it in a JSON body — and a test that compares a warning
    against its text has to keep passing.  The provenance rides alongside.

    Attributes:
        code: What kind of loss it was — :data:`DROPPED`, :data:`DOWNGRADED`,
            :data:`FAILED`, or a converter's own string.
        leg: The leg that noticed: ``"request"``, ``"response"`` or
            ``"stream"``.
        converter: The conversion it belongs to, as
            ``"<source>-><target>"``.
        subject: The item the loss is about (a tool name, a field, a content
            type), when the reporter knows it.
    """

    # Set in ``__new__``; declared here so a str subclass still types.
    code: str
    leg: str
    converter: str
    subject: str

    def __new__(
        cls,
        message: str,
        *,
        code: str = DROPPED,
        leg: str = "",
        converter: str = "",
        subject: str = "",
    ) -> ConversionWarning:
        """Build the warning from its message and provenance."""
        self = super().__new__(cls, message)
        self.code = code
        self.leg = leg
        self.converter = converter
        self.subject = subject
        return self

    def __getnewargs_ex__(self) -> tuple[tuple[str], dict[str, str]]:
        """Carry the provenance through copy and pickle."""
        return (str(self),), {
            "code": self.code,
            "leg": self.leg,
            "converter": self.converter,
            "subject": self.subject,
        }

    def as_dict(self) -> dict[str, str]:
        """Return the warning as a plain dict for a structured log or body."""
        return {
            "code": self.code,
            "leg": self.leg,
            "converter": self.converter,
            "subject": self.subject,
            "message": str(self),
        }


def log_warning(warning: ConversionWarning) -> None:
    """Write the one log line a recorded warning gets."""
    logger.warning(
        "Conversion warning (%s, %s): %s",
        warning.leg or "unknown",
        warning.code,
        warning,
    )


class WarningSink(list):
    """The losses of one conversion, and the place they are logged.

    A ``list`` subclass, because ``ConversionContext.warnings`` has always
    been a list: callers slice it, join it, test it for emptiness and compare
    it against plain strings.  Subclassing keeps all of that, and lets an
    existing ``warnings.append("...")`` keep working — the string is lifted
    into a :class:`ConversionWarning` carrying the sink's provenance.

    Attributes:
        leg: The leg the warnings recorded now belong to.  The pipeline moves
            it forward as the conversion advances.
        converter: The conversion they belong to, as ``"<source>-><target>"``.
        log: Whether recording writes a log line.  A fidelity shadow
            round-trip turns it off: what it loses is a diagnostic, not
            something the request lost.
    """

    leg = "request"
    converter = ""
    log = True

    def __init__(
        self,
        *,
        leg: str = "request",
        converter: str = "",
        log: bool = True,
    ) -> None:
        """Create an empty sink for one conversion.

        Args:
            leg: The leg the conversion starts on; the pipeline moves it as
                it advances.
            converter: The conversion's label, as ``"<source>-><target>"``.
            log: Whether recording writes a log line.
        """
        super().__init__()
        self.leg = leg
        self.converter = converter
        self.log = log

    def warn(
        self,
        message: str,
        *,
        code: str = DROPPED,
        subject: str = "",
    ) -> ConversionWarning:
        """Record one loss, lifting *message* into a warning.

        Args:
            message: What was lost, in words the caller can act on.
            code: The kind of loss; see :data:`DROPPED`.
            subject: The item the loss is about, when it is known.

        Returns:
            The warning as recorded.
        """
        warning = ConversionWarning(
            message,
            code=code,
            leg=self.leg,
            converter=self.converter,
            subject=subject,
        )
        self.append(warning)
        return warning

    def append(self, message: str) -> None:
        """Record *message* — lifting it into a warning — and log it."""
        warning = self._lift(message)
        super().append(warning)
        if self.log:
            log_warning(warning)

    def extend(self, messages: Any) -> None:
        """Record each of *messages*, as :meth:`append` would."""
        for message in messages:
            self.append(message)

    def __iadd__(self, messages: Any) -> WarningSink:
        """Record each of *messages* — ``+=`` goes through :meth:`extend`.

        ``list.__iadd__`` is implemented in C and would append raw strings
        without lifting them or logging, making it the one mutator that
        silently escapes the sink.  Routing it through :meth:`extend` keeps
        ``sink += [...]`` the same act as ``sink.extend([...])``.
        """
        self.extend(messages)
        return self

    def _lift(self, message: Any) -> ConversionWarning:
        """Return *message* as a warning carrying this sink's provenance."""
        if isinstance(message, ConversionWarning):
            if message.leg:
                return message
            return ConversionWarning(
                str(message),
                code=message.code,
                leg=self.leg,
                converter=message.converter or self.converter,
                subject=message.subject,
            )
        return ConversionWarning(str(message), leg=self.leg, converter=self.converter)


def lift_warning(
    message: Any, *, code: str = DROPPED, sink: WarningSink | None = None
) -> ConversionWarning:
    """Return *message* as a warning, taking provenance from *sink* if given."""
    if sink is not None:
        return sink._lift(message)
    if isinstance(message, ConversionWarning):
        return message
    return ConversionWarning(str(message), code=code)


#: The sink of the conversion in progress, if any.  A converter helper with no
#: context in scope reports here; the pipeline (and the stream processor) set
#: it for the duration of a leg, and reset it afterwards.
_CURRENT: ContextVar[Any] = ContextVar("llm_rosetta_conversion_warnings", default=None)


def current_sink() -> WarningSink | None:
    """Return the sink of the conversion in progress, when there is one."""
    return _CURRENT.get()


def use_sink(sink: WarningSink | None) -> Token:
    """Make *sink* the conversion in progress.

    Args:
        sink: The sink to report to, or ``None`` to record nothing ambiently.

    Returns:
        The token to hand back to :func:`reset_sink`.
    """
    return _CURRENT.set(sink)


def reset_sink(token: Token) -> None:
    """Undo the :func:`use_sink` that produced *token*."""
    _CURRENT.reset(token)


@contextmanager
def recording_to(sink: WarningSink | None) -> Iterator[WarningSink | None]:
    """Run a converter call with *sink* as the conversion in progress.

    Args:
        sink: The sink the conversion's losses belong to, or ``None`` to
            record nothing ambiently.

    Yields:
        *sink*, for the caller's convenience.
    """
    token = use_sink(sink)
    try:
        yield sink
    finally:
        reset_sink(token)


def report_drop(
    message: str,
    *,
    code: str = DROPPED,
    subject: str = "",
    warnings: Any = None,
) -> ConversionWarning:
    """Record one conversion loss, wherever it was noticed.

    Prefers the sink the caller names, then the conversion in progress, and
    only when neither exists writes the log line itself — so a library caller
    that reached a helper with no pipeline behind it still learns what was
    lost, and a caller inside a pipeline gets it in the list instead of in
    the log twice.

    Args:
        message: What was lost, in words the caller can act on.
        code: The kind of loss; see :data:`DROPPED`.
        subject: The item the loss is about, when it is known.
        warnings: An explicit sink or plain list to append to.  A plain list
            is appended to without logging — it is the caller's, not this
            conversion's.

    Returns:
        The warning as recorded (or as logged, when there was no sink).
    """
    if isinstance(warnings, WarningSink):
        return warnings.warn(message, code=code, subject=subject)
    if isinstance(warnings, list):
        warning = ConversionWarning(str(message), code=code, subject=subject)
        warnings.append(warning)
        return warning
    sink = _CURRENT.get()
    if sink is not None:
        return sink.warn(message, code=code, subject=subject)
    warning = ConversionWarning(str(message), code=code, subject=subject)
    log_warning(warning)
    return warning
