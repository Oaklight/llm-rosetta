"""What the lanes of one gateway have reported about their allowance.

An agent that serves a lane reports the account's rate-limit windows during a
turn (:mod:`llm_rosetta_gateway.transport.acp.usage` decodes them), and a shim
may declare a local command that prints them (:mod:`.lane_command`).  Both land
here, in a :class:`LaneAllowanceStore` that the app owns: one per app, bound to
every provider the app serves (:func:`bind_lane_allowance`), so two apps in one
process never read each other's readings and a config reload, which builds new
providers, keeps what was reported.

A good reading is also written beside the lane's profile
(``.llm-rosetta-quota.json``) and read back whenever memory has none, so a
gateway restart or an image upgrade keeps the card populated until the agent
reports again.  The file holds shares and reset times only, never a token.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import tempfile
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from ..transport.acp.usage import ACP_QUOTA_PROBE, windows_reported
from .models import STATUS_ERROR, STATUS_OK, QuotaReading, QuotaWindow

#: The last good reading is kept beside the lane's profile.
_READING_FILE = ".llm-rosetta-quota.json"
_READING_SCHEMA = 1


def lane_identity(provider_info: Any) -> str:
    """A stable, non-secret identity for one lane's login profile.

    The profile directory is what the sign-in writes to, so a renamed provider
    entry that keeps ``acp.home`` keeps the same identity; two entries with
    different profiles are different accounts even when they name one CLI.
    """
    return str(
        getattr(provider_info, "acp_state_dir", None)
        or getattr(provider_info, "config_name", "")
        or ""
    )


def lane_credential(provider_info: Any) -> str:
    """The fingerprint the quota payload names for this lane.

    The digest rule is the one :func:`quota.models.credential_fingerprint`
    uses; the two must agree, and a test pins that.  The raw profile path is
    never part of a payload.
    """
    identity = lane_identity(provider_info)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _reading_path(provider_info: Any) -> str | None:
    """Where this lane's reading is kept, when its profile directory exists."""
    home = str(getattr(provider_info, "acp_state_dir", "") or "")
    if not home or not os.path.isdir(home):
        return None
    return os.path.join(home, _READING_FILE)


def _timestamp(value: Any) -> datetime | None:
    """Parse an ISO timestamp from the profile file."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _write_reading(provider_info: Any, reading: QuotaReading) -> None:
    """Keep a good reading beside the profile; an error is not persisted."""
    if reading.status != STATUS_OK or not reading.windows:
        return
    path = _reading_path(provider_info)
    if path is None:
        return
    temp = ""
    stored = reading.to_dict()
    # The credential is derived from this very path, so a stored copy could
    # only echo the filename back.  This file is the profile's, not an
    # account's: a re-login inside one profile is replaced by the next turn.
    stored.pop("credential", None)
    try:
        fd, temp = tempfile.mkstemp(prefix=".quota-", dir=os.path.dirname(path))
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                {"schema": _READING_SCHEMA, "reading": stored},
                handle,
                separators=(",", ":"),
            )
        os.replace(temp, path)
        temp = ""
    except OSError:
        pass
    finally:
        if temp:
            try:
                os.unlink(temp)
            except OSError:
                pass


def _load_reading(provider_info: Any) -> QuotaReading | None:
    """Read the profile's kept reading, or ``None`` when there is none."""
    path = _reading_path(provider_info)
    if path is None or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != _READING_SCHEMA:
        return None
    raw = payload.get("reading")
    if not isinstance(raw, dict) or raw.get("status", STATUS_OK) != STATUS_OK:
        return None
    windows: list[QuotaWindow] = []
    for item in raw.get("windows") or []:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        used = item.get("used_percent")
        used_percent = None
        if isinstance(used, (int, float)) and not isinstance(used, bool):
            used = float(used)
            used_percent = used if math.isfinite(used) else None
        span = item.get("span_seconds")
        windows.append(
            QuotaWindow(
                name=item["name"],
                used_percent=used_percent,
                span_seconds=(
                    int(span)
                    if isinstance(span, int) and not isinstance(span, bool)
                    else None
                ),
                resets_at=_timestamp(item.get("resets_at")),
                status=str(item.get("status") or ""),
            )
        )
    if not windows:
        return None
    read_at = _timestamp(raw.get("read_at"))
    if read_at is None:
        # "now" would read as freshly taken.  The file's mtime is the honest
        # fallback; without it, refuse the reading rather than claim an age.
        try:
            read_at = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
        except OSError:
            return None
    demote_at = raw.get("demote_at")
    return QuotaReading(
        probe=str(raw.get("probe") or ACP_QUOTA_PROBE),
        credential=lane_credential(provider_info),
        windows=windows,
        plan=raw.get("plan") if isinstance(raw.get("plan"), str) else None,
        status=STATUS_OK,
        read_at=read_at,
        demote_at=(
            float(demote_at)
            if isinstance(demote_at, (int, float)) and not isinstance(demote_at, bool)
            else 100.0
        ),
        advisory=bool(raw.get("advisory", False)),
    )


class LaneAllowanceStore:
    """The readings of every lane one app serves, in memory and beside profiles.

    It is the :class:`~llm_rosetta_gateway.transport.lane_allowance.LaneAllowanceSink`
    the providers are bound to.
    """

    def __init__(self) -> None:
        self._readings: dict[str, QuotaReading] = {}
        self._readings_lock = threading.Lock()
        # A lock is bound to the loop that created it, so a restarted loop
        # gets a new one.
        self._locks: dict[str, tuple[Any, asyncio.Lock]] = {}
        self._locks_lock = threading.Lock()

    def clear(self) -> None:
        """Drop every reading held in memory.

        A reading kept beside a profile stays on disk and is read again on the
        next call.
        """
        with self._readings_lock:
            self._readings.clear()

    def reading(self, lane: Any) -> QuotaReading | None:
        """The last allowance this lane reported, from memory or its profile."""
        key = lane_identity(lane)
        with self._readings_lock:
            reading = self._readings.get(key)
            if reading is not None:
                return reading
        # A miss is retried: a login creates the profile directory, and a
        # volume may be mounted a moment after the gateway starts.  One stat
        # per card open is cheap, and caching a miss would hide the file that
        # appears.
        reading = _load_reading(lane)
        if reading is None:
            return None
        with self._readings_lock:
            # A turn may have recorded a newer reading while the file was read.
            return self._readings.setdefault(key, reading)

    def reading_of(
        self, update: Mapping[str, Any], provider_info: Any
    ) -> QuotaReading | None:
        """Normalize one ``usage_update`` into a reading, or ``None``."""
        windows = windows_reported(dict(update))
        if not windows:
            return None
        return QuotaReading(
            probe=ACP_QUOTA_PROBE,
            credential=lane_credential(provider_info),
            windows=windows,
            status=STATUS_OK,
            read_at=datetime.now(timezone.utc),
        )

    def record(self, update: Mapping[str, Any], lane: Any) -> bool:
        """Record the reading *update* carries, if it carries one.

        Returns:
            Whether the update reported an allowance.  A token-count-only
            update leaves the last reading untouched and returns ``False``.
        """
        reading = self.reading_of(update, lane)
        if reading is None:
            return False
        self.store(lane, reading)
        return True

    def store(self, provider_info: Any, reading: QuotaReading) -> None:
        """Replace this lane's last reading with *reading*, memory and profile."""
        key = lane_identity(provider_info)
        with self._readings_lock:
            self._readings[key] = reading
        _write_reading(provider_info, reading)

    def store_local(self, provider_info: Any, reading: QuotaReading) -> None:
        """Store a local read, never replacing a good reading with a failure.

        A lane may have two sources: its agent's ``usage_update`` and a
        declared command.  A failed command is worth reporting, but it is not
        evidence that the windows the agent just sent are wrong, so the good
        reading stays.
        """
        if reading.status == STATUS_ERROR:
            current = self.reading(provider_info)
            if current is not None and current.status != STATUS_ERROR:
                return
        self.store(provider_info, reading)

    def lock(self, provider_info: Any) -> asyncio.Lock:
        """The lock that serialises reads of this lane's local command."""
        loop = asyncio.get_running_loop()
        key = lane_identity(provider_info)
        with self._locks_lock:
            held = self._locks.get(key)
            if held is None or held[0] is not loop:
                held = (loop, asyncio.Lock())
                self._locks[key] = held
            return held[1]


def bind_lane_allowance(config: Any, store: LaneAllowanceStore) -> None:
    """Bind *store* to every provider of *config*.

    Called when an app serves a config, and again for each config a reload
    builds, so the lanes of the new providers report to the same store.
    """
    for provider_info in getattr(config, "providers", {}).values():
        provider_info.lane_allowance = store
