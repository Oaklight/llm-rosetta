"""The config lock is re-entrant within the thread that holds it (#1106).

A config mutation takes ``async_config_lock`` and then loads the document.  A
load-time migration with work to do writes the document back, which takes
``config_lock`` — on the same sidecar file, from the same thread.  ``flock`` is
per open file description, so that second acquisition used to block the event
loop forever.
"""

from __future__ import annotations

import threading

from llm_rosetta_gateway.config.io import async_config_lock, config_lock


def test_a_nested_acquisition_in_one_thread_does_not_deadlock(tmp_path):
    path = str(tmp_path / "config.json")
    done = threading.Event()

    def body() -> None:
        with config_lock(path):
            with config_lock(path):
                pass
        done.set()

    worker = threading.Thread(target=body, daemon=True)
    worker.start()

    assert done.wait(5), "nested config_lock deadlocked"


def test_the_async_lock_and_a_nested_sync_lock_coexist(tmp_path):
    import asyncio

    path = str(tmp_path / "config.json")
    reached = []

    async def body() -> None:
        async with async_config_lock(path):
            with config_lock(path):
                reached.append(True)

    asyncio.run(asyncio.wait_for(body(), timeout=5))

    assert reached == [True]
