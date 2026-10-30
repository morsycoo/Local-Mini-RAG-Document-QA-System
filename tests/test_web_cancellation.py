import asyncio
import threading

import pytest

from app.web.main import run_sync_operation_to_completion


def test_cancellation_waits_for_sync_worker_to_finish():
    async def scenario():
        worker_started = threading.Event()
        allow_worker_to_finish = threading.Event()
        worker_finished = threading.Event()

        def blocking_operation():
            worker_started.set()
            try:
                if not allow_worker_to_finish.wait(timeout=5):
                    raise TimeoutError("Test did not release the worker in time.")
                return "completed"
            finally:
                worker_finished.set()

        task = asyncio.create_task(
            run_sync_operation_to_completion(blocking_operation)
        )

        try:
            started = await asyncio.to_thread(worker_started.wait, 2)
            assert started, "The synchronous worker did not start."

            task.cancel()
            await asyncio.sleep(0)

            assert not task.done(), (
                "The cancelled task finished before its worker completed."
            )
            assert not worker_finished.is_set(), (
                "The worker unexpectedly finished before it was released."
            )

            allow_worker_to_finish.set()

            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=3)

            assert worker_finished.is_set(), (
                "The worker must finish before cancellation propagates."
            )
        finally:
            allow_worker_to_finish.set()
            if not task.done():
                try:
                    await asyncio.wait_for(task, timeout=3)
                except (asyncio.CancelledError, TimeoutError):
                    pass

    asyncio.run(scenario())
