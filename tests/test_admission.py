import asyncio
import threading

import pytest

from app.errors import ServiceError
from app.main import Admission


def test_bounded_queue_and_timeout(settings):
    settings.queue_capacity = 1
    settings.queue_timeout_seconds = 0.05
    started, release = threading.Event(), threading.Event()

    def block(_):
        started.set()
        release.wait(5)

    async def scenario():
        admission = Admission(settings)
        first = asyncio.create_task(admission.run(block, None))
        await asyncio.to_thread(started.wait, 2)
        second = asyncio.create_task(admission.run(block, None))
        await asyncio.sleep(0)
        try:
            with pytest.raises(ServiceError) as error:
                await admission.run(block, None)
            assert error.value.code == "QUEUE_FULL"
            with pytest.raises(ServiceError) as error:
                await second
            assert error.value.code == "QUEUE_TIMEOUT"
        finally:
            release.set()
            await first
        assert admission.count == 0
    asyncio.run(scenario())


def test_cancelled_http_caller_does_not_release_inference_slot(settings):
    settings.queue_capacity = 0
    started, release = threading.Event(), threading.Event()

    def block(_):
        started.set()
        release.wait(5)
        return "finished"

    async def scenario():
        admission = Admission(settings)
        first = asyncio.create_task(admission.run(block, None))
        await asyncio.to_thread(started.wait, 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        try:
            with pytest.raises(ServiceError) as error:
                await admission.run(block, None)
            assert error.value.code == "QUEUE_FULL"
        finally:
            release.set()
            await admission.drain()
        assert admission.count == 0
    asyncio.run(scenario())

