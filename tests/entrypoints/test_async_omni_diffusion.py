import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from vllm_omni.entrypoints.async_omni_diffusion import (
    AsyncOmniDiffusion,
    DEFAULT_DIFFUSION_EXECUTOR_WORKERS,
    _resolve_diffusion_executor_workers,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput


pytestmark = [pytest.mark.cpu]


def _make_sampling_params() -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        num_inference_steps=2,
        width=256,
        height=256,
        guidance_scale=0.0,
    )


class _ConcurrentStepEngine:
    def __init__(self) -> None:
        self.calls_started = 0
        self.second_entered_before_first_finished = threading.Event()
        self._first_entered = threading.Event()
        self._allow_first_to_finish = threading.Event()
        self._lock = threading.Lock()

    def step(self, request):
        request_id = request.request_ids[0]
        with self._lock:
            self.calls_started += 1
            call_index = self.calls_started

        if call_index == 1:
            self._first_entered.set()
            if not self._allow_first_to_finish.wait(timeout=5):
                raise TimeoutError("first request was never released")
        else:
            if self._first_entered.is_set():
                self.second_entered_before_first_finished.set()

        return [OmniRequestOutput.from_diffusion(request_id=request_id, images=[])]

    def close(self) -> None:
        pass


def test_generate_allows_second_request_to_enter_engine_before_first_finishes():
    async def _run() -> None:
        engine = _ConcurrentStepEngine()
        async_engine = AsyncOmniDiffusion.__new__(AsyncOmniDiffusion)
        async_engine.engine = engine
        async_engine._executor_workers = 2
        async_engine._executor = ThreadPoolExecutor(max_workers=2)
        async_engine._closed = False
        async_engine._weak_finalizer = None

        first_task = asyncio.create_task(
            async_engine.generate(
                prompt={"prompt": "small"},
                sampling_params=_make_sampling_params(),
                request_id="req-small",
            )
        )

        await asyncio.to_thread(engine._first_entered.wait, 5)

        second_task = asyncio.create_task(
            async_engine.generate(
                prompt={"prompt": "big"},
                sampling_params=_make_sampling_params(),
                request_id="req-big",
            )
        )

        await asyncio.to_thread(engine.second_entered_before_first_finished.wait, 5)
        assert engine.second_entered_before_first_finished.is_set()

        engine._allow_first_to_finish.set()
        first_result, second_result = await asyncio.gather(first_task, second_task)

        assert first_result.request_id == "req-small"
        assert second_result.request_id == "req-big"

        async_engine.close()

    asyncio.run(_run())


def test_resolve_diffusion_executor_workers_defaults_and_validates_env(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_DIFFUSION_EXECUTOR_WORKERS", raising=False)
    assert _resolve_diffusion_executor_workers() == DEFAULT_DIFFUSION_EXECUTOR_WORKERS

    monkeypatch.setenv("VLLM_OMNI_DIFFUSION_EXECUTOR_WORKERS", "16")
    assert _resolve_diffusion_executor_workers() == 16

    monkeypatch.setenv("VLLM_OMNI_DIFFUSION_EXECUTOR_WORKERS", "0")
    assert _resolve_diffusion_executor_workers() == DEFAULT_DIFFUSION_EXECUTOR_WORKERS

    monkeypatch.setenv("VLLM_OMNI_DIFFUSION_EXECUTOR_WORKERS", "bad")
    assert _resolve_diffusion_executor_workers() == DEFAULT_DIFFUSION_EXECUTOR_WORKERS
