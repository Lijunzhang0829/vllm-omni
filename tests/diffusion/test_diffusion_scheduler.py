# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from unittest.mock import Mock, patch

import pytest

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.diffusion_engine import _make_default_scheduler
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import (
    DiffusionRequestStatus,
    RequestScheduler,
    Scheduler,
    SchedulerInterface,
    SuperP95StepScheduler,
)
from vllm_omni.diffusion.sched.interface import CachedRequestData, NewRequestData
from vllm_omni.diffusion.super_p95 import (
    EXTRA_ARG_SUPER_P95_ESTIMATED_SERVICE_S,
    EXTRA_ARG_SUPER_P95_SACRIFICIAL,
    estimate_service_time_s,
    normalize_wan2_2_num_frames,
)
from vllm_omni.diffusion.worker.utils import RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_request(req_id: str) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[f"prompt_{req_id}"],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_ids=[req_id],
    )


def _make_super_request(
    req_id: str,
    *,
    estimated_service_s: float,
    sacrificial: bool = False,
    num_inference_steps: int = 10,
) -> OmniDiffusionRequest:
    req = OmniDiffusionRequest(
        prompts=[f"prompt_{req_id}"],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=num_inference_steps),
        request_ids=[req_id],
    )
    req.sampling_params.extra_args[EXTRA_ARG_SUPER_P95_SACRIFICIAL] = sacrificial
    req.sampling_params.extra_args[EXTRA_ARG_SUPER_P95_ESTIMATED_SERVICE_S] = estimated_service_s
    return req


def _make_request_output(req_id: str, *, error: str | None = None) -> DiffusionOutput:
    del req_id
    return DiffusionOutput(output=None, error=error)


def _new_ids(sched_output) -> list[str]:
    return [req.sched_req_id for req in sched_output.scheduled_new_reqs]


def _cached_ids(sched_output) -> list[str]:
    return list(sched_output.scheduled_cached_reqs.sched_req_ids)


class _StubScheduler(SchedulerInterface):
    def __init__(self, request: OmniDiffusionRequest, output: DiffusionOutput) -> None:
        self._request = request
        self._output = output
        self.initialized_with = None
        self._sched_req_id = request.request_ids[0]
        self._state = None
        self._scheduled = False

    def initialize(self, od_config) -> None:
        self.initialized_with = od_config

    def add_request(self, request: OmniDiffusionRequest) -> str:
        assert request is self._request
        self._state = Mock(sched_req_id=self._sched_req_id, req=request)
        return self._sched_req_id

    def schedule(self):
        if self._scheduled or self._state is None:
            return Mock(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=CachedRequestData.make_empty(),
                scheduled_req_ids=[],
                is_empty=True,
            )
        self._scheduled = True
        return Mock(
            scheduled_new_reqs=[NewRequestData.from_state(self._state)],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            scheduled_req_ids=[self._state.sched_req_id],
            is_empty=False,
        )

    def update_from_output(self, sched_output, output: DiffusionOutput) -> set[str]:
        del sched_output
        assert output is self._output
        return {self._sched_req_id}

    def update_from_runner_output(self, sched_output, runner_output: RunnerOutput) -> set[str]:
        del sched_output
        assert runner_output.req_id == self._sched_req_id
        assert runner_output.result is self._output
        return {self._sched_req_id}

    def has_requests(self) -> bool:
        return not self._scheduled

    def get_request_state(self, sched_req_id: str):
        del sched_req_id
        return self._state

    def get_sched_req_id(self, request_id: str) -> str | None:
        if request_id in self._request.request_ids:
            return self._sched_req_id
        return None

    def pop_request_state(self, sched_req_id: str):
        del sched_req_id
        return self._state

    def preempt_request(self, sched_req_id: str) -> bool:
        del sched_req_id
        return False

    def finish_requests(self, sched_req_ids, status) -> None:
        del sched_req_ids, status
        return None

    def close(self) -> None:
        return None


class TestRequestScheduler:
    def setup_method(self) -> None:
        self.scheduler: RequestScheduler = RequestScheduler()
        self.scheduler.initialize(Mock())

    def test_single_request_success_lifecycle(self) -> None:
        req_id = self.scheduler.add_request(_make_request("a"))
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.WAITING

        sched_output = self.scheduler.schedule()
        assert _new_ids(sched_output) == [req_id]
        assert _cached_ids(sched_output) == []
        assert sched_output.num_running_reqs == 1
        assert sched_output.num_waiting_reqs == 0

        finished = self.scheduler.update_from_output(sched_output, _make_request_output(req_id))
        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        assert self.scheduler.has_requests() is False

    def test_error_output_marks_finished_error(self) -> None:
        req_id = self.scheduler.add_request(_make_request("err"))

        sched_output = self.scheduler.schedule()
        finished = self.scheduler.update_from_output(
            sched_output,
            _make_request_output(req_id, error="worker failed"),
        )

        assert finished == {req_id}
        state = self.scheduler.get_request_state(req_id)
        assert state.status == DiffusionRequestStatus.FINISHED_ERROR
        assert state.error == "worker failed"

    def test_stepwise_runner_output_keeps_request_running_until_finished(self) -> None:
        req_id = self.scheduler.add_request(_make_request("step"))

        first = self.scheduler.schedule()
        unfinished = RunnerOutput(req_id=req_id, step_index=1, finished=False, result=None)
        finished = self.scheduler.update_from_runner_output(first, unfinished)

        assert finished == set()
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.RUNNING
        assert self.scheduler.has_requests() is True

        second = self.scheduler.schedule()
        assert _new_ids(second) == []
        assert _cached_ids(second) == [req_id]

        done = RunnerOutput(req_id=req_id, step_index=2, finished=True, result=DiffusionOutput(output=None))
        finished = self.scheduler.update_from_runner_output(second, done)
        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED

    def test_empty_output_without_error_marks_completed(self) -> None:
        req_id = self.scheduler.add_request(_make_request("empty"))

        sched_output = self.scheduler.schedule()
        finished = self.scheduler.update_from_output(sched_output, _make_request_output(req_id))

        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED

    def test_fifo_single_request_scheduling(self) -> None:
        req_id_a = self.scheduler.add_request(_make_request("a"))
        req_id_b = self.scheduler.add_request(_make_request("b"))

        first = self.scheduler.schedule()
        assert _new_ids(first) == [req_id_a]
        assert _cached_ids(first) == []
        assert first.num_running_reqs == 1
        assert first.num_waiting_reqs == 1

        # Request A is still running; scheduling again should not pull B.
        second = self.scheduler.schedule()
        assert _new_ids(second) == []
        assert _cached_ids(second) == [req_id_a]
        assert second.num_running_reqs == 1
        assert second.num_waiting_reqs == 1

        self.scheduler.update_from_output(first, _make_request_output(req_id_a))

        third = self.scheduler.schedule()
        assert _new_ids(third) == [req_id_b]
        assert _cached_ids(third) == []
        assert third.num_running_reqs == 1
        assert third.num_waiting_reqs == 0

    def test_abort_request_for_waiting_and_running(self) -> None:
        req_id_a = self.scheduler.add_request(_make_request("a"))
        req_id_b = self.scheduler.add_request(_make_request("b"))

        # Abort waiting request.
        self.scheduler.finish_requests(req_id_b, DiffusionRequestStatus.FINISHED_ABORTED)
        state_b = self.scheduler.get_request_state(req_id_b)
        assert state_b.status == DiffusionRequestStatus.FINISHED_ABORTED

        # A should still run normally.
        output_a = self.scheduler.schedule()
        assert _new_ids(output_a) == [req_id_a]

        # Abort running request.
        self.scheduler.finish_requests(req_id_a, DiffusionRequestStatus.FINISHED_ABORTED)
        state_a = self.scheduler.get_request_state(req_id_a)
        assert state_a.status == DiffusionRequestStatus.FINISHED_ABORTED

        assert self.scheduler.has_requests() is False
        assert self.scheduler.schedule().scheduled_req_ids == []

    def test_has_requests_state_transition(self) -> None:
        assert self.scheduler.has_requests() is False

        req_id = self.scheduler.add_request(_make_request("has"))
        assert self.scheduler.has_requests() is True

        sched_output = self.scheduler.schedule()
        assert self.scheduler.has_requests() is True

        self.scheduler.update_from_output(sched_output, _make_request_output(req_id))
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        assert self.scheduler.has_requests() is False

    def test_request_id_mapping_lifecycle(self) -> None:
        request = OmniDiffusionRequest(
            prompts=["prompt_map_a", "prompt_map_b"],
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
            request_ids=["map-a", "map-b"],
        )

        sched_req_id = self.scheduler.add_request(request)

        assert self.scheduler.get_sched_req_id("map-a") == sched_req_id
        assert self.scheduler.get_sched_req_id("map-b") == sched_req_id

        self.scheduler.pop_request_state(sched_req_id)

        assert self.scheduler.get_sched_req_id("map-a") is None
        assert self.scheduler.get_sched_req_id("map-b") is None

    def test_estimate_service_time_uses_wan_profile_for_video_requests(self) -> None:
        request = OmniDiffusionRequest(
            prompts=["prompt_video"],
            sampling_params=OmniDiffusionSamplingParams(
                width=854,
                height=480,
                num_inference_steps=3,
                num_frames=80,
            ),
            request_ids=["video"],
        )

        assert estimate_service_time_s(request) == pytest.approx(38.07)

    def test_normalize_wan_num_frames_matches_pipeline_contract(self) -> None:
        assert normalize_wan2_2_num_frames(80) == 81
        assert normalize_wan2_2_num_frames(81) == 81
        assert normalize_wan2_2_num_frames(82) == 81


class TestSuperP95StepScheduler:
    def setup_method(self) -> None:
        self.scheduler = SuperP95StepScheduler()
        self.scheduler.initialize(Mock())

    def test_prefers_longer_normal_request_over_shorter_normal(self) -> None:
        short_id = self.scheduler.add_request(_make_super_request("short", estimated_service_s=5.0))
        long_id = self.scheduler.add_request(_make_super_request("long", estimated_service_s=20.0))

        sched_output = self.scheduler.schedule()
        assert _new_ids(sched_output) == [long_id]
        assert short_id != long_id

    def test_prefers_normal_request_over_sacrificial(self) -> None:
        sacrificial_id = self.scheduler.add_request(
            _make_super_request("s", estimated_service_s=20.0, sacrificial=True)
        )
        normal_id = self.scheduler.add_request(_make_super_request("n", estimated_service_s=5.0, sacrificial=False))

        sched_output = self.scheduler.schedule()
        assert _new_ids(sched_output) == [normal_id]
        assert sacrificial_id != normal_id


class TestDiffusionEngine:
    def test_add_req_and_wait_for_response_single_path(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.scheduler = RequestScheduler()
        engine.scheduler.initialize(Mock())
        engine.executor = Mock()
        engine._rpc_lock = threading.Lock()

        request = _make_request("engine")
        expected = DiffusionOutput(output=None)
        engine.executor.add_req.return_value = expected

        output = engine.add_req_and_wait_for_response(request)

        assert output is expected
        engine.executor.add_req.assert_called_once_with(request)

    def test_supports_scheduler_interface_injection(self) -> None:
        request = _make_request("engine_iface")
        expected = DiffusionOutput(output=None)
        scheduler = _StubScheduler(request, expected)

        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.scheduler = scheduler
        engine.executor = Mock()
        engine.executor.add_req = Mock(return_value=expected)
        engine._rpc_lock = threading.Lock()

        output = engine.add_req_and_wait_for_response(request)

        assert output is expected
        engine.executor.add_req.assert_called_once_with(request)

    def test_add_req_and_wait_for_response_step_execution_path(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.od_config = Mock(step_execution=True)
        engine.scheduler = RequestScheduler()
        engine.scheduler.initialize(Mock())
        engine.executor = Mock()
        engine._rpc_lock = threading.Lock()

        request = _make_request("step-engine")
        expected = DiffusionOutput(output=None)
        engine.executor.execute_stepwise.side_effect = [
            RunnerOutput(req_id="step-engine", step_index=1, finished=False, result=None),
            RunnerOutput(req_id="step-engine", step_index=2, finished=True, result=expected),
        ]

        output = engine.add_req_and_wait_for_response(request)

        assert output is expected
        assert engine.executor.execute_stepwise.call_count == 2

    def test_step_execution_allows_new_arrival_while_first_step_is_running(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.od_config = Mock(step_execution=True)
        engine.scheduler = RequestScheduler()
        engine.scheduler.initialize(Mock())
        engine.executor = Mock()
        engine._rpc_lock = threading.Lock()

        first_started = threading.Event()
        second_enqueued = threading.Event()
        first_output = DiffusionOutput(output=None)
        second_output = DiffusionOutput(output=None)

        original_add_request = engine.scheduler.add_request

        def tracked_add_request(request: OmniDiffusionRequest) -> str:
            sched_req_id = original_add_request(request)
            if request.request_ids == ["second"]:
                second_enqueued.set()
            return sched_req_id

        engine.scheduler.add_request = tracked_add_request

        def execute_stepwise(sched_output):
            req_id = sched_output.scheduled_req_ids[0]
            if req_id == "first":
                first_started.set()
                assert second_enqueued.wait(timeout=2.0) is True
                return RunnerOutput(req_id="first", step_index=1, finished=True, result=first_output)
            assert req_id == "second"
            return RunnerOutput(req_id="second", step_index=1, finished=True, result=second_output)

        engine.executor.execute_stepwise.side_effect = execute_stepwise

        results: dict[str, DiffusionOutput] = {}

        def run_first() -> None:
            results["first"] = engine.add_req_and_wait_for_response(_make_request("first"))

        def run_second() -> None:
            assert first_started.wait(timeout=2.0) is True
            results["second"] = engine.add_req_and_wait_for_response(_make_request("second"))

        first_thread = threading.Thread(target=run_first, daemon=True)
        second_thread = threading.Thread(target=run_second, daemon=True)
        first_thread.start()
        second_thread.start()
        first_thread.join(timeout=5.0)
        second_thread.join(timeout=5.0)

        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        assert results["first"] is first_output
        assert results["second"] is second_output

    def test_initializes_injected_scheduler(self) -> None:
        request = _make_request("init")
        scheduler = _StubScheduler(request, DiffusionOutput(output=None))
        od_config = Mock(model_class_name="mock_model")
        fake_executor_cls = Mock(return_value=Mock())

        with (
            patch("vllm_omni.diffusion.diffusion_engine.get_diffusion_post_process_func", return_value=None),
            patch("vllm_omni.diffusion.diffusion_engine.get_diffusion_pre_process_func", return_value=None),
            patch("vllm_omni.diffusion.diffusion_engine.DiffusionExecutor.get_class", return_value=fake_executor_cls),
            patch.object(DiffusionEngine, "_dummy_run", return_value=None),
        ):
            DiffusionEngine(od_config, scheduler=scheduler)

        assert scheduler.initialized_with is od_config
        fake_executor_cls.assert_called_once_with(od_config)

    def test_scheduler_alias_keeps_default_request_scheduler(self) -> None:
        scheduler = Scheduler()
        scheduler.initialize(Mock())

        req_id = scheduler.add_request(_make_request("alias"))
        sched_output = scheduler.schedule()
        finished = scheduler.update_from_output(sched_output, _make_request_output(req_id))

        assert req_id in finished
        assert scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED

    def test_make_default_scheduler_uses_super_p95_step_env(self, monkeypatch) -> None:
        od_config = Mock(step_execution=False)
        monkeypatch.setenv("VLLM_OMNI_DIFFUSION_SCHEDULER", "super_p95_step")

        scheduler = _make_default_scheduler(od_config)

        assert isinstance(scheduler, SuperP95StepScheduler)
        assert od_config.step_execution is True

    def test_dummy_run_raises_on_output_error(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.od_config = Mock(model_class_name="mock_model")
        engine.pre_process_func = None
        engine.add_req_and_wait_for_response = Mock(return_value=DiffusionOutput(error="boom"))

        with pytest.raises(RuntimeError, match="Dummy run failed: boom"):
            engine._dummy_run()
