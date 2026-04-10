# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
import os
from dataclasses import dataclass, field

from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import _BaseScheduler
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionRequestState,
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    NewRequestData,
)
from vllm_omni.diffusion.super_p95 import (
    estimate_service_time_s,
    normalize_super_p95_hardware_profile,
    SuperP95LoadSnapshot,
)
logger = init_logger(__name__)


@dataclass(order=True)
class _QueuedRequest:
    sort_key: tuple[float, int] = field(init=False, repr=False)
    arrival_seq: int
    arrival_time_s: float
    sched_req_id: str = field(compare=False)
    estimated_service_s: float = field(compare=False)
    total_steps: int = field(compare=False, default=1)
    completed_steps: int = field(compare=False, default=0)
    dispatch_start_completed_steps: int = field(compare=False, default=0)
    is_sacrificial: bool = field(compare=False, default=False)

    def __post_init__(self) -> None:
        self.refresh_sort_key(self.estimated_service_s)

    def refresh_sort_key(self, remaining_service_s: float) -> None:
        priority = remaining_service_s - self.arrival_time_s
        if self.is_sacrificial:
            self.sort_key = (priority, self.arrival_seq)
        else:
            self.sort_key = (-priority, self.arrival_seq)

    @property
    def remaining_service_s(self) -> float:
        total_steps = max(self.total_steps, 1)
        remaining_steps = max(total_steps - self.completed_steps, 0)
        return self.estimated_service_s * remaining_steps / total_steps

    @property
    def service_per_step_s(self) -> float:
        return self.estimated_service_s / max(self.total_steps, 1)


class SuperP95RequestScheduler(_BaseScheduler):
    """A minimal v0.18-compatible scheduler variant for super-p95 ordering."""

    def __init__(self) -> None:
        super().__init__()
        self._normal_pending: list[_QueuedRequest] = []
        self._sacrificial_pending: list[_QueuedRequest] = []
        self._queue_metadata: dict[str, _QueuedRequest] = {}
        self._arrival_seq: int = 0
        self._current_time_s: float = 0.0
        self._hardware_profile = normalize_super_p95_hardware_profile(
            os.environ.get("VLLM_OMNI_SUPER_P95_HARDWARE_PROFILE")
        )

    def get_load_snapshot(self) -> SuperP95LoadSnapshot:
        normal_load_s = 0.0
        sacrificial_load_s = 0.0
        for sched_req_id in [*self._waiting, *self._running]:
            queued = self._queue_metadata.get(sched_req_id)
            if queued is None:
                continue
            if queued.is_sacrificial:
                sacrificial_load_s += queued.remaining_service_s
            else:
                normal_load_s += queued.remaining_service_s
        return SuperP95LoadSnapshot(
            normal_load_s=normal_load_s,
            sacrificial_load_s=sacrificial_load_s,
        )

    def add_request(self, request: OmniDiffusionRequest) -> str:
        sched_req_id = self._make_sched_req_id(request)
        state = DiffusionRequestState(sched_req_id=sched_req_id, req=request)
        self._request_states[sched_req_id] = state
        self._register_request_ids(request.request_ids, sched_req_id)
        self._waiting.append(sched_req_id)

        estimated_service_s = request.super_p95_estimated_service_s
        if estimated_service_s is None:
            estimated_service_s = estimate_service_time_s(request, hardware_profile=self._hardware_profile)
        queued = _QueuedRequest(
            arrival_seq=self._arrival_seq,
            arrival_time_s=self._current_time_s,
            sched_req_id=sched_req_id,
            estimated_service_s=estimated_service_s,
            total_steps=max(int(request.sampling_params.num_inference_steps or 1), 1),
            is_sacrificial=request.super_p95_sacrificial,
        )
        self._arrival_seq += 1
        self._queue_metadata[sched_req_id] = queued
        self._push_pending(queued)
        logger.debug(
            "SuperP95 add_request: %s sacrificial=%s estimated_service_s=%.3f",
            sched_req_id,
            request.super_p95_sacrificial,
            estimated_service_s,
        )
        return sched_req_id

    def schedule(self) -> DiffusionSchedulerOutput:
        scheduled_new_reqs: list[NewRequestData] = []
        scheduled_cached_req_ids: list[str] = []

        for sched_req_id in self._running:
            state = self._request_states.get(sched_req_id)
            if state is not None:
                scheduled_cached_req_ids.append(sched_req_id)

        while (self._normal_pending or self._sacrificial_pending) and len(self._running) < self._max_batch_size:
            queued = self._pop_next_queued()
            if queued is None:
                break
            sched_req_id = queued.sched_req_id
            state = self._request_states.get(sched_req_id)
            if state is None:
                continue
            try:
                self._waiting.remove(sched_req_id)
            except ValueError:
                continue
            was_new_request = state.status == DiffusionRequestStatus.WAITING
            state.status = DiffusionRequestStatus.RUNNING
            self._running.append(sched_req_id)
            if was_new_request:
                scheduled_new_reqs.append(NewRequestData.from_state(state))
            else:
                scheduled_cached_req_ids.append(sched_req_id)

        scheduler_output = DiffusionSchedulerOutput(
            step_id=self._step_id,
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=CachedRequestData(sched_req_ids=scheduled_cached_req_ids),
            finished_req_ids=set(self._finished_req_ids),
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._waiting),
        )

        self._step_id += 1
        self._finished_req_ids.clear()
        return scheduler_output

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: DiffusionOutput) -> set[str]:
        scheduled_req_ids = sched_output.scheduled_req_ids
        if not scheduled_req_ids:
            return set()

        finished_req_ids = {
            sched_req_id for sched_req_id in scheduled_req_ids if sched_req_id in self._finished_req_ids
        }
        terminal_statuses: dict[str, DiffusionRequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}
        for sched_req_id in scheduled_req_ids:
            state = self._request_states.get(sched_req_id)
            if state is None or state.is_finished():
                continue
            queued = self._queue_metadata.get(sched_req_id)
            if queued is not None and not output.finished:
                self._apply_partial_progress(queued, output)
                if sched_req_id in self._running:
                    self._running.remove(sched_req_id)
                self._waiting.appendleft(sched_req_id)
                state.status = DiffusionRequestStatus.PREEMPTED
                self._push_pending(queued)
                continue
            if queued is not None:
                self._current_time_s += queued.remaining_service_s
                queued.completed_steps = queued.total_steps
            if output.error:
                terminal_statuses[sched_req_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[sched_req_id] = output.error
            else:
                terminal_statuses[sched_req_id] = DiffusionRequestStatus.FINISHED_COMPLETED
                terminal_errors[sched_req_id] = None

        finished_req_ids |= self._finish_requests(terminal_statuses, terminal_errors)
        return finished_req_ids

    def abort_request(self, sched_req_id: str) -> bool:
        if self.get_request_state(sched_req_id) is None:
            return False
        self.finish_requests(sched_req_id, DiffusionRequestStatus.FINISHED_ABORTED)
        return True

    def preempt_request(self, sched_req_id: str) -> bool:
        if sched_req_id not in self._request_states:
            return False
        if sched_req_id in self._running:
            self._running.remove(sched_req_id)
            self._waiting.appendleft(sched_req_id)
            state = self._request_states[sched_req_id]
            state.status = DiffusionRequestStatus.PREEMPTED
            queued = self._queue_metadata.get(sched_req_id)
            if queued is not None:
                self._push_pending(queued)
            return True
        return False

    def _reset_scheduler_state(self) -> None:
        self._normal_pending.clear()
        self._sacrificial_pending.clear()
        self._queue_metadata.clear()
        self._arrival_seq = 0
        self._current_time_s = 0.0

    def _pop_extra_request_state(self, sched_req_id: str) -> None:
        self._queue_metadata.pop(sched_req_id, None)

    def _push_pending(self, queued: _QueuedRequest) -> None:
        queued.refresh_sort_key(queued.remaining_service_s)
        heap = self._sacrificial_pending if queued.is_sacrificial else self._normal_pending
        heapq.heappush(heap, queued)

    def _pop_next_queued(self) -> _QueuedRequest | None:
        while self._normal_pending or self._sacrificial_pending:
            heap = self._normal_pending if self._normal_pending else self._sacrificial_pending
            queued = heapq.heappop(heap)
            state = self._request_states.get(queued.sched_req_id)
            if state is None or state.is_finished():
                continue
            queued.dispatch_start_completed_steps = queued.completed_steps
            return queued
        return None

    def peek_next_request(self) -> _QueuedRequest | None:
        for heap in (self._normal_pending, self._sacrificial_pending):
            while heap:
                queued = heap[0]
                state = self._request_states.get(queued.sched_req_id)
                if state is None or state.is_finished():
                    heapq.heappop(heap)
                    continue
                return queued
        return None

    @staticmethod
    def request_outranks(candidate: _QueuedRequest, incumbent: _QueuedRequest) -> bool:
        if candidate.is_sacrificial != incumbent.is_sacrificial:
            return not candidate.is_sacrificial and incumbent.is_sacrificial
        return candidate.sort_key < incumbent.sort_key

    def get_queued_request(self, sched_req_id: str) -> _QueuedRequest | None:
        return self._queue_metadata.get(sched_req_id)

    def get_active_remaining_service_s(self, sched_req_id: str, completed_steps: int) -> float:
        queued = self._queue_metadata.get(sched_req_id)
        if queued is None:
            return 0.0
        dispatch_start_remaining_steps = max(queued.total_steps - queued.dispatch_start_completed_steps, 0)
        completed_in_quantum = min(max(completed_steps, 0), dispatch_start_remaining_steps)
        remaining_steps = max(dispatch_start_remaining_steps - completed_in_quantum, 0)
        return queued.estimated_service_s * remaining_steps / max(queued.total_steps, 1)

    def _apply_partial_progress(self, queued: _QueuedRequest, output: DiffusionOutput) -> None:
        scheduler_state = output.scheduler_state or {}
        completed_steps = max(int(scheduler_state.get("completed_steps", 0)), 0)
        if completed_steps <= 0:
            raise RuntimeError("Async-preempted diffusion request did not report any completed steps.")
        dispatch_start_remaining_steps = max(queued.total_steps - queued.dispatch_start_completed_steps, 0)
        completed_in_quantum = min(completed_steps, dispatch_start_remaining_steps)
        self._current_time_s += queued.estimated_service_s * completed_in_quantum / max(queued.total_steps, 1)
        queued.completed_steps = min(queued.dispatch_start_completed_steps + completed_in_quantum, queued.total_steps)
