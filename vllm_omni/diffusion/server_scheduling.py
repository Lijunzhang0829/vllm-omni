# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
import threading
from dataclasses import dataclass, field

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.super_p95 import (
    SuperP95LoadSnapshot,
    get_super_p95_request_metadata,
)

# Qwen-Image service-time anchors from benchmarks/diffusion/super_p95_strategy.md.
_KNOWN_SERVICE_TIME_ANCHORS_S: dict[tuple[int, int], tuple[int, float]] = {
    (512, 512): (20, 22.35),
    (768, 768): (20, 20.62),
    (1024, 1024): (25, 33.90),
    (1536, 1536): (35, 102.66),
}

_FALLBACK_BASE_RESOLUTION = (1024, 1024)
_FALLBACK_BASE_STEPS = 25
_FALLBACK_BASE_SERVICE_S = 33.90


def estimate_service_time_s(request: OmniDiffusionRequest) -> float:
    params = request.sampling_params
    width = int(params.width or params.resolution or _FALLBACK_BASE_RESOLUTION[0])
    height = int(params.height or params.resolution or _FALLBACK_BASE_RESOLUTION[1])
    steps = max(int(params.num_inference_steps or _FALLBACK_BASE_STEPS), 1)
    frames = max(int(params.num_frames or 1), 1)

    anchor = _KNOWN_SERVICE_TIME_ANCHORS_S.get((width, height))
    if anchor is not None:
        anchor_steps, anchor_latency_s = anchor
        return anchor_latency_s * (steps / anchor_steps) * frames

    return (
        _FALLBACK_BASE_SERVICE_S
        * (width * height * steps * frames)
        / (
            _FALLBACK_BASE_RESOLUTION[0]
            * _FALLBACK_BASE_RESOLUTION[1]
            * _FALLBACK_BASE_STEPS
        )
    )


@dataclass(order=True)
class ScheduledRequest:
    sort_key: tuple[float, int] = field(init=False, repr=False)
    arrival_seq: int
    arrival_time_s: float
    request: OmniDiffusionRequest = field(compare=False)
    estimated_service_s: float = field(compare=False)
    remaining_service_s: float = field(compare=False)
    total_steps: int = field(compare=False, default=1)
    remaining_steps: int = field(compare=False, default=1)
    is_sacrificial: bool = field(compare=False, default=False)
    output: object | None = field(default=None, compare=False)
    error: BaseException | None = field(default=None, compare=False)
    done_event: threading.Event = field(default_factory=threading.Event, compare=False, repr=False)

    def __post_init__(self) -> None:
        self.refresh_sort_key()

    def refresh_sort_key(self) -> None:
        priority = self.remaining_service_s - self.arrival_time_s
        if self.is_sacrificial:
            self.sort_key = (priority, self.arrival_seq)
        else:
            # heapq is min-heap, so negate the priority score to emulate argmax.
            self.sort_key = (-priority, self.arrival_seq)


class PredictedLatencyPolicy:
    def __init__(self) -> None:
        self._normal_pending: list[ScheduledRequest] = []
        self._sacrificial_pending: list[ScheduledRequest] = []
        self._arrival_seq = 0
        self.current_time_s = 0.0
        self._normal_load_s = 0.0
        self._sacrificial_load_s = 0.0

    def add_request(self, request: OmniDiffusionRequest) -> ScheduledRequest:
        extra_args = getattr(request.sampling_params, "extra_args", None)
        is_sacrificial, estimated_service_s = get_super_p95_request_metadata(extra_args)
        if estimated_service_s is None:
            estimated_service_s = estimate_service_time_s(request)
        total_steps = max(int(request.sampling_params.num_inference_steps or 1), 1)
        scheduled = ScheduledRequest(
            arrival_seq=self._arrival_seq,
            arrival_time_s=self.current_time_s,
            request=request,
            estimated_service_s=estimated_service_s,
            remaining_service_s=estimated_service_s,
            total_steps=total_steps,
            remaining_steps=total_steps,
            is_sacrificial=is_sacrificial,
        )
        self._arrival_seq += 1
        self._push_pending(scheduled)
        return scheduled

    def has_pending(self) -> bool:
        return bool(self._normal_pending or self._sacrificial_pending)

    def pop_next_request(self) -> ScheduledRequest:
        if self._normal_pending:
            return self._pop_heap(self._normal_pending, is_sacrificial=False)
        return self._pop_heap(self._sacrificial_pending, is_sacrificial=True)

    def peek_next_request(self) -> ScheduledRequest | None:
        if self._normal_pending:
            return self._normal_pending[0]
        if self._sacrificial_pending:
            return self._sacrificial_pending[0]
        return None

    def requeue_request(self, scheduled: ScheduledRequest) -> None:
        scheduled.refresh_sort_key()
        self._push_pending(scheduled)

    def update_after_quantum(self, scheduled: ScheduledRequest, completed_steps: int) -> None:
        completed_steps = max(completed_steps, 0)
        self.current_time_s += scheduled.estimated_service_s * completed_steps / scheduled.total_steps
        scheduled.remaining_steps = max(scheduled.remaining_steps - completed_steps, 0)
        if scheduled.remaining_steps > 0:
            scheduled.remaining_service_s = (
                scheduled.estimated_service_s * scheduled.remaining_steps / scheduled.total_steps
            )
        else:
            scheduled.remaining_service_s = 0.0

    def mark_finished(self, scheduled: ScheduledRequest) -> None:
        self.current_time_s += scheduled.remaining_service_s
        scheduled.remaining_steps = 0
        scheduled.remaining_service_s = 0.0

    def get_pending_load_snapshot(self) -> SuperP95LoadSnapshot:
        return SuperP95LoadSnapshot(
            normal_load_s=self._normal_load_s,
            sacrificial_load_s=self._sacrificial_load_s,
        )

    @staticmethod
    def request_outranks(candidate: ScheduledRequest, incumbent: ScheduledRequest) -> bool:
        if candidate.is_sacrificial != incumbent.is_sacrificial:
            return not candidate.is_sacrificial and incumbent.is_sacrificial
        return candidate.sort_key < incumbent.sort_key

    def _push_pending(self, scheduled: ScheduledRequest) -> None:
        heap = self._sacrificial_pending if scheduled.is_sacrificial else self._normal_pending
        heapq.heappush(heap, scheduled)
        self._adjust_pending_load(scheduled.is_sacrificial, scheduled.remaining_service_s)

    def _pop_heap(self, heap: list[ScheduledRequest], *, is_sacrificial: bool) -> ScheduledRequest:
        scheduled = heapq.heappop(heap)
        self._adjust_pending_load(is_sacrificial, -scheduled.remaining_service_s)
        return scheduled

    def _adjust_pending_load(self, is_sacrificial: bool, delta_s: float) -> None:
        if is_sacrificial:
            self._sacrificial_load_s = max(self._sacrificial_load_s + delta_s, 0.0)
        else:
            self._normal_load_s = max(self._normal_load_s + delta_s, 0.0)
