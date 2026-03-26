# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
import threading
from dataclasses import dataclass, field

from vllm_omni.diffusion.request import OmniDiffusionRequest

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
    output: object | None = field(default=None, compare=False)
    error: BaseException | None = field(default=None, compare=False)
    done_event: threading.Event = field(default_factory=threading.Event, compare=False, repr=False)

    def __post_init__(self) -> None:
        self.refresh_sort_key()

    def refresh_sort_key(self) -> None:
        # heapq is min-heap, so negate the priority score to emulate argmax.
        priority = self.remaining_service_s - self.arrival_time_s
        self.sort_key = (-priority, self.arrival_seq)


class PredictedLatencyPolicy:
    def __init__(self) -> None:
        self._pending: list[ScheduledRequest] = []
        self._arrival_seq = 0
        self.current_time_s = 0.0

    def add_request(self, request: OmniDiffusionRequest) -> ScheduledRequest:
        estimated_service_s = estimate_service_time_s(request)
        scheduled = ScheduledRequest(
            arrival_seq=self._arrival_seq,
            arrival_time_s=self.current_time_s,
            request=request,
            estimated_service_s=estimated_service_s,
            remaining_service_s=estimated_service_s,
        )
        self._arrival_seq += 1
        heapq.heappush(self._pending, scheduled)
        return scheduled

    def has_pending(self) -> bool:
        return bool(self._pending)

    def pop_next_request(self) -> ScheduledRequest:
        return heapq.heappop(self._pending)

    def mark_finished(self, scheduled: ScheduledRequest) -> None:
        self.current_time_s += scheduled.estimated_service_s
