# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
import threading
from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.super_p95 import (
    SuperP95LoadSnapshot,
    get_super_p95_request_metadata,
)

logger = init_logger(__name__)

DEFAULT_SUPER_P95_HARDWARE_PROFILE = "910B2"
_KNOWN_SUPER_P95_HARDWARE_PROFILES = {"910B2", "910B3"}

# Qwen-Image service-time anchors for the supported NPU hardware profiles.
_HARDWARE_ANCHORS_S: dict[str, dict[tuple[int, int], tuple[int, float]]] = {
    "910B2": {
        (512, 512): (20, 8.60),
        (768, 768): (20, 8.94),
        (1024, 1024): (25, 14.22),
        (1536, 1536): (35, 43.22),
    },
    "910B3": {
        (512, 512): (20, 8.00),
        (768, 768): (20, 9.00),
        (1024, 1024): (25, 22.11),
        (1536, 1536): (35, 71.68),
    },
}

_FALLBACK_BASE_RESOLUTION = (1024, 1024)
_FALLBACK_BASE_STEPS = 25
_PER_PIXEL_STEP_MS: dict[str, dict[tuple[int, int], float]] = {
    hardware_profile: {
        resolution: (anchor_latency_s * 1000.0) / (resolution[0] * resolution[1] * anchor_steps)
        for resolution, (anchor_steps, anchor_latency_s) in anchors.items()
    }
    for hardware_profile, anchors in _HARDWARE_ANCHORS_S.items()
}
_WARNED_SUPER_P95_HARDWARE_MESSAGES: set[str] = set()


def normalize_super_p95_hardware_profile(hardware_profile: str | None) -> str:
    normalized = (hardware_profile or DEFAULT_SUPER_P95_HARDWARE_PROFILE).strip().upper()
    if normalized in _KNOWN_SUPER_P95_HARDWARE_PROFILES:
        return normalized
    return DEFAULT_SUPER_P95_HARDWARE_PROFILE


def resolve_super_p95_hardware_profile(
    hardware_profile: str | None,
    *,
    warn_on_default: bool = False,
    context: str = "super_p95",
) -> str:
    raw_value = hardware_profile.strip() if isinstance(hardware_profile, str) else None
    if not raw_value:
        if warn_on_default:
            _warn_super_p95_hardware_once(
                f"{context}: super_p95 hardware profile was not specified; defaulting to {DEFAULT_SUPER_P95_HARDWARE_PROFILE}."
            )
        return DEFAULT_SUPER_P95_HARDWARE_PROFILE

    normalized = raw_value.upper()
    if normalized in _KNOWN_SUPER_P95_HARDWARE_PROFILES:
        return normalized

    _warn_super_p95_hardware_once(
        f"{context}: unknown super_p95 hardware profile {hardware_profile!r}; "
        f"defaulting to {DEFAULT_SUPER_P95_HARDWARE_PROFILE}."
    )
    return DEFAULT_SUPER_P95_HARDWARE_PROFILE


def _warn_super_p95_hardware_once(message: str) -> None:
    if message in _WARNED_SUPER_P95_HARDWARE_MESSAGES:
        return
    _WARNED_SUPER_P95_HARDWARE_MESSAGES.add(message)
    logger.warning(message)


def estimate_service_time_s(
    request: OmniDiffusionRequest,
    hardware_profile: str | None = None,
) -> float:
    params = request.sampling_params
    width = int(params.width or params.resolution or _FALLBACK_BASE_RESOLUTION[0])
    height = int(params.height or params.resolution or _FALLBACK_BASE_RESOLUTION[1])
    steps = max(int(params.num_inference_steps or _FALLBACK_BASE_STEPS), 1)
    frames = max(int(params.num_frames or 1), 1)

    profile = normalize_super_p95_hardware_profile(hardware_profile)
    per_pixel_step_ms = _PER_PIXEL_STEP_MS[profile].get((width, height))
    if per_pixel_step_ms is None:
        per_pixel_step_ms = _PER_PIXEL_STEP_MS[profile][_FALLBACK_BASE_RESOLUTION]
    return per_pixel_step_ms * width * height * steps * frames / 1000.0


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
    def __init__(self, *, hardware_profile: str | None = None) -> None:
        self._normal_pending: list[ScheduledRequest] = []
        self._sacrificial_pending: list[ScheduledRequest] = []
        self._arrival_seq = 0
        self.current_time_s = 0.0
        self._normal_load_s = 0.0
        self._sacrificial_load_s = 0.0
        self._hardware_profile = normalize_super_p95_hardware_profile(hardware_profile)

    def add_request(self, request: OmniDiffusionRequest) -> ScheduledRequest:
        extra_args = getattr(request.sampling_params, "extra_args", None)
        is_sacrificial, estimated_service_s = get_super_p95_request_metadata(extra_args)
        if estimated_service_s is None:
            estimated_service_s = estimate_service_time_s(request, hardware_profile=self._hardware_profile)
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
