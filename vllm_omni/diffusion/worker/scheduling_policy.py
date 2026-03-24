# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduling policy abstractions for diffusion worker request ordering."""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Protocol

from vllm_omni.diffusion.request import OmniDiffusionRequest


DEFAULT_SECONDS_PER_COST_S = 33.9 / float(1024 * 1024 * 25)


def req_debug_id(req: OmniDiffusionRequest) -> str:
    return req.request_ids[0] if req.request_ids else req.request_key


def req_step_index(req: OmniDiffusionRequest) -> int:
    if req.execution_state is None:
        return 0
    return int(req.execution_state.step_index)


def estimate_remaining_cost(req: OmniDiffusionRequest) -> float:
    """Simple estimator: remaining_steps * width * height * frames."""
    total_steps = int(req.sampling_params.num_inference_steps or 0)
    done_steps = req_step_index(req)
    remaining_steps = max(0, total_steps - done_steps)
    width = int(req.sampling_params.width or 1024)
    height = int(req.sampling_params.height or 1024)
    num_frames = int(req.sampling_params.num_frames or 1)
    return float(remaining_steps * width * height * num_frames)


def estimate_total_cost(req: OmniDiffusionRequest) -> float:
    """Simple estimator: total_steps * width * height * frames."""
    total_steps = max(0, int(req.sampling_params.num_inference_steps or 0))
    width = int(req.sampling_params.width or 1024)
    height = int(req.sampling_params.height or 1024)
    num_frames = int(req.sampling_params.num_frames or 1)
    return float(total_steps * width * height * num_frames)


@dataclass
class PreemptionEvent:
    preempted_req: OmniDiffusionRequest
    selected_req: OmniDiffusionRequest
    preempted_cost: float
    selected_cost: float


@dataclass
class ArrivalDecision:
    current_req: OmniDiffusionRequest | None
    preemption: PreemptionEvent | None = None


class DiffusionSchedulingPolicy(Protocol):
    def on_request_arrival(
        self, req: OmniDiffusionRequest, current_req: OmniDiffusionRequest | None
    ) -> ArrivalDecision:
        ...

    def on_request_finish(self, finished_req: OmniDiffusionRequest | None = None) -> OmniDiffusionRequest | None:
        ...

    def defer_request(self, req: OmniDiffusionRequest) -> None:
        ...

    def on_request_complete(self, req: OmniDiffusionRequest) -> None:
        ...

    def observe_execution(self, req: OmniDiffusionRequest, elapsed_s: float, work_done: float) -> None:
        ...

    def abort_request_ids(self, request_ids: list[str]) -> None:
        ...

    def is_aborted(self, request_id: str) -> bool:
        ...

    def consume_recent_dropped_request_ids(self) -> list[str]:
        ...

    def consume_recent_sacrificial_request_ids(self) -> list[str]:
        ...


class TargetFreeGlobalReorderPolicy:
    """Global queue reorder policy with arrival/finish scheduling points."""

    def __init__(self, aging_alpha: float = 0.0, aging_cap: float = 8.0, aging_cost_ref: float = 1.0) -> None:
        self._waiting_reqs: list[tuple[float, int, OmniDiffusionRequest]] = []
        self._queued_request_keys: set[str] = set()
        self._queued_arrival_work: dict[str, float] = {}
        self._queue_seq: int = 0
        self._cumulative_arrival_work: float = 0.0
        self._aborted_request_ids: set[str] = set()
        self._recent_dropped_request_ids: list[str] = []
        self._aging_alpha = max(0.0, float(aging_alpha))
        self._aging_cap = max(0.0, float(aging_cap))
        self._aging_cost_ref = max(1.0, float(aging_cost_ref))

    def on_request_arrival(
        self, req: OmniDiffusionRequest, current_req: OmniDiffusionRequest | None
    ) -> ArrivalDecision:
        self._cumulative_arrival_work += estimate_total_cost(req)
        req_id = req_debug_id(req)
        if req_id in self._aborted_request_ids:
            self._recent_dropped_request_ids.append(req_id)
            return ArrivalDecision(current_req=current_req)

        if current_req is None and not self._waiting_reqs:
            return ArrivalDecision(current_req=req)

        self._queue_request(req)
        candidate = self._pop_best_waiting_request()
        if candidate is None:
            return ArrivalDecision(current_req=current_req)
        if current_req is None:
            return ArrivalDecision(current_req=candidate)

        current_cost = self._priority_cost(current_req)
        candidate_cost = self._priority_cost(candidate)
        if candidate_cost < current_cost:
            self._queue_request(current_req)
            return ArrivalDecision(
                current_req=candidate,
                preemption=PreemptionEvent(
                    preempted_req=current_req,
                    selected_req=candidate,
                    preempted_cost=estimate_remaining_cost(current_req),
                    selected_cost=estimate_remaining_cost(candidate),
                ),
            )

        self._queue_request(candidate)
        return ArrivalDecision(current_req=current_req)

    def on_request_finish(self, finished_req: OmniDiffusionRequest | None = None) -> OmniDiffusionRequest | None:
        if finished_req is not None:
            self.on_request_complete(finished_req)
        return self._pop_best_waiting_request()

    def defer_request(self, req: OmniDiffusionRequest) -> None:
        self._queue_request(req)

    def on_request_complete(self, req: OmniDiffusionRequest) -> None:
        self._queued_arrival_work.pop(req.request_key, None)

    def observe_execution(self, req: OmniDiffusionRequest, elapsed_s: float, work_done: float) -> None:
        del req, elapsed_s, work_done

    def abort_request_ids(self, request_ids: list[str]) -> None:
        for rid in request_ids:
            if isinstance(rid, str):
                self._aborted_request_ids.add(rid)

    def is_aborted(self, request_id: str) -> bool:
        return request_id in self._aborted_request_ids

    def consume_recent_dropped_request_ids(self) -> list[str]:
        dropped = self._recent_dropped_request_ids
        self._recent_dropped_request_ids = []
        return dropped

    def consume_recent_sacrificial_request_ids(self) -> list[str]:
        return []

    def _queue_request(self, req: OmniDiffusionRequest) -> None:
        req_id = req_debug_id(req)
        if req_id in self._aborted_request_ids:
            self._recent_dropped_request_ids.append(req_id)
            return
        if req.request_key in self._queued_request_keys:
            return
        cost = self._priority_cost(req)
        heapq.heappush(self._waiting_reqs, (cost, self._queue_seq, req))
        self._queued_request_keys.add(req.request_key)
        self._queued_arrival_work[req.request_key] = self._cumulative_arrival_work
        self._queue_seq += 1

    def _pop_best_waiting_request(self) -> OmniDiffusionRequest | None:
        self._refresh_waiting_heap()
        while self._waiting_reqs:
            _, _, req = heapq.heappop(self._waiting_reqs)
            self._queued_request_keys.discard(req.request_key)
            self._queued_arrival_work.pop(req.request_key, None)
            req_id = req_debug_id(req)
            if req_id in self._aborted_request_ids:
                self._recent_dropped_request_ids.append(req_id)
                continue
            return req
        return None

    def _priority_cost(self, req: OmniDiffusionRequest) -> float:
        remaining_cost = estimate_remaining_cost(req)
        if remaining_cost <= 0:
            return 0.0
        if self._aging_alpha <= 0:
            return remaining_cost

        queued_arrival_work = self._queued_arrival_work.get(req.request_key, self._cumulative_arrival_work)
        waited_work = max(0.0, self._cumulative_arrival_work - queued_arrival_work)
        waited_work_units = waited_work / self._aging_cost_ref
        aged_units = min(waited_work_units, self._aging_cap)
        aging_boost = 1.0 + self._aging_alpha * aged_units
        return remaining_cost / aging_boost

    def _refresh_waiting_heap(self) -> None:
        if len(self._waiting_reqs) <= 1 or self._aging_alpha <= 0:
            return

        refreshed_heap: list[tuple[float, int, OmniDiffusionRequest]] = []
        while self._waiting_reqs:
            _, queue_seq, req = heapq.heappop(self._waiting_reqs)
            if req.request_key not in self._queued_request_keys:
                continue
            refreshed_heap.append((self._priority_cost(req), queue_seq, req))
        heapq.heapify(refreshed_heap)
        self._waiting_reqs = refreshed_heap


class DelayXPolicy:
    """Delay-X policy: prioritize the request with the largest predicted delay.

    Every N arrivals, mark one existing request in the local queue as sacrificial.
    Sacrificial requests are scheduled after normal requests, while preserving
    their internal priority order.
    """

    def __init__(
        self,
        quota_every: int = 20,
        quota_amount: int = 1,
        tail_penalty: float = 100.0,
        initial_seconds_per_cost_s: float = DEFAULT_SECONDS_PER_COST_S,
    ) -> None:
        self._waiting_reqs: list[OmniDiffusionRequest] = []
        self._queued_request_keys: set[str] = set()
        self._arrival_time_s: dict[str, float] = {}
        self._sacrificial_request_keys: set[str] = set()
        self._arrival_counter: int = 0
        self._aborted_request_ids: set[str] = set()
        self._recent_dropped_request_ids: list[str] = []
        self._recent_sacrificial_request_ids: list[str] = []
        self._quota_every = max(0, int(quota_every))
        self._quota_amount = max(0, int(quota_amount))
        self._tail_penalty = max(1.0, float(tail_penalty))
        self._seconds_per_cost_ema = max(1e-9, float(initial_seconds_per_cost_s))

    def on_request_arrival(
        self, req: OmniDiffusionRequest, current_req: OmniDiffusionRequest | None
    ) -> ArrivalDecision:
        now_s = time.perf_counter()
        req_id = req_debug_id(req)
        if req_id in self._aborted_request_ids:
            self._recent_dropped_request_ids.append(req_id)
            return ArrivalDecision(current_req=current_req)

        self._arrival_time_s.setdefault(req.request_key, now_s)
        self._arrival_counter += 1
        if self._quota_every > 0 and self._arrival_counter % self._quota_every == 0:
            for _ in range(self._quota_amount):
                self._issue_sacrificial_quota(current_req, now_s)

        if current_req is None and not self._waiting_reqs:
            return ArrivalDecision(current_req=req)

        self._queue_request(req)
        candidate = self._pop_best_waiting_request(now_s)
        if candidate is None:
            return ArrivalDecision(current_req=current_req)
        if current_req is None:
            return ArrivalDecision(current_req=candidate)

        current_priority = self._priority_tuple(current_req, now_s)
        candidate_priority = self._priority_tuple(candidate, now_s)
        if candidate_priority < current_priority:
            self._queue_request(current_req)
            return ArrivalDecision(
                current_req=candidate,
                preemption=PreemptionEvent(
                    preempted_req=current_req,
                    selected_req=candidate,
                    preempted_cost=estimate_remaining_cost(current_req),
                    selected_cost=estimate_remaining_cost(candidate),
                ),
            )

        self._queue_request(candidate)
        return ArrivalDecision(current_req=current_req)

    def on_request_finish(self, finished_req: OmniDiffusionRequest | None = None) -> OmniDiffusionRequest | None:
        if finished_req is not None:
            self.on_request_complete(finished_req)
        return self._pop_best_waiting_request(time.perf_counter())

    def defer_request(self, req: OmniDiffusionRequest) -> None:
        self._queue_request(req)

    def on_request_complete(self, req: OmniDiffusionRequest) -> None:
        self._queued_request_keys.discard(req.request_key)
        self._arrival_time_s.pop(req.request_key, None)
        self._sacrificial_request_keys.discard(req.request_key)

    def observe_execution(self, req: OmniDiffusionRequest, elapsed_s: float, work_done: float) -> None:
        del req
        if elapsed_s <= 0 or work_done <= 0:
            return
        observed_seconds_per_cost = elapsed_s / work_done
        self._seconds_per_cost_ema = 0.9 * self._seconds_per_cost_ema + 0.1 * observed_seconds_per_cost

    def abort_request_ids(self, request_ids: list[str]) -> None:
        for rid in request_ids:
            if isinstance(rid, str):
                self._aborted_request_ids.add(rid)

    def is_aborted(self, request_id: str) -> bool:
        return request_id in self._aborted_request_ids

    def consume_recent_dropped_request_ids(self) -> list[str]:
        dropped = self._recent_dropped_request_ids
        self._recent_dropped_request_ids = []
        return dropped

    def consume_recent_sacrificial_request_ids(self) -> list[str]:
        sacrificial = self._recent_sacrificial_request_ids
        self._recent_sacrificial_request_ids = []
        return sacrificial

    def _queue_request(self, req: OmniDiffusionRequest) -> None:
        req_id = req_debug_id(req)
        if req_id in self._aborted_request_ids:
            self._recent_dropped_request_ids.append(req_id)
            self.on_request_complete(req)
            return
        if req.request_key in self._queued_request_keys:
            return
        self._arrival_time_s.setdefault(req.request_key, time.perf_counter())
        self._waiting_reqs.append(req)
        self._queued_request_keys.add(req.request_key)

    def _pop_best_waiting_request(self, now_s: float) -> OmniDiffusionRequest | None:
        while self._waiting_reqs:
            best_index = min(range(len(self._waiting_reqs)), key=lambda idx: self._priority_tuple(self._waiting_reqs[idx], now_s))
            req = self._waiting_reqs.pop(best_index)
            self._queued_request_keys.discard(req.request_key)
            req_id = req_debug_id(req)
            if req_id in self._aborted_request_ids:
                self._recent_dropped_request_ids.append(req_id)
                self.on_request_complete(req)
                continue
            return req
        return None

    def _remaining_service_s(self, req: OmniDiffusionRequest) -> float:
        remaining_cost = estimate_remaining_cost(req)
        if remaining_cost <= 0:
            return 0.0
        return remaining_cost * self._seconds_per_cost_ema

    def _predicted_latency_s(self, req: OmniDiffusionRequest, now_s: float) -> float:
        arrival_time_s = self._arrival_time_s.get(req.request_key, now_s)
        return max(0.0, now_s - arrival_time_s) + self._remaining_service_s(req)

    def _base_priority_tuple(self, req: OmniDiffusionRequest, now_s: float) -> tuple[float, float, str]:
        return (-self._predicted_latency_s(req, now_s), self._arrival_time_s.get(req.request_key, now_s), req.request_key)

    def _priority_tuple(self, req: OmniDiffusionRequest, now_s: float) -> tuple[int, float, float, str]:
        predicted_latency_s = self._predicted_latency_s(req, now_s)
        sacrificial = req.request_key in self._sacrificial_request_keys
        scaled_latency_s = predicted_latency_s / self._tail_penalty if sacrificial else predicted_latency_s
        return (
            1 if sacrificial else 0,
            -scaled_latency_s,
            self._arrival_time_s.get(req.request_key, now_s),
            req.request_key,
        )

    def _issue_sacrificial_quota(self, current_req: OmniDiffusionRequest | None, now_s: float) -> None:
        candidate = self._select_sacrificial_candidate(current_req, now_s)
        if candidate is None:
            return
        if candidate.request_key in self._sacrificial_request_keys:
            return
        self._sacrificial_request_keys.add(candidate.request_key)
        self._recent_sacrificial_request_ids.append(req_debug_id(candidate))

    def _select_sacrificial_candidate(
        self,
        current_req: OmniDiffusionRequest | None,
        now_s: float,
    ) -> OmniDiffusionRequest | None:
        candidates: list[OmniDiffusionRequest] = []
        if current_req is not None and req_debug_id(current_req) not in self._aborted_request_ids:
            if current_req.request_key not in self._sacrificial_request_keys:
                candidates.append(current_req)
        for req in self._waiting_reqs:
            if req.request_key in self._sacrificial_request_keys:
                continue
            if req_debug_id(req) in self._aborted_request_ids:
                continue
            candidates.append(req)
        if not candidates:
            return None

        ordered = sorted(candidates, key=lambda req: self._base_priority_tuple(req, now_s))
        cumulative_wait_s = 0.0
        selected: OmniDiffusionRequest | None = None
        selected_latency_s = -1.0
        for req in ordered:
            remaining_service_s = self._remaining_service_s(req)
            arrival_time_s = self._arrival_time_s.get(req.request_key, now_s)
            predicted_completion_latency_s = max(0.0, now_s - arrival_time_s) + cumulative_wait_s + remaining_service_s
            if predicted_completion_latency_s > selected_latency_s:
                selected = req
                selected_latency_s = predicted_completion_latency_s
            cumulative_wait_s += remaining_service_s
        return selected
