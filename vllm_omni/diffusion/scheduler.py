# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import threading
from multiprocessing.synchronize import Event as MpEvent
from typing import Any

import zmq
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.server_scheduling import (
    PredictedLatencyPolicy,
    ScheduledRequest,
    estimate_service_time_s,
    resolve_super_p95_hardware_profile,
)
from vllm_omni.diffusion.super_p95 import (
    SuperP95LoadSnapshot,
    get_super_p95_request_metadata,
)

logger = init_logger(__name__)


class Scheduler:
    def initialize(
        self,
        od_config: OmniDiffusionConfig,
        preempt_event: MpEvent | None = None,
        active_completed_steps: Any | None = None,
    ):
        existing_mq = getattr(self, "mq", None)
        if existing_mq is not None and not existing_mq.closed:
            logger.warning("SyncSchedulerClient is already initialized. Re-initializing.")
            self.close()

        self.num_workers = od_config.num_gpus
        self.od_config = od_config
        self._lock = threading.Lock()
        scheduling_env = os.environ.get("VLLM_OMNI_ENABLE_DIFFUSION_SERVER_SCHEDULING", "1")
        self._server_scheduling_enabled = scheduling_env.strip().lower() not in {"0", "false", "off", "no"}
        self._pending_cv = threading.Condition()
        self._closed = False
        self._policy = None
        self._result_lock = threading.Lock()
        self._result_cv = threading.Condition(self._result_lock)
        self._pending_results: dict[str, DiffusionOutput] = {}
        self._reader_error: Exception | None = None
        self._direct_normal_load_s = 0.0
        self._direct_sacrificial_load_s = 0.0
        preemption_env = os.environ.get("VLLM_OMNI_ENABLE_DIFFUSION_PREEMPTION", "1")
        self._preemption_enabled = preemption_env.strip().lower() not in {"0", "false", "off", "no"}
        self._active_scheduled: ScheduledRequest | None = None
        self._active_request_preemptible = False
        self._active_preemption_requested = False
        self._preempt_event = preempt_event
        self._active_completed_steps = active_completed_steps

        if self._server_scheduling_enabled:
            self._policy = PredictedLatencyPolicy(
                hardware_profile=resolve_super_p95_hardware_profile(
                    od_config.super_p95_hardware_profile,
                    warn_on_default=True,
                    context="server",
                )
            )
            logger.info("Diffusion server-side scheduling is enabled.")
        else:
            logger.warning(
                "Diffusion server-side scheduling is disabled; requests will execute in direct arrival order."
            )

        # Initialize single MessageQueue for all message types (generation & RPC)
        # Assuming all readers are local for now as per current launch_engine implementation
        self.mq = MessageQueue(
            n_reader=self.num_workers,
            n_local_reader=self.num_workers,
            local_reader_ranks=list(range(self.num_workers)),
        )
        self.result_mq = None
        self._scheduler_thread = None
        if self._server_scheduling_enabled:
            self._scheduler_thread = threading.Thread(
                target=self._run_scheduler_loop,
                name="DiffusionPredictedLatencyScheduler",
                daemon=True,
            )
            self._scheduler_thread.start()

    def initialize_result_queue(self, handle):
        # Initialize MessageQueue for receiving results
        # We act as rank 0 reader for this queue
        self.result_mq = MessageQueue.create_from_handle(handle, rank=0)
        logger.info("SyncScheduler initialized result MessageQueue")

    def publish_result(self, request_key: str, output: DiffusionOutput, source: str = "external") -> None:
        with self._result_cv:
            logger.info(
                "Scheduler publishing result from %s: request_key=%s finished=%s error=%s",
                source,
                request_key,
                output.finished,
                bool(output.error),
            )
            self._pending_results[request_key] = output
            self._result_cv.notify_all()

    def set_reader_error(self, error: Exception) -> None:
        with self._result_cv:
            self._reader_error = error
            self._result_cv.notify_all()

    def get_broadcast_handle(self):
        return self.mq.export_handle()

    def add_req(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        """Queue a request and wait for server-side scheduling to execute it."""
        if not self._server_scheduling_enabled:
            estimated_service_s, is_sacrificial = self._estimate_request_load(request)
            with self._pending_cv:
                self._adjust_direct_load(is_sacrificial, estimated_service_s)
            try:
                return self._execute_request(request)
            finally:
                with self._pending_cv:
                    self._adjust_direct_load(is_sacrificial, -estimated_service_s)

        should_preempt = False
        with self._pending_cv:
            assert self._policy is not None
            scheduled = self._policy.add_request(
                request,
                arrival_time_s=self._get_effective_current_time_s_locked(),
            )
            should_preempt = self._maybe_request_active_preemption_locked()
            self._pending_cv.notify_all()
        if should_preempt and self._preempt_event is not None:
            self._preempt_event.set()
        scheduled.done_event.wait()
        if scheduled.error is not None:
            raise scheduled.error
        assert isinstance(scheduled.output, DiffusionOutput)
        return scheduled.output

    def _run_scheduler_loop(self) -> None:
        while True:
            with self._pending_cv:
                assert self._policy is not None
                self._pending_cv.wait_for(lambda: self._closed or self._policy.has_pending())
                if self._closed and not self._policy.has_pending():
                    return
                scheduled = self._policy.pop_next_request()
                self._mark_active_request_locked(scheduled)

            try:
                output = self._execute_request(scheduled.request)
                if output.finished:
                    scheduled.output = output
                    self._policy.mark_finished(scheduled)
                    scheduled.done_event.set()
                    continue

                scheduler_state = output.scheduler_state or {}
                completed_steps = int(scheduler_state.get("completed_steps", 0))
                if completed_steps <= 0:
                    raise RuntimeError("Preempted diffusion request did not report any completed steps.")
                self._policy.update_after_quantum(scheduled, completed_steps)
                with self._pending_cv:
                    self._policy.requeue_request(scheduled)
                    self._pending_cv.notify_all()
            except BaseException as exc:  # pragma: no cover - surfaced to caller
                scheduled.error = exc
                scheduled.done_event.set()
            finally:
                with self._pending_cv:
                    self._clear_active_request_locked()

    def _execute_request(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        with self._lock:
            try:
                supports_preemption = self._supports_async_preemption(request)
                request.sampling_params.extra_args["_server_preemption_enabled"] = supports_preemption
                request_key = self._get_request_key(request)
                rpc_request = {
                    "type": "rpc",
                    "method": "generate",
                    "args": (request,),
                    "kwargs": {},
                    "output_rank": 0,
                    "exec_all_ranks": True,
                }
                self.mq.enqueue(rpc_request)
                with self._result_cv:
                    while True:
                        if self._reader_error is not None:
                            raise self._reader_error
                        output = self._pending_results.pop(request_key, None)
                        if output is not None:
                            return output
                        self._result_cv.wait(timeout=0.1)
            except zmq.error.Again as exc:
                logger.error("Timeout waiting for response from scheduler.")
                raise TimeoutError("Scheduler did not respond in time.") from exc

    def _supports_async_preemption(self, request: OmniDiffusionRequest) -> bool:
        return (
            self._preemption_enabled
            and self.od_config.model_class_name in {
                "QwenImagePipeline",
                "Wan22Pipeline",
                "WanPipeline",
            }
            and bool(getattr(request, "request_ids", None))
        )

    @staticmethod
    def _get_request_key(request: OmniDiffusionRequest) -> str:
        request_ids = getattr(request, "request_ids", None)
        if request_ids:
            return request_ids[0]
        raise ValueError("Diffusion requests must carry a request_id.")

    def _mark_active_request_locked(self, scheduled: ScheduledRequest) -> None:
        request = scheduled.request
        preemptible = self._supports_async_preemption(request)
        if self._preempt_event is not None:
            self._preempt_event.clear()
        self._set_active_completed_steps(0)
        scheduled.dispatch_start_remaining_steps = max(int(scheduled.remaining_steps), 0)
        self._active_scheduled = scheduled
        self._active_request_preemptible = preemptible
        self._active_preemption_requested = False

    def _clear_active_request_locked(self) -> None:
        if self._preempt_event is not None:
            self._preempt_event.clear()
        self._set_active_completed_steps(0)
        self._active_scheduled = None
        self._active_request_preemptible = False
        self._active_preemption_requested = False

    def _maybe_request_active_preemption_locked(self) -> bool:
        if self._policy is None:
            return False
        if not (
            self._active_request_preemptible
            and self._active_scheduled is not None
            and not self._active_preemption_requested
        ):
            return False

        best_pending = self._policy.peek_next_request()
        if best_pending is None:
            return False

        active_remaining_s = self._get_active_remaining_service_s_locked()
        active = self._active_scheduled
        original_remaining_s = active.remaining_service_s
        original_sort_key = active.sort_key
        active.remaining_service_s = active_remaining_s
        active.refresh_sort_key()
        try:
            outranks = self._policy.request_outranks(best_pending, active)
        finally:
            active.remaining_service_s = original_remaining_s
            active.sort_key = original_sort_key

        if not outranks:
            return False

        logger.info(
            "Requesting async preemption: active=%s active_remaining=%.3fs pending=%s pending_remaining=%.3fs",
            self._active_scheduled.request.request_ids[0] if self._active_scheduled.request.request_ids else "<unknown>",
            self._active_scheduled.remaining_service_s,
            best_pending.request.request_ids[0] if best_pending.request.request_ids else "<unknown>",
            best_pending.remaining_service_s,
        )
        self._active_preemption_requested = True
        return True

    def _get_effective_current_time_s_locked(self) -> float:
        assert self._policy is not None
        return self._policy.current_time_s + self._get_active_elapsed_service_s_locked()

    def _get_active_elapsed_service_s_locked(self) -> float:
        active = self._active_scheduled
        if active is None:
            return 0.0

        completed_steps = self._get_active_completed_steps()
        if completed_steps <= 0:
            return 0.0

        total_steps = max(int(active.total_steps), 1)
        dispatch_start_remaining_steps = max(
            int(getattr(active, "dispatch_start_remaining_steps", active.remaining_steps)),
            0,
        )
        completed_in_quantum = min(completed_steps, dispatch_start_remaining_steps)
        return active.estimated_service_s * completed_in_quantum / total_steps

    def _get_active_remaining_service_s_locked(self) -> float:
        active = self._active_scheduled
        if active is None:
            return 0.0

        completed_steps = self._get_active_completed_steps()
        total_steps = max(int(active.total_steps), 1)
        if completed_steps <= 0:
            return active.remaining_service_s

        dispatch_start_remaining_steps = max(
            int(getattr(active, "dispatch_start_remaining_steps", active.remaining_steps)),
            0,
        )
        remaining_steps = max(dispatch_start_remaining_steps - min(completed_steps, dispatch_start_remaining_steps), 0)
        return active.estimated_service_s * remaining_steps / total_steps

    def _get_active_completed_steps(self) -> int:
        progress = getattr(self, "_active_completed_steps", None)
        if progress is None:
            return 0
        try:
            return max(int(progress.value), 0)
        except (AttributeError, TypeError, ValueError):
            return 0

    def _set_active_completed_steps(self, completed_steps: int) -> None:
        progress = getattr(self, "_active_completed_steps", None)
        if progress is None:
            return
        try:
            progress.value = max(int(completed_steps), 0)
        except (AttributeError, TypeError, ValueError):
            return

    def get_super_p95_load_snapshot(self) -> SuperP95LoadSnapshot:
        with self._pending_cv:
            if not self._server_scheduling_enabled or self._policy is None:
                return SuperP95LoadSnapshot(
                    normal_load_s=self._direct_normal_load_s,
                    sacrificial_load_s=self._direct_sacrificial_load_s,
                )
            snapshot = self._policy.get_pending_load_snapshot()
            if self._active_scheduled is None:
                return snapshot

            active_remaining_s = self._get_active_remaining_service_s_locked()
            if self._active_scheduled.is_sacrificial:
                return SuperP95LoadSnapshot(
                    normal_load_s=snapshot.normal_load_s,
                    sacrificial_load_s=snapshot.sacrificial_load_s + active_remaining_s,
                )
            return SuperP95LoadSnapshot(
                normal_load_s=snapshot.normal_load_s + active_remaining_s,
                sacrificial_load_s=snapshot.sacrificial_load_s,
            )

    def close(self):
        """Closes the socket and terminates the context."""
        with self._pending_cv:
            self._closed = True
            self._pending_cv.notify_all()
        scheduler_thread = getattr(self, "_scheduler_thread", None)
        if scheduler_thread is not None:
            scheduler_thread.join(timeout=1)
        self.mq = None
        self.result_mq = None
        self._pending_results = {}
        self._reader_error = None

    def _estimate_request_load(self, request: OmniDiffusionRequest) -> tuple[float, bool]:
        extra_args = getattr(request.sampling_params, "extra_args", None)
        is_sacrificial, estimated_service_s = get_super_p95_request_metadata(extra_args)
        if estimated_service_s is None:
            estimated_service_s = estimate_service_time_s(
                request,
                hardware_profile=resolve_super_p95_hardware_profile(
                    self.od_config.super_p95_hardware_profile,
                    warn_on_default=True,
                    context="server",
                ),
            )
        return estimated_service_s, is_sacrificial

    def _adjust_direct_load(self, is_sacrificial: bool, delta_s: float) -> None:
        if is_sacrificial:
            self._direct_sacrificial_load_s = max(self._direct_sacrificial_load_s + delta_s, 0.0)
        else:
            self._direct_normal_load_s = max(self._direct_normal_load_s + delta_s, 0.0)
