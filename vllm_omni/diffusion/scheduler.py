# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading

import zmq
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.server_scheduling import PredictedLatencyPolicy

logger = init_logger(__name__)


class Scheduler:
    def initialize(self, od_config: OmniDiffusionConfig):
        existing_mq = getattr(self, "mq", None)
        if existing_mq is not None and not existing_mq.closed:
            logger.warning("SyncSchedulerClient is already initialized. Re-initializing.")
            self.close()

        self.num_workers = od_config.num_gpus
        self.od_config = od_config
        self._lock = threading.Lock()
        self._pending_cv = threading.Condition()
        self._closed = False
        self._policy = PredictedLatencyPolicy()
        self._step_budget = 1

        # Initialize single MessageQueue for all message types (generation & RPC)
        # Assuming all readers are local for now as per current launch_engine implementation
        self.mq = MessageQueue(
            n_reader=self.num_workers,
            n_local_reader=self.num_workers,
            local_reader_ranks=list(range(self.num_workers)),
        )

        self.result_mq = None
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

    def get_broadcast_handle(self):
        return self.mq.export_handle()

    def add_req(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        """Queue a request and wait for server-side scheduling to execute it."""
        with self._pending_cv:
            scheduled = self._policy.add_request(request)
            self._pending_cv.notify_all()
        scheduled.done_event.wait()
        if scheduled.error is not None:
            raise scheduled.error
        assert isinstance(scheduled.output, DiffusionOutput)
        return scheduled.output

    def _run_scheduler_loop(self) -> None:
        while True:
            with self._pending_cv:
                self._pending_cv.wait_for(lambda: self._closed or self._policy.has_pending())
                if self._closed and not self._policy.has_pending():
                    return
                scheduled = self._policy.pop_next_request()

            try:
                output = self._execute_request(scheduled.request)
                if output.finished:
                    scheduled.request.sampling_params.extra_args.pop("_server_state", None)
                    scheduled.output = output
                    self._policy.mark_finished(scheduled)
                    scheduled.done_event.set()
                    continue

                scheduler_state = output.scheduler_state or {}
                completed_steps = int(scheduler_state.get("completed_steps", 0))
                if completed_steps <= 0:
                    raise RuntimeError("Preempted diffusion request did not report any completed steps.")
                scheduled.request.sampling_params.extra_args["_server_state"] = scheduler_state
                self._policy.update_after_quantum(scheduled, completed_steps)
                with self._pending_cv:
                    self._policy.requeue_request(scheduled)
                    self._pending_cv.notify_all()
            except BaseException as exc:  # pragma: no cover - surfaced to caller
                scheduled.error = exc
                scheduled.done_event.set()

    def _execute_request(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        with self._lock:
            try:
                supports_preemption = self.od_config.model_class_name == "QwenImagePipeline"
                request.sampling_params.extra_args["_server_preemption_enabled"] = supports_preemption
                if supports_preemption:
                    request.sampling_params.extra_args["_server_step_budget"] = self._step_budget
                else:
                    request.sampling_params.extra_args.pop("_server_step_budget", None)
                    request.sampling_params.extra_args.pop("_server_state", None)
                rpc_request = {
                    "type": "rpc",
                    "method": "generate",
                    "args": (request,),
                    "kwargs": {},
                    "output_rank": 0,
                    "exec_all_ranks": True,
                }
                self.mq.enqueue(rpc_request)

                if self.result_mq is None:
                    raise RuntimeError("Result queue not initialized")

                output = self.result_mq.dequeue()
                if isinstance(output, dict) and output.get("status") == "error":
                    raise RuntimeError("worker error")
                return output
            except zmq.error.Again as exc:
                logger.error("Timeout waiting for response from scheduler.")
                raise TimeoutError("Scheduler did not respond in time.") from exc

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
