# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import threading
import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import PIL.Image
import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.registry import (
    DiffusionModelRegistry,
    get_diffusion_post_process_func,
    get_diffusion_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import RequestScheduler, SchedulerInterface, SuperP95RequestScheduler
from vllm_omni.diffusion.super_p95 import snapshot_to_metrics
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


def _server_scheduling_enabled() -> bool:
    value = os.environ.get("VLLM_OMNI_ENABLE_DIFFUSION_SERVER_SCHEDULING", "0")
    return value.strip().lower() not in {"0", "false", "off", "no"}


def _preemption_enabled() -> bool:
    value = os.environ.get("VLLM_OMNI_ENABLE_DIFFUSION_PREEMPTION", "0")
    return value.strip().lower() not in {"0", "false", "off", "no"}


def supports_image_input(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_image_input", False))


def supports_audio_input(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_audio_input", False))


def image_color_format(model_class_name: str) -> str:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    return getattr(model_cls, "color_format", "RGB")


def supports_audio_output(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_audio_output", False))


class DiffusionEngine:
    """The diffusion engine for vLLM-Omni diffusion models."""

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        scheduler: SchedulerInterface | None = None,
    ):
        """Initialize the diffusion engine.

        Args:
            config: The configuration for the diffusion engine.
        """
        self.od_config = od_config

        self.post_process_func = get_diffusion_post_process_func(od_config)
        self.pre_process_func = get_diffusion_pre_process_func(od_config)

        executor_class = DiffusionExecutor.get_class(od_config)
        self.executor = executor_class(od_config)
        if scheduler is None:
            scheduler = SuperP95RequestScheduler() if _server_scheduling_enabled() else RequestScheduler()
        self.scheduler = scheduler
        self.scheduler.initialize(od_config)
        self._rpc_lock = threading.Lock()
        self._scheduler_cv = threading.Condition()
        self._async_outputs: dict[str, DiffusionOutput] = {}
        self._async_errors: dict[str, BaseException] = {}
        self._closed = False
        self._active_sched_req_id: str | None = None
        self._active_preemption_requested = False
        self._scheduler_thread: threading.Thread | None = None
        if self._uses_async_super_p95():
            self._scheduler_thread = threading.Thread(
                target=self._run_super_p95_scheduler_loop,
                name="DiffusionSuperP95Scheduler",
                daemon=True,
            )
            self._scheduler_thread.start()

        try:
            self._dummy_run()
        except Exception as e:
            logger.error(f"Dummy run failed: {e}")
            self.close()
            raise e

    def step(self, request: OmniDiffusionRequest) -> list[OmniRequestOutput]:
        diffusion_engine_start_time = time.perf_counter()

        # Apply pre-processing if available
        preprocess_time = 0.0
        if self.pre_process_func is not None:
            preprocess_start_time = time.perf_counter()
            request = self.pre_process_func(request)
            preprocess_time = time.perf_counter() - preprocess_start_time
            logger.info(f"Pre-processing completed in {preprocess_time:.4f} seconds")

        exec_start_time = time.perf_counter()
        output = self.add_req_and_wait_for_response(request)
        exec_total_time = time.perf_counter() - exec_start_time

        if output.error:
            raise Exception(f"{output.error}")
        logger.info("Generation completed successfully.")

        if output.output is None:
            logger.warning("Output is None, returning empty OmniRequestOutput")
            return [
                OmniRequestOutput.from_diffusion(
                    request_id=request.request_ids[i] if i < len(request.request_ids) else "",
                    images=[],
                    prompt=prompt,
                    metrics={},
                    latents=None,
                )
                for i, prompt in enumerate(request.prompts)
            ]

        # When CPU offload is enabled, move output to CPU before
        # post-processing to avoid device OOM — model weights may still
        # reside on the device and leave no headroom for intermediates.
        output_data = output.output
        if (
            self.od_config.enable_cpu_offload
            and isinstance(output_data, torch.Tensor)
            and output_data.device.type != "cpu"
        ):
            output_data = output_data.cpu()

        postprocess_start_time = time.perf_counter()
        outputs = self.post_process_func(output_data) if self.post_process_func is not None else output_data
        audio_payload = None
        if isinstance(outputs, dict):
            audio_payload = outputs.get("audio")
            outputs = outputs.get("video", outputs)
        postprocess_time = time.perf_counter() - postprocess_start_time
        logger.info(f"Post-processing completed in {postprocess_time:.4f} seconds")

        step_total_ms = (time.perf_counter() - diffusion_engine_start_time) * 1000
        logger.info(
            "DiffusionEngine.step breakdown: preprocess=%.2f ms, "
            "add_req_and_wait=%.2f ms, postprocess=%.2f ms, total=%.2f ms",
            preprocess_time * 1000,
            exec_total_time * 1000,
            postprocess_time * 1000,
            step_total_ms,
        )

        # Convert to OmniRequestOutput format
        # Ensure outputs is a list
        if not isinstance(outputs, list):
            outputs = [outputs] if outputs is not None else []

        metrics = {
            "preprocess_time_ms": preprocess_time * 1000,
            "diffusion_engine_exec_time_ms": (time.perf_counter() - diffusion_engine_start_time) * 1000,
            "diffusion_engine_total_time_ms": exec_total_time * 1000,
            "image_num": int(request.sampling_params.num_outputs_per_prompt),
            "resolution": int(request.sampling_params.resolution),
            "postprocess_time_ms": postprocess_time * 1000,
        }
        load_snapshot = self.scheduler.get_load_snapshot()
        if load_snapshot is not None:
            metrics.update(snapshot_to_metrics(load_snapshot))
        if self.pre_process_func is not None:
            metrics["preprocessing_time_ms"] = preprocess_time * 1000

        # Handle single request or multiple requests
        if len(request.prompts) == 1:
            # Single request: return single OmniRequestOutput
            prompt = request.prompts[0]
            request_id = request.request_ids[0] if request.request_ids else ""

            if supports_audio_output(self.od_config.model_class_name):
                request_audio_payload = outputs[0] if len(outputs) == 1 else outputs
                return [
                    OmniRequestOutput.from_diffusion(
                        request_id=request_id,
                        images=[],
                        prompt=prompt,
                        metrics=metrics,
                        latents=output.trajectory_latents,
                        multimodal_output={"audio": request_audio_payload},
                        final_output_type="audio",
                        stage_durations=output.stage_durations,
                        peak_memory_mb=output.peak_memory_mb,
                    ),
                ]
            else:
                mm_output = {}
                if audio_payload is not None:
                    mm_output["audio"] = audio_payload
                return [
                    OmniRequestOutput.from_diffusion(
                        request_id=request_id,
                        images=outputs,
                        prompt=prompt,
                        metrics=metrics,
                        latents=output.trajectory_latents,
                        custom_output=output.custom_output or {},
                        multimodal_output=mm_output,
                        stage_durations=output.stage_durations,
                        peak_memory_mb=output.peak_memory_mb,
                    ),
                ]
        else:
            # Multiple requests: return list of OmniRequestOutput
            # Split images based on num_outputs_per_prompt for each request
            results = []
            output_idx = 0

            for i, prompt in enumerate(request.prompts):
                request_id = request.request_ids[i] if i < len(request.request_ids) else ""

                # Get images for this request
                num_outputs = request.sampling_params.num_outputs_per_prompt
                start_idx = output_idx
                end_idx = start_idx + num_outputs
                request_outputs = outputs[start_idx:end_idx] if output_idx < len(outputs) else []
                output_idx = end_idx

                if supports_audio_output(self.od_config.model_class_name):
                    request_audio_payload = request_outputs[0] if len(request_outputs) == 1 else request_outputs
                    results.append(
                        OmniRequestOutput.from_diffusion(
                            request_id=request_id,
                            images=[],
                            prompt=prompt,
                            metrics=metrics,
                            latents=output.trajectory_latents,
                            multimodal_output={"audio": request_audio_payload},
                            final_output_type="audio",
                            stage_durations=output.stage_durations,
                            peak_memory_mb=output.peak_memory_mb,
                        ),
                    )
                else:
                    mm_output = {}
                    if audio_payload is not None:
                        sliced_audio = audio_payload
                        if isinstance(audio_payload, (list, tuple)):
                            sliced_audio = audio_payload[start_idx:end_idx]
                            if len(sliced_audio) == 1:
                                sliced_audio = sliced_audio[0]
                        elif hasattr(audio_payload, "shape") and getattr(audio_payload, "shape", None) is not None:
                            if len(audio_payload.shape) > 0 and audio_payload.shape[0] >= end_idx:
                                sliced_audio = audio_payload[start_idx:end_idx]
                                if num_outputs == 1:
                                    sliced_audio = sliced_audio[0]
                        mm_output["audio"] = sliced_audio
                    results.append(
                        OmniRequestOutput.from_diffusion(
                            request_id=request_id,
                            images=request_outputs,
                            prompt=prompt,
                            metrics=metrics,
                            latents=output.trajectory_latents,
                            custom_output=output.custom_output or {},
                            multimodal_output=mm_output,
                            stage_durations=output.stage_durations,
                            peak_memory_mb=output.peak_memory_mb,
                        ),
                    )

            return results

    @staticmethod
    def make_engine(
        config: OmniDiffusionConfig,
        scheduler: SchedulerInterface | None = None,
    ) -> "DiffusionEngine":
        """Factory method to create a DiffusionEngine instance.

        Args:
            config: The configuration for the diffusion engine.

        Returns:
            An instance of DiffusionEngine.
        """
        return DiffusionEngine(config, scheduler=scheduler)

    def add_req_and_wait_for_response(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        if self._uses_async_super_p95():
            return self._add_req_and_wait_async_super_p95(request)

        with self._rpc_lock:
            target_sched_req_id = self.scheduler.add_request(request)

            # keep scheduling and executing until the target request is finished
            while True:
                sched_output = self.scheduler.schedule()
                if sched_output.is_empty:
                    if not self.scheduler.has_requests():
                        raise RuntimeError("Diffusion scheduler has no runnable requests.")
                    continue

                # NOTE: add_req_and_wait_for_response() is synchronous, and
                # the scheduler currently enforces _max_batch_size = 1 (see
                # vllm_omni/diffusion/sched/base_scheduler.py), so we directly
                # take the single scheduled request here.
                sched_req_id = sched_output.scheduled_req_ids[0]
                req = sched_output.scheduled_new_reqs[0].req
                try:
                    output = self.executor.add_req(req)
                except Exception as exc:
                    logger.error(
                        "Execution failed for diffusion request %s",
                        sched_req_id,
                        exc_info=True,
                    )
                    output = DiffusionOutput(error=str(exc))

                finished_req_ids = self.scheduler.update_from_output(sched_output, output)
                if target_sched_req_id in finished_req_ids:
                    self.scheduler.pop_request_state(target_sched_req_id)
                    return output

    def _uses_async_super_p95(self) -> bool:
        return (
            isinstance(self.scheduler, SuperP95RequestScheduler)
            and _server_scheduling_enabled()
        )

    def _add_req_and_wait_async_super_p95(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        with self._scheduler_cv:
            target_sched_req_id = self.scheduler.add_request(request)
            should_preempt = self._maybe_request_active_preemption_locked()
            self._scheduler_cv.notify_all()
        if should_preempt:
            preempt_event = getattr(self.executor, "_preempt_event", None)
            if preempt_event is not None:
                preempt_event.set()

        with self._scheduler_cv:
            while True:
                error = self._async_errors.pop(target_sched_req_id, None)
                if error is not None:
                    self.scheduler.pop_request_state(target_sched_req_id)
                    raise error
                output = self._async_outputs.pop(target_sched_req_id, None)
                if output is not None:
                    self.scheduler.pop_request_state(target_sched_req_id)
                    return output
                self._scheduler_cv.wait(timeout=0.1)

    def _run_super_p95_scheduler_loop(self) -> None:
        while True:
            with self._scheduler_cv:
                self._scheduler_cv.wait_for(lambda: self._closed or self.scheduler.has_requests())
                if self._closed and not self.scheduler.has_requests():
                    return
                sched_output = self.scheduler.schedule()
                if sched_output.is_empty:
                    continue
                sched_req_id = sched_output.scheduled_req_ids[0]
                state = self.scheduler.get_request_state(sched_req_id)
                if state is None:
                    continue
                request = state.req
                request.sampling_params.extra_args["_server_preemption_enabled"] = self._request_supports_async_preemption(
                    request
                )
                self._mark_active_request_locked(sched_req_id)

            try:
                with self._rpc_lock:
                    output = self.executor.add_req(request)
            except Exception as exc:  # pragma: no cover - surfaced to caller
                logger.error("Async super-p95 execution failed for %s", sched_req_id, exc_info=True)
                output = DiffusionOutput(error=str(exc), request_key=request.request_ids[0] if request.request_ids else None)

            with self._scheduler_cv:
                finished_req_ids = self.scheduler.update_from_output(sched_output, output)
                if sched_req_id in finished_req_ids:
                    self._async_outputs[sched_req_id] = output
                    self._scheduler_cv.notify_all()
                self._clear_active_request_locked()

    def _request_supports_async_preemption(self, request: OmniDiffusionRequest) -> bool:
        model_name = self.od_config.model_class_name
        if not _preemption_enabled() or not bool(getattr(request, "request_ids", None)):
            return False
        if model_name == "QwenImagePipeline":
            return True
        if model_name == "Wan22Pipeline":
            prompts = getattr(request, "prompts", None) or []
            if len(prompts) != 1 or isinstance(prompts[0], str):
                return True
            multi_modal_data = prompts[0].get("multi_modal_data", {})
            return multi_modal_data.get("image") is None
        return (
            # TODO(super-p95-v018): widen async preemption support once Wan
            # image-conditioned request-mode save/load parity is implemented.
            False
        )

    def _mark_active_request_locked(self, sched_req_id: str) -> None:
        self._active_sched_req_id = sched_req_id
        self._active_preemption_requested = False
        preempt_event = getattr(self.executor, "_preempt_event", None)
        if preempt_event is not None:
            preempt_event.clear()
        progress = getattr(self.executor, "_active_completed_steps", None)
        if progress is not None:
            progress.value = 0

    def _clear_active_request_locked(self) -> None:
        preempt_event = getattr(self.executor, "_preempt_event", None)
        if preempt_event is not None:
            preempt_event.clear()
        progress = getattr(self.executor, "_active_completed_steps", None)
        if progress is not None:
            progress.value = 0
        self._active_sched_req_id = None
        self._active_preemption_requested = False

    def _maybe_request_active_preemption_locked(self) -> bool:
        if not _preemption_enabled() or self._active_preemption_requested:
            return False
        if not isinstance(self.scheduler, SuperP95RequestScheduler):
            return False
        active_sched_req_id = self._active_sched_req_id
        if active_sched_req_id is None:
            return False
        active_state = self.scheduler.get_request_state(active_sched_req_id)
        if active_state is None or not self._request_supports_async_preemption(active_state.req):
            return False
        candidate = self.scheduler.peek_next_request()
        incumbent = self.scheduler.get_queued_request(active_sched_req_id)
        if candidate is None or incumbent is None:
            return False
        progress = getattr(self.executor, "_active_completed_steps", None)
        completed_steps = 0 if progress is None else max(int(progress.value), 0)
        active_remaining_s = self.scheduler.get_active_remaining_service_s(active_sched_req_id, completed_steps)
        original_sort_key = incumbent.sort_key
        incumbent.refresh_sort_key(active_remaining_s)
        try:
            outranks = self.scheduler.request_outranks(candidate, incumbent)
        finally:
            incumbent.sort_key = original_sort_key
        if not outranks:
            return False
        self._active_preemption_requested = True
        return True

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        """Start or stop torch profiling on all diffusion workers.

        Args:
            is_start: True to start profiling, False to stop.
            profile_prefix: Optional prefix for trace filename (vLLM compat).

        Note:
            Matches vLLM's worker.profile() signature for consistency.
            Traces are saved automatically via on_trace_ready callback.
        """
        if is_start:
            if profile_prefix is None:
                profile_prefix = f"diffusion_{int(time.time())}"
            logger.info(f"Starting diffusion profiling with prefix: {profile_prefix}")
        else:
            logger.info("Stopping diffusion profiling...")

        try:
            self.collective_rpc(method="profile", args=(is_start, profile_prefix))
        except Exception as e:
            action = "start" if is_start else "stop"
            logger.error(f"Failed to {action} profiling on workers", exc_info=True)
            if is_start:
                raise RuntimeError(f"Could not {action} profiler: {e}") from e

    def _dummy_run(self):
        """A dummy run to warm up the model."""
        num_inference_steps = 1
        height = 1024
        width = 1024
        if supports_image_input(self.od_config.model_class_name):
            # Provide a dummy image input if the model supports it
            color_format = image_color_format(self.od_config.model_class_name)
            dummy_image = PIL.Image.new(color_format, (width, height))
        else:
            dummy_image = None

        if supports_audio_input(self.od_config.model_class_name):
            audio_sr = 16000
            audio_duration_sec = 4
            audio_array = np.random.randn(audio_sr * audio_duration_sec).astype(np.float32)
            dummy_audio = audio_array[audio_sr * 1 : audio_sr * 3]
        else:
            dummy_audio = None

        prompt: OmniTextPrompt = {
            "prompt": "dummy run",
            "multi_modal_data": {"image": dummy_image, "audio": dummy_audio},
        }
        req = OmniDiffusionRequest(
            prompts=[prompt],
            request_ids=["dummy_req_id"],
            sampling_params=OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                # Keep warmup path minimal and robust across text encoders.
                # Some models may fail when warmup implicitly triggers
                # classifier-free guidance with an empty negative prompt.
                guidance_scale=0.0,
                num_outputs_per_prompt=1,
                # Disable CFG for warmup to avoid triggering CFG parallel
                # validation when cfg_parallel_size > 1.
                extra_args={"cfg_text_scale": 1.0, "cfg_img_scale": 1.0},
            ),
        )
        logger.info("dummy run to warm up the model")
        request = self.pre_process_func(req) if self.pre_process_func is not None else req
        output = self.add_req_and_wait_for_response(request)
        if output.error:
            raise RuntimeError(f"Dummy run failed: {output.error}")

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Call a method on worker processes and get results immediately.

        Args:
            method: The method name (str) to execute on workers
            timeout: Optional timeout in seconds
            args: Positional arguments for the method
            kwargs: Keyword arguments for the method
            unique_reply_rank: If set, only get reply from this rank

        Returns:
            Single result if unique_reply_rank is provided, otherwise list of results
        """
        assert isinstance(method, str), "Only string method names are supported for now"

        deadline = None if timeout is None else time.monotonic() + timeout
        acquired = False
        try:
            if deadline is None:
                self._rpc_lock.acquire()
                acquired = True
            else:
                lock_timeout = max(0, deadline - time.monotonic())
                acquired = self._rpc_lock.acquire(timeout=lock_timeout)
            if not acquired:
                raise TimeoutError(f"RPC call to {method} timed out waiting for engine lock.")

            rpc_timeout = None if deadline is None else max(0, deadline - time.monotonic())
            if deadline is not None and rpc_timeout <= 0:
                raise TimeoutError(f"RPC call to {method} timed out.")

            return self.executor.collective_rpc(
                method=method,
                timeout=rpc_timeout,
                args=args,
                kwargs=kwargs,
                unique_reply_rank=unique_reply_rank,
            )
        finally:
            if acquired:
                self._rpc_lock.release()

    def close(self) -> None:
        self._closed = True
        if hasattr(self, "_scheduler_cv"):
            with self._scheduler_cv:
                self._scheduler_cv.notify_all()
        if getattr(self, "_scheduler_thread", None) is not None:
            self._scheduler_thread.join(timeout=5)
        if hasattr(self, "scheduler"):
            self.scheduler.close()
        if hasattr(self, "executor"):
            self.executor.shutdown()

    def abort(self, request_id: str | Iterable[str]) -> None:
        # TODO implement it
        logger.warning("DiffusionEngine abort is not implemented yet")
        pass
