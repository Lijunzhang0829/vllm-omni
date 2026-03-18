# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Diffusion Worker for vLLM-Omni.

Handles GPU infrastructure initialization and delegates model operations
to DiffusionModelRunner.
"""

import gc
import multiprocessing as mp
import os
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime
from typing import Any

import torch
import torch.distributed
import zmq
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.mem_utils import GiB_bytes
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.diffusion.data import (
    DiffusionOutput,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.profiler import CurrentProfiler
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.scheduling_policy import (
    DiffusionSchedulingPolicy,
    TargetFreeGlobalReorderPolicy,
    req_debug_id,
    req_step_index,
)
from vllm_omni.lora.request import LoRARequest
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.gpu_memory_utils import get_process_gpu_memory

logger = init_logger(__name__)


class DiffusionWorker:
    """
    A worker that manages GPU infrastructure and delegates to the model runner.

    This class handles infrastructure initialization only:
    - Device setup (CUDA device selection)
    - Distributed environment (NCCL, model parallel)
    - Memory management (sleep/wake)

    All model-related operations (loading, compilation, execution) are
    delegated to DiffusionModelRunner.
    """

    def __init__(
        self,
        local_rank: int,
        rank: int,
        od_config: OmniDiffusionConfig,
    ):
        self.local_rank = local_rank
        self.rank = rank
        self.od_config = od_config
        self.device: torch.device | None = None
        self.vllm_config: VllmConfig | None = None
        self.model_runner: DiffusionModelRunner | None = None
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}
        self.lora_manager: DiffusionLoRAManager | None = None
        self.init_device()
        # Create model runner
        self.model_runner = DiffusionModelRunner(
            vllm_config=self.vllm_config,
            od_config=self.od_config,
            device=self.device,
        )
        self.load_model(load_format=self.od_config.diffusion_load_format)
        self.init_lora_manager()
        logger.info(f"Worker {self.rank}: Initialization complete.")

    def init_device(self) -> None:
        """Initialize the device and distributed environment."""
        world_size = self.od_config.num_gpus
        rank = self.rank

        # Set environment variables for distributed initialization
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.od_config.master_port)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Setup device
        self.device = current_omni_platform.get_torch_device(rank)
        current_omni_platform.set_device(self.device)

        # Create vllm_config for parallel configuration
        vllm_config = VllmConfig(compilation_config=CompilationConfig())
        vllm_config.parallel_config.tensor_parallel_size = self.od_config.parallel_config.tensor_parallel_size
        vllm_config.parallel_config.data_parallel_size = self.od_config.parallel_config.data_parallel_size
        self.vllm_config = vllm_config

        # Initialize distributed environment
        with (
            set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config),
            set_current_vllm_config(self.vllm_config),
        ):
            init_distributed_environment(world_size=world_size, rank=rank)
            logger.info(f"Worker {self.rank}: Initialized device and distributed environment.")

            parallel_config = self.od_config.parallel_config
            initialize_model_parallel(
                data_parallel_size=parallel_config.data_parallel_size,
                cfg_parallel_size=parallel_config.cfg_parallel_size,
                sequence_parallel_size=parallel_config.sequence_parallel_size,
                ulysses_degree=parallel_config.ulysses_degree,
                ring_degree=parallel_config.ring_degree,
                tensor_parallel_size=parallel_config.tensor_parallel_size,
                pipeline_parallel_size=parallel_config.pipeline_parallel_size,
                fully_shard_degree=parallel_config.hsdp_shard_size if parallel_config.use_hsdp else 1,
            )
            init_workspace_manager(self.device)

    def load_model(self, load_format: str = "default", custom_pipeline_name: str | None = None) -> None:
        """Load the diffusion model using DiffusionModelRunner."""
        with (
            set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config),
            set_current_vllm_config(self.vllm_config),
        ):
            self.model_runner.load_model(
                memory_pool_context_fn=self._maybe_get_memory_pool_context,
                load_format=load_format,
                custom_pipeline_name=custom_pipeline_name,
            )
        process_memory = get_process_gpu_memory(self.local_rank)
        if process_memory is not None:
            logger.info(
                "Worker %d: Process-scoped GPU memory after model loading: %.2f GiB.",
                self.rank,
                process_memory / GiB_bytes,
            )

        # When load_format is "dummy", pipeline will init with custom pipeline later
        if load_format != "dummy":
            assert self.model_runner.pipeline is not None

    def init_lora_manager(self) -> None:
        """Initialize the LoRA manager for this worker."""
        if self.model_runner.pipeline is None:
            return
        self.lora_manager = DiffusionLoRAManager(
            pipeline=self.model_runner.pipeline,
            device=self.device,
            dtype=self.od_config.dtype,
            max_cached_adapters=self.od_config.max_cpu_loras,
            lora_path=self.od_config.lora_path,
            lora_scale=self.od_config.lora_scale,
        )

    def generate(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        """Generate output for the given requests."""
        return self.execute_model(request, self.od_config)

    @classmethod
    def start_profile(cls, trace_path_template: str) -> str:
        """Start profiling for this GPU worker."""
        return CurrentProfiler.start(trace_path_template)

    @classmethod
    def stop_profile(cls) -> dict | None:
        """Stop profiling and return the result dictionary."""
        return CurrentProfiler.stop()

    def execute_model(self, req: OmniDiffusionRequest, od_config: OmniDiffusionConfig) -> DiffusionOutput:
        """Execute a forward pass by delegating to the model runner."""
        assert self.model_runner is not None, "Model runner not initialized"
        if self.lora_manager is not None:
            try:
                self.lora_manager.set_active_adapter(req.sampling_params.lora_request, req.sampling_params.lora_scale)
            except Exception as exc:
                if req.sampling_params.lora_request is not None:
                    raise
                logger.warning("LoRA activation skipped: %s", exc)
        return self.model_runner.execute_model(req)

    def load_weights(self, weights) -> set[str]:
        """Load weights by delegating to the model runner."""
        assert self.model_runner is not None, "Model runner not initialized"
        return self.model_runner.load_weights(weights)

    def remove_lora(self, adapter_id: int) -> bool:
        return self.lora_manager.remove_adapter(adapter_id)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        # NOTE (Alex): We have not implemented the API routing
        # for the frontend server yet.
        return self.lora_manager.add_adapter(lora_request)

    def list_loras(self) -> list[int]:
        return self.lora_manager.list_adapters()

    def pin_lora(self, adapter_id: int) -> bool:
        return self.lora_manager.pin_adapter(adapter_id)

    def sleep(self, level: int = 1) -> bool:
        """
        Put the worker to sleep, offloading model weights.

        Args:
            level: Sleep level. Level 1 offloads weights, level 2 also saves buffers.
        """
        from vllm.device_allocator.cumem import CuMemAllocator

        process_memory_before_sleep = get_process_gpu_memory(self.local_rank)
        free_bytes_before_sleep = None
        if process_memory_before_sleep is None:
            free_bytes_before_sleep = current_omni_platform.get_free_memory()

        # Save the buffers before level 2 sleep
        if level == 2 and self.model_runner is not None:
            model = self.model_runner.pipeline
            self._sleep_saved_buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
        process_memory_after_sleep = get_process_gpu_memory(self.local_rank)
        if process_memory_before_sleep is not None and process_memory_after_sleep is not None:
            freed_bytes = process_memory_before_sleep - process_memory_after_sleep
            used_bytes = process_memory_after_sleep
            accounting_scope = "process-scoped"
        else:
            free_bytes_after_sleep = current_omni_platform.get_free_memory()
            assert free_bytes_before_sleep is not None
            device_id = self.device.index if self.device.index is not None else 0
            total = current_omni_platform.get_device_total_memory(device_id)
            freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
            used_bytes = total - free_bytes_after_sleep
            accounting_scope = "device-scoped fallback"
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode (%s) freed %.2f GiB memory, %.2f GiB memory is still in use.",
            accounting_scope,
            freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes,
        )
        return True

    def wake_up(self, tags: list[str] | None = None) -> bool:
        """
        Wake up the worker from sleep mode. See the sleep function
        method for more details.

        Args:
            tags: An optional list of tags to reallocate the worker memory
                for specific memory allocations. Values must be in
                `("weights")`. If None, all memory is reallocated.
                wake_up should be called with all tags (or None) before the
                worker is used again.
        """
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers) and self.model_runner is not None:
            model = self.model_runner.pipeline
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}
        return True

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        """Get memory pool context for sleep mode support."""
        if self.od_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, "Sleep mode can only be used for one instance per process."
            return allocator.use_memory_pool(tag=tag)
        else:
            return nullcontext()

    def shutdown(self) -> None:
        """Shutdown the worker and cleanup distributed environment."""
        destroy_distributed_env()


class CustomPipelineWorkerExtension:
    def re_init_pipeline(self, custom_pipeline_args: dict[str, Any]) -> None:
        """
        Re-initialize the pipeline with custom arguments.

        Args:
            custom_pipeline_args: Dictionary of arguments for custom pipeline initialization
        """

        # Clean up old pipeline
        if self.model_runner.pipeline is not None:
            del self.model_runner.pipeline
            gc.collect()
            torch.cuda.empty_cache()

        # Get custom pipeline class name
        custom_pipeline_name = custom_pipeline_args["pipeline_class"]

        # Use the DiffusionWorker's load_model method which handles the forward context
        self.load_model(
            load_format="custom_pipeline",
            custom_pipeline_name=custom_pipeline_name,
        )
        self.init_lora_manager()


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        gpu_id: int,
        control_pipe: mp.connection.Connection,
        broadcast_handle,
        worker_extension_cls: str | None = None,
        custom_pipeline_args: dict[str, Any] | None = None,
    ):
        self.od_config = od_config
        self.gpu_id = gpu_id
        self.control_pipe = control_pipe

        # Inter-process Communication
        self.context = zmq.Context(io_threads=2)

        # Initialize MessageQueue reader from handle
        self.mq = MessageQueue.create_from_handle(broadcast_handle, gpu_id)

        self.result_mq = None
        self.result_mq_handle = None

        # Setup result sender (only for rank 0)
        if gpu_id == 0:
            self.result_mq = MessageQueue(n_reader=1, n_local_reader=1, local_reader_ranks=[0])
            self.result_mq_handle = self.result_mq.export_handle()
            logger.info(f"Worker {gpu_id} created result MessageQueue")

        assert od_config.master_port is not None

        # Create worker using WorkerWrapperBase for extension support
        self.worker = self._create_worker(gpu_id, od_config, worker_extension_cls, custom_pipeline_args)
        self._running = True
        self._current_req: OmniDiffusionRequest | None = None
        self._scheduling_policy: DiffusionSchedulingPolicy = TargetFreeGlobalReorderPolicy()

    def _create_worker(
        self,
        gpu_id: int,
        od_config: OmniDiffusionConfig,
        worker_extension_cls: str | None,
        custom_pipeline_args: dict[str, Any] | None = None,
    ) -> DiffusionWorker:
        """Create a worker instance. Override in subclasses for different worker types."""
        wrapper = WorkerWrapperBase(
            gpu_id=gpu_id,
            od_config=od_config,
            worker_extension_cls=worker_extension_cls,
            custom_pipeline_args=custom_pipeline_args,
        )
        return wrapper

    def return_result(self, output: DiffusionOutput):
        """Reply to client, only on rank 0."""
        request_key = None
        result_type = type(output).__name__
        if isinstance(output, dict):
            request_key = output.get("request_key")
            result_type = output.get("type", result_type)
        elif isinstance(output, DiffusionOutput):
            request_key = output.request_key

        if self._should_route_result_via_control_pipe(output):
            prepared = self._prepare_result_for_scheduler(output)
            logger.info(
                "Worker %s sending result via control pipe: type=%s request_key=%s",
                self.gpu_id,
                result_type,
                request_key,
            )
            self.control_pipe.send(
                {
                    "status": "generation_result",
                    "rank": self.gpu_id,
                    "payload": prepared,
                }
            )
            logger.info(
                "Worker %s sent result via control pipe: type=%s request_key=%s",
                self.gpu_id,
                result_type,
                request_key,
            )
            return

        if self.result_mq is not None:
            output = self._prepare_result_for_scheduler(output)
            logger.info(
                "Worker %s sending result to scheduler: type=%s request_key=%s",
                self.gpu_id,
                result_type,
                request_key,
            )
            self.result_mq.enqueue(output)
            logger.info(
                "Worker %s sent result to scheduler: type=%s request_key=%s",
                self.gpu_id,
                result_type,
                request_key,
            )

    @staticmethod
    def _tensor_tree_to_cpu(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, list):
            return [WorkerProc._tensor_tree_to_cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(WorkerProc._tensor_tree_to_cpu(v) for v in value)
        if isinstance(value, dict):
            return {k: WorkerProc._tensor_tree_to_cpu(v) for k, v in value.items()}
        return value

    @classmethod
    def _prepare_diffusion_output_for_scheduler(cls, output: DiffusionOutput) -> DiffusionOutput:
        return DiffusionOutput(
            output=cls._tensor_tree_to_cpu(output.output),
            trajectory_timesteps=cls._tensor_tree_to_cpu(output.trajectory_timesteps),
            trajectory_latents=cls._tensor_tree_to_cpu(output.trajectory_latents),
            trajectory_decoded=cls._tensor_tree_to_cpu(output.trajectory_decoded),
            error=output.error,
            finished=output.finished,
            request_key=output.request_key,
            post_process_func=output.post_process_func,
        )

    @classmethod
    def _prepare_result_for_scheduler(cls, output: Any) -> Any:
        if isinstance(output, DiffusionOutput):
            return cls._prepare_diffusion_output_for_scheduler(output)
        if isinstance(output, dict) and isinstance(output.get("output"), DiffusionOutput):
            prepared = dict(output)
            prepared["output"] = cls._prepare_diffusion_output_for_scheduler(output["output"])
            return prepared
        return output

    @staticmethod
    def _is_generation_result(output: Any) -> bool:
        if isinstance(output, dict):
            return output.get("type") == "generation_result"
        return isinstance(output, DiffusionOutput) and output.request_key is not None

    def _should_route_result_via_control_pipe(self, output: Any) -> bool:
        return self.gpu_id == 0 and self._is_generation_result(output)

    def _report_request_error_to_parent(self, request_key: str, error: str) -> None:
        try:
            self.control_pipe.send(
                {
                    "status": "request_error",
                    "rank": self.gpu_id,
                    "request_key": request_key,
                    "error": error,
                }
            )
        except Exception as exc:
            logger.warning(
                "Worker %s failed to report request error for %s to parent: %s",
                self.gpu_id,
                request_key,
                exc,
            )

    def recv_message(self, timeout: float | None = None):
        """Receive messages from broadcast queue."""
        return self.mq.dequeue(timeout=timeout, indefinite=timeout is None)

    def _supports_step_preemption(self) -> bool:
        if getattr(self.od_config, "disable_diffusion_preemption", False):
            return False
        model_cls_name = self.od_config.model_class_name or ""
        # Only enable step preemption for pipelines that persist enough
        # per-request execution state to resume safely after interleaving.
        return model_cls_name.startswith("QwenImage") or model_cls_name in {
            "WanPipeline",
            "Wan22Pipeline",
        }

    def _get_preemption_min_free_memory_bytes(self) -> int:
        configured_gb = getattr(self.od_config, "diffusion_preemption_min_free_memory_gb", None)
        if configured_gb is None:
            # NPU HSDP/FSDP unshard can require a sizable HCCL workspace at
            # runtime, so a tiny default headroom is not enough to make
            # arrival-time preemption decisions safe.
            configured_gb = 4.0 if current_omni_platform.is_npu() else 0.0
        return max(0, int(float(configured_gb) * GiB_bytes))

    def _get_synced_min_free_memory_bytes(self) -> int | None:
        free_bytes = current_omni_platform.get_free_memory(self.worker.device)
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and self.od_config.num_gpus
            and self.od_config.num_gpus > 1
        ):
            return int(free_bytes)

        world_group = get_world_group()
        if current_omni_platform.is_npu():
            # HCCL may need a large device-side workspace even for tiny tensors.
            # Use the CPU/gloo world group for this diagnostic MIN reduction so
            # the low-memory check itself does not trigger an NPU OOM.
            free_tensor = torch.tensor([int(free_bytes)], dtype=torch.int64)
            torch.distributed.all_reduce(
                free_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=world_group.cpu_group,
            )
            return int(free_tensor.item())

        free_tensor = torch.tensor([int(free_bytes)], device=self.worker.device, dtype=torch.int64)
        torch.distributed.all_reduce(
            free_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=world_group.device_group,
        )
        return int(free_tensor.item())

    def _should_skip_preemption_due_to_memory(self) -> tuple[bool, int | None, int]:
        min_required_bytes = self._get_preemption_min_free_memory_bytes()
        if min_required_bytes <= 0:
            return False, None, min_required_bytes

        synced_free_bytes = self._get_synced_min_free_memory_bytes()
        if synced_free_bytes is None:
            return False, None, min_required_bytes
        return synced_free_bytes < min_required_bytes, synced_free_bytes, min_required_bytes

    def _enqueue_generation_req(self, req: OmniDiffusionRequest) -> None:
        req.get_or_assign_priority()
        req.preempt_enabled = self._supports_step_preemption()
        req.preempt_step_chunk_size = max(1, int(req.preempt_step_chunk_size))
        incoming_req_id = self._req_debug_id(req)
        if self._scheduling_policy.is_aborted(incoming_req_id):
            print(
                f"[DiffusionDrop][{self._now_str()}][rank={self.gpu_id}] req_id={incoming_req_id} reason=aborted",
                flush=True,
            )
            return
        print(
            f"[DiffusionArrive][{self._now_str()}][rank={self.gpu_id}] "
            f"req_id={incoming_req_id} preempt_enabled={req.preempt_enabled}",
            flush=True,
        )
        if self._current_req is not None and req.preempt_enabled:
            skip_preempt, synced_free_bytes, min_required_bytes = self._should_skip_preemption_due_to_memory()
            if skip_preempt:
                self._scheduling_policy.defer_request(req)
                if self.gpu_id == 0:
                    print(
                        f"[DiffusionPreemptSkip][{self._now_str()}][rank={self.gpu_id}] "
                        f"req_id={incoming_req_id} reason=low_memory "
                        f"free_gib={(synced_free_bytes or 0) / GiB_bytes:.2f} "
                        f"threshold_gib={min_required_bytes / GiB_bytes:.2f}",
                        flush=True,
                    )
                return
        decision = self._scheduling_policy.on_request_arrival(req, self._current_req)
        for dropped_id in self._scheduling_policy.consume_recent_dropped_request_ids():
            print(
                f"[DiffusionDrop][{self._now_str()}][rank={self.gpu_id}] req_id={dropped_id} reason=aborted",
                flush=True,
            )
        if decision.preemption is not None:
            preempted_req_id = self._req_debug_id(decision.preemption.preempted_req)
            current_step = self._req_step_index(decision.preemption.preempted_req)
            candidate_req_id = self._req_debug_id(decision.preemption.selected_req)
            print(
                f"[DiffusionPreempt][{self._now_str()}][rank={self.gpu_id}] "
                f"preempt req_id={preempted_req_id} at step={current_step}; "
                f"switch_to req_id={candidate_req_id} "
                f"cost_cur={decision.preemption.preempted_cost:.0f} "
                f"cost_new={decision.preemption.selected_cost:.0f}",
                flush=True,
            )
        self._current_req = decision.current_req

    def _schedule_next_after_finish(self) -> OmniDiffusionRequest | None:
        # Global rescheduling decision point #2: current request finished.
        next_req = self._scheduling_policy.on_request_finish()
        for dropped_id in self._scheduling_policy.consume_recent_dropped_request_ids():
            print(
                f"[DiffusionDrop][{self._now_str()}][rank={self.gpu_id}] req_id={dropped_id} reason=aborted",
                flush=True,
            )
        if next_req is None:
            return None
        next_req_id = self._req_debug_id(next_req)
        next_step = self._req_step_index(next_req)
        print(
            f"[DiffusionResume][{self._now_str()}][rank={self.gpu_id}] "
            f"resume req_id={next_req_id} from step={next_step}",
            flush=True,
        )
        return next_req

    @staticmethod
    def _req_debug_id(req: OmniDiffusionRequest) -> str:
        return req_debug_id(req)

    @staticmethod
    def _req_step_index(req: OmniDiffusionRequest) -> int:
        return req_step_index(req)

    @staticmethod
    def _now_str() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _abort_requests(self, request_ids: list[str]) -> None:
        if not request_ids:
            return
        self._scheduling_policy.abort_request_ids(request_ids)
        for rid in request_ids:
            if isinstance(rid, str):
                print(f"[DiffusionAbort][{self._now_str()}][rank={self.gpu_id}] req_id={rid}", flush=True)

        if self._current_req is not None:
            cur_id = self._req_debug_id(self._current_req)
            if self._scheduling_policy.is_aborted(cur_id):
                aborted_req = self._current_req
                self._current_req = self._schedule_next_after_finish()
                self.return_result(
                    {
                        "type": "generation_result",
                        "request_key": aborted_req.request_key,
                        "output": DiffusionOutput(
                            error=f"request {cur_id} aborted",
                            finished=True,
                            request_key=aborted_req.request_key,
                        ),
                    }
                )

    def _drain_incoming_messages(self) -> None:
        while self._running:
            try:
                msg = self.recv_message(timeout=0.0)
            except TimeoutError:
                return
            except Exception as e:
                logger.error("Error receiving message in worker loop: %s", e, exc_info=True)
                return

            if msg is None or (hasattr(msg, "__len__") and len(msg) == 0):
                logger.warning("Worker %s: Received empty payload, ignoring", self.gpu_id)
                continue

            if isinstance(msg, dict) and msg.get("type") == "rpc":
                try:
                    result, should_reply = self.execute_rpc(msg)
                    if should_reply:
                        self.return_result(result)
                except Exception as e:
                    logger.error("Error processing RPC: %s", e, exc_info=True)
                    if self.result_mq is not None:
                        self.return_result(DiffusionOutput(error=str(e)))
                continue

            if isinstance(msg, dict) and msg.get("type") == "shutdown":
                logger.info("Worker %s: Received shutdown message", self.gpu_id)
                self._running = False
                return

            if isinstance(msg, dict) and msg.get("type") == "abort":
                self._abort_requests(msg.get("request_ids", []))
                continue

            if isinstance(msg, OmniDiffusionRequest):
                self._enqueue_generation_req(msg)
            else:
                logger.warning("Worker %s: Ignoring unsupported message type %s", self.gpu_id, type(msg))

    def execute_rpc(self, rpc_request: dict) -> tuple[object | None, bool]:
        """Execute an RPC request and indicate whether to reply."""
        method = rpc_request["method"]
        args = rpc_request.get("args", ())
        kwargs = rpc_request.get("kwargs", {})
        output_rank = rpc_request.get("output_rank")
        exec_all_ranks = rpc_request.get("exec_all_ranks", False)

        should_execute = exec_all_ranks or output_rank is None or output_rank == self.gpu_id
        should_reply = (output_rank is None or output_rank == self.gpu_id) and self.result_mq is not None

        if not should_execute:
            return None, False

        try:
            # Use execute_method from WorkerWrapperBase for consistent method resolution
            result = self.worker.execute_method(method, *args, **kwargs)
            return result, should_reply
        except Exception as e:
            logger.error(f"Error executing RPC: {e}", exc_info=True)
            raise e

    def worker_busy_loop(self) -> None:
        """Main busy loop for Multiprocessing Workers."""
        logger.info(f"Worker {self.gpu_id} ready to receive requests via shared memory")

        while self._running:
            self._drain_incoming_messages()
            if not self._running:
                break

            if self._current_req is None:
                try:
                    msg = self.recv_message(timeout=None)
                except Exception as e:
                    logger.error("Error receiving message in worker loop: %s", e, exc_info=True)
                    continue
                if isinstance(msg, OmniDiffusionRequest):
                    self._enqueue_generation_req(msg)
                elif isinstance(msg, dict) and msg.get("type") == "rpc":
                    try:
                        result, should_reply = self.execute_rpc(msg)
                        if should_reply:
                            self.return_result(result)
                    except Exception as e:
                        logger.error("Error processing RPC: %s", e, exc_info=True)
                        if self.result_mq is not None:
                            self.return_result(DiffusionOutput(error=str(e)))
                    continue
                elif isinstance(msg, dict) and msg.get("type") == "shutdown":
                    logger.info("Worker %s: Received shutdown message", self.gpu_id)
                    self._running = False
                    break
                elif isinstance(msg, dict) and msg.get("type") == "abort":
                    self._abort_requests(msg.get("request_ids", []))
                    continue
                else:
                    logger.warning("Worker %s: Ignoring unsupported message type %s", self.gpu_id, type(msg))
                    continue

            if self._current_req is None:
                continue

            current_req_id = self._req_debug_id(self._current_req)
            if self._scheduling_policy.is_aborted(current_req_id):
                self.return_result(
                    {
                        "type": "generation_result",
                        "request_key": self._current_req.request_key,
                        "output": DiffusionOutput(
                            error=f"request {current_req_id} aborted",
                            finished=True,
                            request_key=self._current_req.request_key,
                        ),
                    }
                )
                self._current_req = self._schedule_next_after_finish()
                continue

            try:
                output = self.worker.execute_model(self._current_req, self.od_config)
            except Exception as e:
                failed_req_id = self._req_debug_id(self._current_req)
                failed_step = self._req_step_index(self._current_req)
                print(
                    f"[DiffusionError][{self._now_str()}][rank={self.gpu_id}] "
                    f"req_id={failed_req_id} step={failed_step} error={e}",
                    flush=True,
                )
                logger.error(
                    "Error executing forward in event loop: %s",
                    e,
                    exc_info=True,
                )
                output = DiffusionOutput(error=str(e), finished=True, request_key=self._current_req.request_key)
                self._report_request_error_to_parent(self._current_req.request_key, str(e))

            if output.finished:
                finished_req_id = self._req_debug_id(self._current_req)
                finished_step = self._req_step_index(self._current_req)
                status = "error" if output.error else "ok"
                print(
                    f"[DiffusionFinish][{self._now_str()}][rank={self.gpu_id}] "
                    f"req_id={finished_req_id} status={status} step={finished_step}",
                    flush=True,
                )
                try:
                    self.return_result(
                        {
                            "type": "generation_result",
                            "request_key": self._current_req.request_key,
                            "output": output,
                        }
                    )
                except zmq.ZMQError as e:
                    logger.error("ZMQ error sending reply: %s", e)
                self._current_req = self._schedule_next_after_finish()
                continue

        logger.info("event loop terminated.")
        try:
            self.worker.shutdown()
        except Exception as exc:
            logger.warning("Worker %s: Shutdown encountered an error: %s", self.gpu_id, exc)
        self.context.term()

    @staticmethod
    def worker_main(
        rank: int,
        od_config: OmniDiffusionConfig,
        pipe_writer: mp.connection.Connection,
        broadcast_handle,
        worker_extension_cls: str | None = None,
        custom_pipeline_args: dict[str, Any] | None = None,
    ) -> None:
        """Worker initialization and execution loops."""
        from vllm_omni.plugins import load_omni_general_plugins

        load_omni_general_plugins()
        worker_proc = WorkerProc(
            od_config,
            gpu_id=rank,
            control_pipe=pipe_writer,
            broadcast_handle=broadcast_handle,
            worker_extension_cls=worker_extension_cls,
            custom_pipeline_args=custom_pipeline_args,
        )
        logger.info(f"Worker {rank}: Scheduler loop started.")
        pipe_writer.send(
            {
                "status": "ready",
                "result_handle": worker_proc.result_mq_handle if rank == 0 else None,
            }
        )
        worker_proc.worker_busy_loop()
        logger.info(f"Worker {rank}: Shutdown complete.")


class WorkerWrapperBase:
    """
    Wrapper base class that creates DiffusionWorker with optional worker_extension_cls support.
    This enables dynamic inheritance for DiffusionWorker to extend with custom functionality.
    """

    def __init__(
        self,
        gpu_id: int,
        od_config: OmniDiffusionConfig,
        base_worker_class: type = DiffusionWorker,
        worker_extension_cls: str | None = None,
        custom_pipeline_args: dict[str, Any] | None = None,
    ):
        """
        Initialize WorkerWrapperBase with support for worker extensions.

        Args:
            gpu_id: GPU device ID
            od_config: OmniDiffusionConfig configuration
            worker_extension_cls: Optional qualified name of worker extension class
            custom_pipeline_args: Optional arguments for custom pipeline initialization
        """
        self.gpu_id = gpu_id
        self.od_config = od_config
        self.base_worker_class = base_worker_class
        self.worker_extension_cls = worker_extension_cls
        self.custom_pipeline_args = custom_pipeline_args

        # Prepare worker class with extension support
        worker_class = self._prepare_worker_class()

        # Create the actual worker instance
        self.worker = worker_class(
            local_rank=gpu_id,
            rank=gpu_id,
            od_config=od_config,
        )

        # Re-initialize pipeline with custom pipeline if provided
        if self.custom_pipeline_args is not None:
            self.worker.re_init_pipeline(self.custom_pipeline_args)

    def _prepare_worker_class(self) -> type:
        """
        Prepare the worker class with optional extension.
        Dynamically extends GPUWorker with worker_extension_cls if provided.

        Returns:
            The worker class (potentially extended)
        """
        worker_class = self.base_worker_class

        # If custom_pipeline_args is provided, use CustomPipelineWorkerExtension
        if self.custom_pipeline_args is not None:
            # Set worker_extension_cls to CustomPipelineWorkerExtension if not already set
            if self.worker_extension_cls is None:
                self.worker_extension_cls = CustomPipelineWorkerExtension

        if self.worker_extension_cls:
            if isinstance(self.worker_extension_cls, str):
                worker_extension_cls = resolve_obj_by_qualname(self.worker_extension_cls)
            else:
                worker_extension_cls = self.worker_extension_cls
            extended_calls = []

            if worker_extension_cls not in worker_class.__bases__:
                # Check for conflicts between worker and extension
                for attr in dir(worker_extension_cls):
                    if attr.startswith("__"):
                        continue
                    if hasattr(worker_class, attr):
                        logger.warning(
                            f"Worker class {worker_class} already has attribute "
                            f"{attr}, which may conflict with worker extension "
                            f"class {worker_extension_cls}."
                        )
                    if callable(getattr(worker_extension_cls, attr)):
                        extended_calls.append(attr)

                # Dynamically inherit the worker extension class
                class_name = f"{worker_class.__name__}With{worker_extension_cls.__name__}"
                worker_class = type(class_name, (worker_extension_cls, worker_class), {})
                logger.info(
                    "Created extended worker class %s from %s for extended calls %s",
                    class_name,
                    worker_extension_cls,
                    extended_calls,
                )

        return worker_class

    def generate(self, requests: list[OmniDiffusionRequest]) -> DiffusionOutput:
        """
        Generate output for the given requests.

        Args:
            requests: List of diffusion requests

        Returns:
            DiffusionOutput with generated results
        """
        return self.worker.generate(requests)

    def execute_model(self, reqs: list[OmniDiffusionRequest], od_config: OmniDiffusionConfig) -> DiffusionOutput:
        """
        Execute a forward pass.

        Args:
            reqs: List of diffusion requests
            od_config: OmniDiffusionConfig configuration

        Returns:
            DiffusionOutput with generated results
        """
        return self.worker.execute_model(reqs, od_config)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load model weights.

        Args:
            weights: Iterable of (name, tensor) tuples

        Returns:
            Set of loaded weight names
        """
        return self.worker.load_weights(weights)

    def sleep(self, level: int = 1) -> bool:
        """
        Put the worker to sleep. The worker should not process any requests.
        The caller should guarantee that no requests are being processed
        during the sleep period, before `wake_up` is called.

        Args:
            level: The sleep level. Level 1 sleep will offload the model
                weights and discard the kv cache.
                Currently only support level 1.

        Returns:
            True on success
        """
        return self.worker.sleep(level)

    def wake_up(self, tags: list[str] | None = None) -> bool:
        """
        Wake up the worker from sleep mode. See the sleep function
        method for more details.

        Args:
            tags: An optional list of tags to reallocate the worker memory
                for specific memory allocations. Values must be in
                `("weights")`. If None, all memory is reallocated.
                wake_up should be called with all tags (or None) before the
                worker is used again.

        Returns:
            True on success
        """
        return self.worker.wake_up(tags)

    def shutdown(self) -> None:
        """Shutdown the worker and cleanup resources."""
        return self.worker.shutdown()

    def execute_method(self, method: str | bytes, *args, **kwargs) -> Any:
        """
        Execute a method on the worker.

        Args:
            method: Method name (str) or serialized callable (bytes)

        Returns:
            Result of the method execution (type depends on the method)

        Raises:
            Exception: If method execution fails
        """
        try:
            # Method resolution order:
            # 1. If method is defined in this class, it will be called directly
            # 2. Otherwise, since we define `__getattr__` and redirect attribute
            #    query to `self.worker`, the method will be called on the worker
            assert isinstance(method, str), "Method must be str"
            func = getattr(self.worker, method)
            return func(*args, **kwargs)

        except Exception as e:
            msg = f"Error executing method {method!r}. This might cause issues in distributed execution."
            logger.exception(msg)
            raise e

    def __getattr__(self, attr: str):
        """Delegate attribute access to the wrapped worker."""
        return getattr(self.worker, attr)
