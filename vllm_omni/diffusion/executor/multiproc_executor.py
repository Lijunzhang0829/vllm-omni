import multiprocessing as mp
import multiprocessing.connection
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any

import zmq
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger

from vllm_omni.diffusion.data import SHUTDOWN_MESSAGE, DiffusionOutput
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.ipc import unpack_diffusion_output_shm
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import SchedulerInterface
from vllm_omni.trace_logging import write_trace_event
from vllm_omni.diffusion.worker import WorkerProc

logger = init_logger(__name__)


@dataclass
class BackgroundResources:
    """
    Used as a finalizer for clean shutdown.
    """

    broadcast_mq: MessageQueue | None = None
    result_mq: MessageQueue | None = None
    num_workers: int = 0
    processes: list[mp.Process] | None = None
    scheduler_pipes: list[mp.connection.Connection] | None = None

    def __call__(self):
        """Clean up background resources."""
        if self.broadcast_mq is not None:
            try:
                for _ in range(self.num_workers):
                    self.broadcast_mq.enqueue(SHUTDOWN_MESSAGE)

                self.broadcast_mq = None
                self.result_mq = None
            except Exception as exc:
                logger.warning("Failed to send shutdown signal: %s", exc)

        if self.processes:
            for proc in self.processes:
                if not proc.is_alive():
                    continue
                proc.join(30)
                if proc.is_alive():
                    logger.warning("Terminating diffusion worker %s after timeout", proc.name)
                    proc.terminate()
                    proc.join(30)
        if self.scheduler_pipes:
            for pipe in self.scheduler_pipes:
                try:
                    pipe.close()
                except Exception:
                    pass


class MultiprocDiffusionExecutor(DiffusionExecutor):
    uses_multiproc: bool = True

    def _init_executor(self) -> None:
        self._processes: list[mp.Process] = []
        self._closed = False
        self._mp_ctx = mp.get_context("spawn")
        self._scheduler_pipe_readers: list[mp.connection.Connection] = []
        self._pipe_reader_stop = threading.Event()
        self._pipe_reader_thread: threading.Thread | None = None
        self._result_scheduler: SchedulerInterface | None = None
        self._preempt_event = self._mp_ctx.Event()
        self._active_completed_steps = self._mp_ctx.Value("i", 0, lock=False)

        num_workers = self.od_config.num_gpus
        self._broadcast_mq = self._init_broadcast_queue(num_workers)
        broadcast_handle = self._broadcast_mq.export_handle()

        # Launch workers
        processes, result_handle, scheduler_pipe_readers = self._launch_workers(broadcast_handle)
        self._result_mq = self._init_result_queue(result_handle)
        self._processes = processes
        self._scheduler_pipe_readers = scheduler_pipe_readers
        self._pipe_reader_thread = threading.Thread(
            target=self._worker_pipe_loop,
            name="diffusion-worker-pipe-reader",
            daemon=True,
        )
        self._pipe_reader_thread.start()

        self.resources = BackgroundResources(
            broadcast_mq=self._broadcast_mq,
            result_mq=self._result_mq,
            num_workers=num_workers,
            processes=self._processes,
            scheduler_pipes=self._scheduler_pipe_readers,
        )
        self._finalizer = weakref.finalize(self, self.resources)

    def set_result_scheduler(self, scheduler: SchedulerInterface) -> None:
        self._result_scheduler = scheduler

    def _init_broadcast_queue(self, num_workers: int) -> MessageQueue:
        return MessageQueue(
            n_reader=num_workers,
            n_local_reader=num_workers,
            local_reader_ranks=list(range(num_workers)),
        )

    def _init_result_queue(self, result_handle) -> MessageQueue | None:
        if result_handle is None:
            logger.error("Failed to get result queue handle from workers")
            return None
        return MessageQueue.create_from_handle(result_handle, 0)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("DiffusionExecutor is closed.")
        if self._result_mq is None:
            raise RuntimeError("Result queue not initialized")

    def _launch_workers(self, broadcast_handle):
        od_config = self.od_config
        logger.info("Starting server...")

        num_gpus = od_config.num_gpus
        processes = []

        # Extract worker_extension_cls and custom_pipeline_args from od_config
        worker_extension_cls = od_config.worker_extension_cls
        custom_pipeline_args = getattr(od_config, "custom_pipeline_args", None)

        # Launch all worker processes
        scheduler_pipe_readers = []
        scheduler_pipe_writers = []

        for i in range(num_gpus):
            reader, writer = self._mp_ctx.Pipe(duplex=False)
            scheduler_pipe_writers.append(writer)
            process = self._mp_ctx.Process(
                target=WorkerProc.worker_main,
                args=(
                    i,  # rank
                    od_config,
                    writer,
                    broadcast_handle,
                    self._preempt_event,
                    self._active_completed_steps,
                    worker_extension_cls,
                    custom_pipeline_args,
                ),
                name=f"DiffusionWorker-{i}",
                daemon=True,
            )
            scheduler_pipe_readers.append(reader)
            process.start()
            processes.append(process)

        # Wait for all workers to be ready
        scheduler_infos = []
        result_handle = None
        for writer in scheduler_pipe_writers:
            writer.close()

        for i, reader in enumerate(scheduler_pipe_readers):
            try:
                data = reader.recv()
            except EOFError:
                logger.error(f"Rank {i} scheduler is dead. Please check if there are relevant logs.")
                processes[i].join()
                logger.error(f"Exit code: {processes[i].exitcode}")
                raise

            if data["status"] != "ready":
                raise RuntimeError("Initialization failed. Please see the error messages above.")

            if i == 0:
                result_handle = data.get("result_handle")

            scheduler_infos.append(data)

        logger.debug("All workers are ready")

        return processes, result_handle, scheduler_pipe_readers

    def _worker_pipe_loop(self) -> None:
        while not self._pipe_reader_stop.is_set():
            readers = list(self._scheduler_pipe_readers)
            if not readers:
                self._pipe_reader_stop.wait(0.1)
                continue

            try:
                ready_readers = multiprocessing.connection.wait(readers, timeout=0.1)
            except Exception as exc:
                logger.warning("Worker control pipe wait failed: %s", exc)
                if self._result_scheduler is not None:
                    self._result_scheduler.set_reader_error(RuntimeError(f"Worker control pipe wait failed: {exc}"))
                return

            for reader in ready_readers:
                try:
                    message = reader.recv()
                except EOFError:
                    try:
                        reader.close()
                    finally:
                        if reader in self._scheduler_pipe_readers:
                            self._scheduler_pipe_readers.remove(reader)
                    continue
                except Exception as exc:
                    logger.warning("Failed to read worker control message: %s", exc)
                    if self._result_scheduler is not None:
                        self._result_scheduler.set_reader_error(RuntimeError(f"Worker control pipe failed: {exc}"))
                    continue

                if not isinstance(message, dict):
                    logger.warning("Ignoring unexpected worker control message type %s", type(message))
                    continue

                if message.get("status") != "generation_result":
                    logger.warning("Ignoring unexpected worker control message: %s", message)
                    continue

                payload = message.get("payload")
                if not isinstance(payload, DiffusionOutput) or not isinstance(payload.request_key, str):
                    logger.warning("Ignoring malformed worker generation_result message: %s", message)
                    continue

                if self._result_scheduler is None:
                    logger.warning("Dropping generation_result for request_key=%s before scheduler binding", payload.request_key)
                    continue
                self._result_scheduler.publish_result(payload.request_key, payload, source="worker_pipe")

    def add_req(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        self._ensure_open()
        request_key = request.request_ids[0] if request.request_ids else None
        trace_request_id = request.trace_request_id or request_key
        rpc_request = {
            "type": "rpc",
            "method": "generate",
            "args": (request,),
            "kwargs": {},
            "output_rank": 0,
            "exec_all_ranks": True,
        }

        if request_key is None:
            logger.warning("Generation request missing request_ids; falling back to result_mq path.")
            return self._add_req_via_result_queue(rpc_request)
        if self._result_scheduler is None:
            raise RuntimeError("Result scheduler is not bound for control-pipe generation path.")

        # Generation results intentionally bypass result_mq/shm_broadcast and
        # return over the worker control pipe. This restores the lighter
        # request->mq, result->pipe split used in v0.16 and keeps large
        # diffusion outputs off the shared-memory ring buffer critical path.

        try:
            write_trace_event(
                os.environ.get("VLLM_OMNI_TRACE_LOG_FILE"),
                "executor_add_req_enter",
                node=os.environ.get("VLLM_OMNI_TRACE_NODE"),
                request_id=trace_request_id,
                server_request_id=request_key,
            )
            self._broadcast_mq.enqueue(rpc_request)
            write_trace_event(
                os.environ.get("VLLM_OMNI_TRACE_LOG_FILE"),
                "executor_add_req_enqueued",
                node=os.environ.get("VLLM_OMNI_TRACE_NODE"),
                request_id=trace_request_id,
                server_request_id=request_key,
            )
            output = self._result_scheduler.wait_for_result(request_key)
            write_trace_event(
                os.environ.get("VLLM_OMNI_TRACE_LOG_FILE"),
                "executor_add_req_exit",
                node=os.environ.get("VLLM_OMNI_TRACE_NODE"),
                request_id=trace_request_id,
                server_request_id=request_key,
            )
            return output
        except Exception as e:
            logger.error(f"Generate call failed: {e}")
            raise

    def _add_req_via_result_queue(self, rpc_request: dict[str, Any]) -> DiffusionOutput:
        self._broadcast_mq.enqueue(rpc_request)
        response = self._result_mq.dequeue()

        try:
            unpack_diffusion_output_shm(response)
        except Exception as e:
            logger.warning("SHM unpack failed (data may already be inline): %s", e)

        if isinstance(response, dict) and response.get("status") == "error":
            raise RuntimeError(
                f"Worker failed with error '{response.get('error')}', "
                "please check the stack trace above for the root cause"
            )
        if not isinstance(response, DiffusionOutput):
            raise RuntimeError(f"Unexpected response type for generate: {type(response)!r}")
        return response

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        self._ensure_open()

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # Prepare RPC request message
        # When unique_reply_rank is None, all workers must execute the RPC
        # but only rank 0 can reply (it's the only one with a result_mq).
        rpc_request = {
            "type": "rpc",
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "output_rank": unique_reply_rank if unique_reply_rank is not None else 0,
            "exec_all_ranks": unique_reply_rank is None,
        }

        try:
            # Broadcast RPC request to all workers via unified message queue
            self._broadcast_mq.enqueue(rpc_request)

            # Only rank 0 has a result_mq, so we always expect exactly 1 response
            num_responses = 1

            responses = []
            for _ in range(num_responses):
                dequeue_timeout = None if deadline is None else max(0, deadline - time.monotonic())
                try:
                    response = self._result_mq.dequeue(timeout=dequeue_timeout)

                    # Check if response indicates an error
                    if isinstance(response, dict) and response.get("status") == "error":
                        raise RuntimeError(
                            f"Worker failed with error '{response.get('error')}', "
                            "please check the stack trace above for the root cause"
                        )

                    responses.append(response)
                except zmq.error.Again as e:
                    raise TimeoutError(f"RPC call to {method} timed out.") from e
                except TimeoutError as e:
                    raise TimeoutError(f"RPC call to {method} timed out.") from e

            return responses[0] if unique_reply_rank is not None else responses
        except Exception as e:
            logger.error(f"RPC call failed: {e}")
            raise

    def check_health(self) -> None:
        # Simple check if processes are alive
        for p in self._processes:
            if not p.is_alive():
                raise RuntimeError(f"Worker process {p.name} is dead")

    def shutdown(self) -> None:
        self._closed = True
        try:
            self._pipe_reader_stop.set()
            if self._pipe_reader_thread is not None:
                self._pipe_reader_thread.join(timeout=5)
            self._finalizer()
        finally:
            self._broadcast_mq = None
            self._result_mq = None
            self.resources = None
            self._processes = []
            self._scheduler_pipe_readers = []
            self._pipe_reader_thread = None
