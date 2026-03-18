import multiprocessing as mp
import threading
import time
import weakref
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.data import SHUTDOWN_MESSAGE, DiffusionOutput
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.scheduler import Scheduler
from vllm_omni.diffusion.worker import WorkerProc

logger = init_logger(__name__)


@dataclass
class BackgroundResources:
    """
    Used as a finalizer for clean shutdown.
    """

    scheduler: Scheduler | None = None
    processes: list[mp.Process] | None = None
    scheduler_pipes: list[mp.connection.Connection] | None = None

    def __call__(self):
        """Clean up background resources."""
        if self.scheduler is not None:
            try:
                for _ in range(self.scheduler.num_workers):
                    self.scheduler.mq.enqueue(SHUTDOWN_MESSAGE)
                self.scheduler.close()
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
        self._scheduler_pipe_readers: list[mp.connection.Connection] = []
        self._pipe_reader_stop = False
        self._pipe_reader_thread: threading.Thread | None = None

        # Initialize scheduler
        self.scheduler = Scheduler()
        self.scheduler.initialize(self.od_config)
        broadcast_handle = self.scheduler.get_broadcast_handle()

        # Launch workers
        processes, result_handle, scheduler_pipe_readers = self._launch_workers(broadcast_handle)

        if result_handle is not None:
            self.scheduler.initialize_result_queue(result_handle)
        else:
            logger.error("Failed to get result queue handle from workers")

        self._processes = processes
        self._scheduler_pipe_readers = scheduler_pipe_readers
        self._pipe_reader_thread = threading.Thread(
            target=self._worker_pipe_loop,
            name="diffusion-worker-pipe-reader",
            daemon=True,
        )
        self._pipe_reader_thread.start()

        self.resources = BackgroundResources(
            scheduler=self.scheduler,
            processes=self._processes,
            scheduler_pipes=self._scheduler_pipe_readers,
        )
        self._finalizer = weakref.finalize(self, self.resources)

    def _launch_workers(self, broadcast_handle):
        od_config = self.od_config
        logger.info("Starting server...")

        num_gpus = od_config.num_gpus
        mp.set_start_method("spawn", force=True)
        processes = []

        # Extract worker_extension_cls and custom_pipeline_args from od_config
        worker_extension_cls = od_config.worker_extension_cls
        custom_pipeline_args = getattr(od_config, "custom_pipeline_args", None)

        # Launch all worker processes
        scheduler_pipe_readers = []
        scheduler_pipe_writers = []

        for i in range(num_gpus):
            reader, writer = mp.Pipe(duplex=False)
            scheduler_pipe_writers.append(writer)
            process = mp.Process(
                target=WorkerProc.worker_main,
                args=(
                    i,  # rank
                    od_config,
                    writer,
                    broadcast_handle,
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
        while not self._pipe_reader_stop:
            if not self._scheduler_pipe_readers:
                time.sleep(0.1)
                continue
            ready_readers = mp.connection.wait(self._scheduler_pipe_readers, timeout=0.1)
            for reader in ready_readers:
                try:
                    message = reader.recv()
                except EOFError:
                    continue
                except Exception as exc:
                    logger.warning("Failed to read worker control message: %s", exc)
                    continue

                if not isinstance(message, dict):
                    logger.warning("Ignoring unexpected worker control message type %s", type(message))
                    continue

                status = message.get("status")
                if status == "request_error":
                    request_key = message.get("request_key")
                    error = message.get("error")
                    rank = message.get("rank")
                    if not isinstance(request_key, str) or not isinstance(error, str):
                        logger.warning("Ignoring malformed worker request_error message: %s", message)
                        continue

                    logger.error(
                        "Received worker request failure from rank %s for %s: %s",
                        rank,
                        request_key,
                        error,
                    )
                    self.scheduler.publish_result(
                        request_key,
                        DiffusionOutput(error=error, finished=True, request_key=request_key),
                        source=f"worker-rank-{rank}",
                    )
                    self.abort(request_key)
                    continue

                if status == "generation_result":
                    rank = message.get("rank")
                    payload = message.get("payload")
                    if isinstance(payload, dict) and payload.get("type") == "generation_result":
                        request_key = payload.get("request_key")
                        output = payload.get("output")
                        if isinstance(request_key, str) and isinstance(output, DiffusionOutput):
                            logger.info(
                                "Received worker generation result from rank %s for %s",
                                rank,
                                request_key,
                            )
                            self.scheduler.publish_result(
                                request_key,
                                output,
                                source=f"worker-rank-{rank}",
                            )
                            continue
                    if isinstance(payload, DiffusionOutput) and isinstance(payload.request_key, str):
                        logger.info(
                            "Received worker legacy generation result from rank %s for %s",
                            rank,
                            payload.request_key,
                        )
                        self.scheduler.publish_result(
                            payload.request_key,
                            payload,
                            source=f"worker-rank-{rank}",
                        )
                        continue
                    logger.warning("Ignoring malformed worker generation_result message: %s", message)
                    continue

    def add_req(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        return self.scheduler.add_req(request)

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        if self._closed:
            raise RuntimeError("DiffusionExecutor is closed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # Prepare RPC request message
        rpc_request = {
            "type": "rpc",
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "output_rank": unique_reply_rank,
        }

        try:
            # Acquire lock with timeout awareness so that a stalled add_req
            # (holding the lock while blocked on dequeue) does not prevent
            # this RPC from honouring its own timeout.
            lock_timeout = None if deadline is None else max(0, deadline - time.monotonic())
            acquired = self.scheduler._lock.acquire(timeout=lock_timeout if lock_timeout is not None else -1)
            if not acquired:
                raise TimeoutError(f"RPC call to {method} timed out waiting for scheduler lock.")
            try:
                # Broadcast RPC request to all workers via unified message queue
                self.scheduler.mq.enqueue(rpc_request)

                # Determine which workers we expect responses from
                num_responses = 1 if unique_reply_rank is not None else self.od_config.num_gpus

                responses = []
                for _ in range(num_responses):
                    dequeue_timeout = None if deadline is None else max(0, deadline - time.monotonic())
                    try:
                        if self.scheduler.result_mq is None:
                            raise RuntimeError("Result queue not initialized")

                        response = self.scheduler.result_mq.dequeue(timeout=dequeue_timeout)

                        # Check if response indicates an error
                        if isinstance(response, dict) and response.get("status") == "error":
                            raise RuntimeError(
                                f"Worker failed with error '{response.get('error')}', "
                                "please check the stack trace above for the root cause"
                            )

                        responses.append(response)
                    except TimeoutError as e:
                        raise TimeoutError(f"RPC call to {method} timed out.") from e

                return responses[0] if unique_reply_rank is not None else responses
            finally:
                self.scheduler._lock.release()

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
        self._pipe_reader_stop = True
        if self._pipe_reader_thread is not None:
            self._pipe_reader_thread.join(timeout=1.0)
            self._pipe_reader_thread = None
        self._finalizer()

    def abort(self, request_id: str | Iterable[str]) -> None:
        """Broadcast abort request IDs to all diffusion workers."""
        if self._closed:
            return
        if isinstance(request_id, str):
            request_ids = [request_id]
        else:
            request_ids = [rid for rid in request_id if isinstance(rid, str)]
        if not request_ids:
            return
        self.scheduler.mq.enqueue(
            {
                "type": "abort",
                "request_ids": request_ids,
            }
        )
