import threading
from types import SimpleNamespace
from unittest.mock import Mock

from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker


def _make_worker():
    worker = object.__new__(DiffusionWorker)
    worker.model_runner = Mock()
    worker.model_runner.pipeline = SimpleNamespace(_interrupt=False, _interrupt_event=None)
    worker.lora_manager = None
    worker._resident_scheduler_states = {}
    worker._preempt_event = threading.Event()
    worker.od_config = SimpleNamespace()
    return worker


def _make_request(request_id: str, *, preemption_enabled: bool = True):
    return SimpleNamespace(
        request_ids=[request_id],
        sampling_params=SimpleNamespace(
            extra_args={"_server_preemption_enabled": preemption_enabled},
            lora_request=None,
            lora_scale=1.0,
        ),
    )


def test_execute_model_caches_partial_scheduler_state_on_worker():
    worker = _make_worker()
    request = _make_request("req-1")
    resident_state = {"latents": object(), "completed_steps": 5}
    worker.model_runner.execute_model.return_value = SimpleNamespace(finished=False, scheduler_state=resident_state)

    output = worker.execute_model(request, worker.od_config)

    assert worker._resident_scheduler_states["req-1"] is resident_state
    assert output.scheduler_state == {"request_key": "req-1", "completed_steps": 5}
    assert request.sampling_params.extra_args.get("_server_state") is None


def test_execute_model_restores_cached_scheduler_state_from_worker():
    worker = _make_worker()
    request = _make_request("req-2")
    resident_state = {"latents": object(), "completed_steps": 3}
    worker._resident_scheduler_states["req-2"] = resident_state
    worker.model_runner.execute_model.return_value = SimpleNamespace(finished=True, scheduler_state=None)

    worker.execute_model(request, worker.od_config)

    assert request.sampling_params.extra_args["_server_state"] is resident_state
    assert "req-2" not in worker._resident_scheduler_states


def test_execute_model_attaches_shared_preempt_event_to_pipeline():
    worker = _make_worker()
    request = _make_request("req-3")

    def _execute(req):
        assert worker.model_runner.pipeline._interrupt_event is worker._preempt_event
        return SimpleNamespace(finished=True, scheduler_state=None)

    worker.model_runner.execute_model.side_effect = _execute

    worker.execute_model(request, worker.od_config)

    assert worker.model_runner.pipeline._interrupt is False
    assert worker.model_runner.pipeline._interrupt_event is None
