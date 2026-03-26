import threading
from types import SimpleNamespace

from vllm_omni.diffusion.scheduler import Scheduler
from vllm_omni.diffusion.server_scheduling import PredictedLatencyPolicy


def _make_request(request_id: str, *, width: int, height: int, steps: int):
    return SimpleNamespace(
        request_ids=[request_id],
        sampling_params=SimpleNamespace(
            width=width,
            height=height,
            resolution=None,
            num_inference_steps=steps,
            num_frames=None,
            extra_args={},
        ),
    )


def _make_scheduler():
    sched = object.__new__(Scheduler)
    sched._pending_cv = threading.Condition()
    sched._policy = PredictedLatencyPolicy()
    sched._preemption_enabled = True
    sched._active_scheduled = None
    sched._active_request_preemptible = False
    sched._active_preemption_requested = False
    sched._preempt_event = threading.Event()
    sched.od_config = SimpleNamespace(model_class_name="QwenImagePipeline")
    return sched


def test_arrival_requests_preemption_when_pending_outranks_active():
    sched = _make_scheduler()
    active = sched._policy.add_request(_make_request("active", width=512, height=512, steps=20))
    sched._policy.pop_next_request()
    sched._mark_active_request_locked(active)
    sched._policy.add_request(_make_request("new", width=1536, height=1536, steps=35))

    should_preempt = sched._maybe_request_active_preemption_locked()

    assert should_preempt is True
    assert sched._active_preemption_requested is True


def test_arrival_does_not_preempt_when_pending_is_not_better_than_active():
    sched = _make_scheduler()
    active = sched._policy.add_request(_make_request("active", width=1536, height=1536, steps=35))
    sched._policy.pop_next_request()
    sched._mark_active_request_locked(active)
    sched._policy.add_request(_make_request("new", width=512, height=512, steps=20))

    should_preempt = sched._maybe_request_active_preemption_locked()

    assert should_preempt is False
    assert sched._active_preemption_requested is False


def test_arrival_only_requests_preemption_once_per_active_request():
    sched = _make_scheduler()
    active = sched._policy.add_request(_make_request("active", width=512, height=512, steps=20))
    sched._policy.pop_next_request()
    sched._mark_active_request_locked(active)
    sched._policy.add_request(_make_request("new", width=1536, height=1536, steps=35))

    first = sched._maybe_request_active_preemption_locked()
    second = sched._maybe_request_active_preemption_locked()

    assert first is True
    assert second is False


def test_mark_and_clear_active_request_manage_preempt_event():
    sched = _make_scheduler()
    sched._preempt_event.set()
    active = sched._policy.add_request(_make_request("active", width=512, height=512, steps=20))
    sched._policy.pop_next_request()

    sched._mark_active_request_locked(active)
    assert sched._preempt_event.is_set() is False

    sched._preempt_event.set()
    sched._clear_active_request_locked()
    assert sched._preempt_event.is_set() is False
