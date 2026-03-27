import threading
from types import SimpleNamespace

from vllm_omni.diffusion.scheduler import Scheduler
from vllm_omni.diffusion.server_scheduling import PredictedLatencyPolicy
from vllm_omni.diffusion.super_p95 import (
    EXTRA_ARG_SUPER_P95_ESTIMATED_SERVICE_S,
    EXTRA_ARG_SUPER_P95_SACRIFICIAL,
)


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


def _make_scheduler(model_class_name: str = "QwenImagePipeline"):
    sched = object.__new__(Scheduler)
    sched._server_scheduling_enabled = True
    sched._pending_cv = threading.Condition()
    sched._policy = PredictedLatencyPolicy()
    sched._preemption_enabled = True
    sched._active_scheduled = None
    sched._active_request_preemptible = False
    sched._active_preemption_requested = False
    sched._preempt_event = threading.Event()
    sched.od_config = SimpleNamespace(model_class_name=model_class_name)
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


def test_arrival_does_not_preempt_when_preemption_is_disabled():
    sched = _make_scheduler()
    sched._preemption_enabled = False
    active = sched._policy.add_request(_make_request("active", width=512, height=512, steps=20))
    sched._policy.pop_next_request()
    sched._mark_active_request_locked(active)
    sched._policy.add_request(_make_request("new", width=1536, height=1536, steps=35))

    should_preempt = sched._maybe_request_active_preemption_locked()

    assert should_preempt is False
    assert sched._active_request_preemptible is False
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


def test_wan22_pipeline_is_treated_as_async_preemptible():
    sched = _make_scheduler(model_class_name="Wan22Pipeline")
    req = _make_request("wan", width=854, height=480, steps=3)

    assert sched._supports_async_preemption(req) is True


def test_wan_pipeline_is_treated_as_async_preemptible():
    sched = _make_scheduler(model_class_name="WanPipeline")
    req = _make_request("wan", width=854, height=480, steps=3)

    assert sched._supports_async_preemption(req) is True


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


def test_normal_pending_preempts_active_sacrificial_request():
    sched = _make_scheduler()
    active_request = _make_request("active", width=1536, height=1536, steps=35)
    active_request.sampling_params.extra_args[EXTRA_ARG_SUPER_P95_SACRIFICIAL] = True
    active = sched._policy.add_request(active_request)
    sched._policy.pop_next_request()
    sched._mark_active_request_locked(active)
    sched._policy.add_request(_make_request("new", width=512, height=512, steps=20))

    should_preempt = sched._maybe_request_active_preemption_locked()

    assert should_preempt is True


def test_add_req_bypasses_server_scheduling_when_disabled():
    sched = object.__new__(Scheduler)
    sched._server_scheduling_enabled = False
    sched._pending_cv = threading.Condition()
    sched._direct_normal_load_s = 0.0
    sched._direct_sacrificial_load_s = 0.0
    sched.od_config = SimpleNamespace(super_p95_hardware_profile="910B3")

    seen = []
    expected = object()

    def _fake_execute(request):
        seen.append(request)
        return expected

    sched._execute_request = _fake_execute

    req = _make_request("direct", width=512, height=512, steps=20)
    assert sched.add_req(req) is expected
    assert seen == [req]


def test_direct_mode_tracks_authoritative_load_while_request_is_running():
    sched = object.__new__(Scheduler)
    sched._server_scheduling_enabled = False
    sched._pending_cv = threading.Condition()
    sched._direct_normal_load_s = 0.0
    sched._direct_sacrificial_load_s = 0.0
    sched.od_config = SimpleNamespace(super_p95_hardware_profile="910B2")

    entered = threading.Event()
    release = threading.Event()
    result = object()

    def _fake_execute(_request):
        entered.set()
        release.wait(timeout=5.0)
        return result

    sched._execute_request = _fake_execute

    req = _make_request("direct", width=512, height=512, steps=20)
    req.sampling_params.extra_args[EXTRA_ARG_SUPER_P95_ESTIMATED_SERVICE_S] = 12.5

    output_holder = {}

    def _run():
        output_holder["result"] = sched.add_req(req)

    worker = threading.Thread(target=_run)
    worker.start()
    assert entered.wait(timeout=5.0) is True

    snapshot = sched.get_super_p95_load_snapshot()
    assert snapshot.normal_load_s == 12.5
    assert snapshot.sacrificial_load_s == 0.0

    release.set()
    worker.join(timeout=5.0)
    assert output_holder["result"] is result

    final_snapshot = sched.get_super_p95_load_snapshot()
    assert final_snapshot.normal_load_s == 0.0
    assert final_snapshot.sacrificial_load_s == 0.0


def test_get_super_p95_load_snapshot_includes_active_remaining_work():
    sched = _make_scheduler()
    pending = sched._policy.add_request(_make_request("pending", width=512, height=512, steps=20))
    active = sched._policy.add_request(_make_request("active", width=1536, height=1536, steps=35))
    popped = sched._policy.pop_next_request()
    assert popped is active
    sched._mark_active_request_locked(active)

    snapshot = sched.get_super_p95_load_snapshot()

    assert snapshot.normal_load_s == active.remaining_service_s + pending.remaining_service_s
    assert snapshot.sacrificial_load_s == 0.0
