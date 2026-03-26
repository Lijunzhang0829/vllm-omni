from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.server_scheduling import (
    PredictedLatencyPolicy,
    estimate_service_time_s,
    normalize_super_p95_hardware_profile,
)
from vllm_omni.diffusion.super_p95 import (
    EXTRA_ARG_SUPER_P95_ESTIMATED_SERVICE_S,
    EXTRA_ARG_SUPER_P95_SACRIFICIAL,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
import pytest


def _make_request(
    *,
    width: int,
    height: int,
    num_inference_steps: int,
    num_frames: int = 1,
    is_sacrificial: bool = False,
    estimated_service_s: float | None = None,
) -> OmniDiffusionRequest:
    extra_args = {}
    if is_sacrificial:
        extra_args[EXTRA_ARG_SUPER_P95_SACRIFICIAL] = True
    if estimated_service_s is not None:
        extra_args[EXTRA_ARG_SUPER_P95_ESTIMATED_SERVICE_S] = estimated_service_s
    return OmniDiffusionRequest(
        prompts=["test"],
        sampling_params=OmniDiffusionSamplingParams(
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            extra_args=extra_args,
        ),
        request_ids=["req"],
    )


def test_estimate_service_time_uses_known_anchor():
    request = _make_request(width=1024, height=1024, num_inference_steps=25)
    assert estimate_service_time_s(request) == pytest.approx(14.22)


def test_estimate_service_time_uses_requested_hardware_profile():
    request = _make_request(width=1536, height=1536, num_inference_steps=35)
    assert estimate_service_time_s(request, hardware_profile="910B3") == pytest.approx(49.34)


def test_estimate_service_time_defaults_unknown_profile_to_910b2():
    request = _make_request(width=512, height=512, num_inference_steps=20)
    assert normalize_super_p95_hardware_profile("unknown") == "910B2"
    assert estimate_service_time_s(request, hardware_profile="unknown") == pytest.approx(8.60)


def test_predicted_latency_policy_prefers_longer_request():
    policy = PredictedLatencyPolicy()
    short_request = _make_request(width=512, height=512, num_inference_steps=20)
    long_request = _make_request(width=1536, height=1536, num_inference_steps=35)

    policy.add_request(short_request)
    policy.add_request(long_request)

    scheduled = policy.pop_next_request()
    assert scheduled.request is long_request


def test_predicted_latency_policy_tracks_logical_time_on_completion():
    policy = PredictedLatencyPolicy()
    first_request = _make_request(width=512, height=512, num_inference_steps=20)
    second_request = _make_request(width=768, height=768, num_inference_steps=20)

    first = policy.add_request(first_request)
    policy.mark_finished(first)
    second = policy.add_request(second_request)

    assert second.arrival_time_s == first.estimated_service_s


def test_predicted_latency_policy_updates_remaining_after_quantum():
    policy = PredictedLatencyPolicy()
    request = _make_request(width=1024, height=1024, num_inference_steps=25)

    scheduled = policy.add_request(request)
    policy.update_after_quantum(scheduled, completed_steps=5)

    assert scheduled.remaining_steps == 20
    assert scheduled.remaining_service_s == scheduled.estimated_service_s * 20 / 25


def test_predicted_latency_policy_prioritizes_normal_before_sacrificial():
    policy = PredictedLatencyPolicy()
    policy.add_request(_make_request(width=512, height=512, num_inference_steps=20, is_sacrificial=True))
    normal = _make_request(width=768, height=768, num_inference_steps=20)
    policy.add_request(normal)

    scheduled = policy.pop_next_request()

    assert scheduled.request is normal
    assert scheduled.is_sacrificial is False


def test_predicted_latency_policy_uses_sacrificial_order_within_tail_queue():
    policy = PredictedLatencyPolicy()
    slower = policy.add_request(
        _make_request(width=1536, height=1536, num_inference_steps=35, is_sacrificial=True, estimated_service_s=100.0)
    )
    faster = policy.add_request(
        _make_request(width=512, height=512, num_inference_steps=20, is_sacrificial=True, estimated_service_s=10.0)
    )

    scheduled = policy.pop_next_request()

    assert scheduled is faster
    assert scheduled is not slower


def test_predicted_latency_policy_reports_pending_load_snapshot():
    policy = PredictedLatencyPolicy()
    policy.add_request(_make_request(width=512, height=512, num_inference_steps=20, estimated_service_s=5.0))
    policy.add_request(
        _make_request(width=1536, height=1536, num_inference_steps=35, is_sacrificial=True, estimated_service_s=7.0)
    )

    snapshot = policy.get_pending_load_snapshot()

    assert snapshot.normal_load_s == 5.0
    assert snapshot.sacrificial_load_s == 7.0
