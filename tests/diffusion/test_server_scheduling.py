from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.server_scheduling import PredictedLatencyPolicy, estimate_service_time_s
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def _make_request(
    *,
    width: int,
    height: int,
    num_inference_steps: int,
    num_frames: int = 1,
) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=["test"],
        sampling_params=OmniDiffusionSamplingParams(
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
        ),
        request_ids=["req"],
    )


def test_estimate_service_time_uses_known_anchor():
    request = _make_request(width=1024, height=1024, num_inference_steps=25)
    assert estimate_service_time_s(request) == 33.90


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
