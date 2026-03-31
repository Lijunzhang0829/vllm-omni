from types import SimpleNamespace

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.outputs import OmniRequestOutput


class _Metrics:
    def __init__(self):
        self.stage_last_ts = [0.0]

    def process_stage_metrics(self, **kwargs):
        self.last_kwargs = kwargs


def test_process_single_result_preserves_diffusion_metrics_on_final_image_output():
    orchestrator = SimpleNamespace(_name="test-orchestrator")

    engine_output = OmniRequestOutput.from_diffusion(
        request_id="req-1",
        images=[],
        prompt={"prompt": "hello"},
        metrics={
            "super_p95_normal_load_s": 1.25,
            "super_p95_sacrificial_load_s": 0.5,
        },
    )
    engine_output.images = [object()]

    stage = SimpleNamespace(final_output=True, final_output_type="image", stage_type="diffusion")
    metrics = _Metrics()
    result = {
        "request_id": "req-1",
        "engine_outputs": engine_output,
    }

    _, finished, output_to_yield = AsyncOmni._process_single_result(orchestrator, result, stage, 0, metrics)

    assert finished is True
    assert output_to_yield is not None
    assert output_to_yield.metrics["super_p95_normal_load_s"] == 1.25
    assert output_to_yield.metrics["super_p95_sacrificial_load_s"] == 0.5


def test_capture_super_p95_headers_falls_back_to_nested_request_output_metrics():
    nested = OmniRequestOutput.from_diffusion(
        request_id="req-2",
        images=[object()],
        prompt={"prompt": "hello"},
        metrics={
            "super_p95_normal_load_s": 2.5,
            "super_p95_sacrificial_load_s": 0.25,
        },
    )
    outer = OmniRequestOutput(
        request_id="req-2",
        stage_id=0,
        final_output_type="image",
        request_output=nested,
        images=[object()],
        metrics={},
        finished=True,
    )

    headers = OmniOpenAIServingChat._capture_super_p95_headers_from_result(outer)

    assert headers["X-Super-P95-Normal-Load-S"] == "2.500000"
    assert headers["X-Super-P95-Sacrificial-Load-S"] == "0.250000"
