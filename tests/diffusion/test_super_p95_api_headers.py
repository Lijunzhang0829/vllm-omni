# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from starlette.requests import Request

from vllm_omni.entrypoints.openai.api_server import (
    _get_super_p95_post_arrival_headers,
    _get_super_p95_response_headers,
)


class _FakeScheduler:
    def get_load_snapshot(self):
        return SimpleNamespace(normal_load_s=12.5, sacrificial_load_s=3.25)


def test_get_super_p95_response_headers_reads_scheduler_snapshot() -> None:
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/health",
            "headers": [],
            "app": SimpleNamespace(
                state=SimpleNamespace(
                    diffusion_engine=SimpleNamespace(engine=SimpleNamespace(scheduler=_FakeScheduler()))
                )
            ),
        }
    )

    headers = _get_super_p95_response_headers(request)

    assert headers == {
        "X-Super-P95-Normal-Load-S": "12.500000",
        "X-Super-P95-Sacrificial-Load-S": "3.250000",
    }


def test_get_super_p95_post_arrival_headers_adds_incoming_request_load() -> None:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/videos",
            "headers": [],
            "app": SimpleNamespace(
                state=SimpleNamespace(
                    diffusion_engine=SimpleNamespace(engine=SimpleNamespace(scheduler=_FakeScheduler()))
                )
            ),
        }
    )

    headers = _get_super_p95_post_arrival_headers(
        request,
        {
            "_super_p95_sacrificial": True,
            "_super_p95_estimated_service_s": 7.5,
        },
    )

    assert headers == {
        "X-Super-P95-Normal-Load-S": "12.500000",
        "X-Super-P95-Sacrificial-Load-S": "10.750000",
    }
