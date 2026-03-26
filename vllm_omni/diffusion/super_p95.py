# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

HEADER_SUPER_P95_SACRIFICIAL = "X-Super-P95-Sacrificial"
HEADER_SUPER_P95_ESTIMATED_SERVICE_S = "X-Super-P95-Estimated-Service-S"
HEADER_SUPER_P95_NORMAL_LOAD_S = "X-Super-P95-Normal-Load-S"
HEADER_SUPER_P95_SACRIFICIAL_LOAD_S = "X-Super-P95-Sacrificial-Load-S"

EXTRA_ARG_SUPER_P95_SACRIFICIAL = "_super_p95_sacrificial"
EXTRA_ARG_SUPER_P95_ESTIMATED_SERVICE_S = "_super_p95_estimated_service_s"

METRIC_SUPER_P95_NORMAL_LOAD_S = "super_p95_normal_load_s"
METRIC_SUPER_P95_SACRIFICIAL_LOAD_S = "super_p95_sacrificial_load_s"


@dataclass(frozen=True)
class SuperP95LoadSnapshot:
    normal_load_s: float = 0.0
    sacrificial_load_s: float = 0.0


def _parse_bool_header(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _parse_float_header(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def apply_super_p95_request_headers(
    sampling_params: OmniDiffusionSamplingParams,
    headers: Mapping[str, str] | None,
) -> None:
    if not headers:
        return

    extra_args = sampling_params.extra_args
    sacrificial = _parse_bool_header(headers.get(HEADER_SUPER_P95_SACRIFICIAL))
    if sacrificial is not None:
        extra_args[EXTRA_ARG_SUPER_P95_SACRIFICIAL] = sacrificial

    estimated_service_s = _parse_float_header(headers.get(HEADER_SUPER_P95_ESTIMATED_SERVICE_S))
    if estimated_service_s is not None and estimated_service_s >= 0.0:
        extra_args[EXTRA_ARG_SUPER_P95_ESTIMATED_SERVICE_S] = estimated_service_s


def build_super_p95_response_headers(snapshot: SuperP95LoadSnapshot) -> dict[str, str]:
    return {
        HEADER_SUPER_P95_NORMAL_LOAD_S: f"{snapshot.normal_load_s:.6f}",
        HEADER_SUPER_P95_SACRIFICIAL_LOAD_S: f"{snapshot.sacrificial_load_s:.6f}",
    }


def parse_super_p95_load_headers(headers: Mapping[str, str] | None) -> SuperP95LoadSnapshot | None:
    if not headers:
        return None

    normal_load_s = _parse_float_header(headers.get(HEADER_SUPER_P95_NORMAL_LOAD_S))
    sacrificial_load_s = _parse_float_header(headers.get(HEADER_SUPER_P95_SACRIFICIAL_LOAD_S))
    if normal_load_s is None or sacrificial_load_s is None:
        return None
    return SuperP95LoadSnapshot(normal_load_s=normal_load_s, sacrificial_load_s=sacrificial_load_s)


def snapshot_to_metrics(snapshot: SuperP95LoadSnapshot) -> dict[str, float]:
    return {
        METRIC_SUPER_P95_NORMAL_LOAD_S: snapshot.normal_load_s,
        METRIC_SUPER_P95_SACRIFICIAL_LOAD_S: snapshot.sacrificial_load_s,
    }


def parse_super_p95_load_metrics(metrics: Mapping[str, Any] | None) -> SuperP95LoadSnapshot | None:
    if not metrics:
        return None

    normal_load_s = metrics.get(METRIC_SUPER_P95_NORMAL_LOAD_S)
    sacrificial_load_s = metrics.get(METRIC_SUPER_P95_SACRIFICIAL_LOAD_S)
    if not isinstance(normal_load_s, (int, float)) or not isinstance(sacrificial_load_s, (int, float)):
        return None
    return SuperP95LoadSnapshot(
        normal_load_s=float(normal_load_s),
        sacrificial_load_s=float(sacrificial_load_s),
    )


def get_super_p95_request_metadata(extra_args: Mapping[str, Any] | None) -> tuple[bool, float | None]:
    if not extra_args:
        return False, None

    sacrificial = bool(extra_args.get(EXTRA_ARG_SUPER_P95_SACRIFICIAL, False))
    estimated_service_s = extra_args.get(EXTRA_ARG_SUPER_P95_ESTIMATED_SERVICE_S)
    if isinstance(estimated_service_s, (int, float)):
        return sacrificial, float(estimated_service_s)
    return sacrificial, None
