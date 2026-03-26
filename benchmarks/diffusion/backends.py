import base64
import mimetypes
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from tqdm import tqdm


DELAY_X_ESTIMATED_SERVICE_HEADER = "X-DelayX-Estimated-Service-S"
QWEN_IMAGE_PROFILE_LATENCY_S: dict[tuple[int, int, int, int], float] = {
    (512, 512, 20, 1): 22.35,
    (768, 768, 20, 1): 20.62,
    (1024, 1024, 25, 1): 33.90,
    (1536, 1536, 35, 1): 102.66,
}
QWEN_IMAGE_RESOLUTION_ANCHORS: dict[tuple[int, int], tuple[int, int, float]] = {
    (512, 512): (20, 1, 22.35),
    (768, 768): (20, 1, 20.62),
    (1024, 1024): (25, 1, 33.90),
    (1536, 1536): (35, 1, 102.66),
}
QWEN_IMAGE_FALLBACK_REFERENCE_KEY = (1024, 1024, 25, 1)
QWEN_IMAGE_FALLBACK_REFERENCE_LATENCY_S = QWEN_IMAGE_PROFILE_LATENCY_S[QWEN_IMAGE_FALLBACK_REFERENCE_KEY]


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    model: str
    width: int | None = None
    height: int | None = None
    num_frames: int | None = None
    num_inference_steps: int | None = None
    seed: int | None = None
    fps: int | None = None
    timestamp: float | None = None
    slo_ms: float | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    image_paths: list[str] | None = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class RequestFuncOutput:
    success: bool = False
    latency: float = 0.0
    error: str = ""
    start_time: float = 0.0
    response_body: dict[str, Any] = field(default_factory=dict)
    peak_memory_mb: float = 0.0
    slo_achieved: bool | None = None


def estimate_request_service_s(input: RequestFuncInput) -> float:
    width = int(input.width or 1024)
    height = int(input.height or 1024)
    steps = int(input.num_inference_steps or 50)
    num_frames = int(input.num_frames or 1)
    profile_key = (width, height, steps, num_frames)
    if profile_key in QWEN_IMAGE_PROFILE_LATENCY_S:
        return QWEN_IMAGE_PROFILE_LATENCY_S[profile_key]

    anchor = QWEN_IMAGE_RESOLUTION_ANCHORS.get((width, height))
    if anchor is not None:
        anchor_steps, anchor_frames, anchor_latency_s = anchor
        if anchor_steps > 0 and anchor_frames > 0:
            return anchor_latency_s * (float(steps) / float(anchor_steps)) * (float(num_frames) / float(anchor_frames))

    total_cost = max(1, width * height * steps * num_frames)
    reference_cost = (
        QWEN_IMAGE_FALLBACK_REFERENCE_KEY[0]
        * QWEN_IMAGE_FALLBACK_REFERENCE_KEY[1]
        * QWEN_IMAGE_FALLBACK_REFERENCE_KEY[2]
        * QWEN_IMAGE_FALLBACK_REFERENCE_KEY[3]
    )
    return QWEN_IMAGE_FALLBACK_REFERENCE_LATENCY_S * (float(total_cost) / float(reference_cost))


def build_delay_x_hint_headers(input: RequestFuncInput) -> dict[str, str]:
    return {
        DELAY_X_ESTIMATED_SERVICE_HEADER: f"{estimate_request_service_s(input):.6f}",
    }


def _guess_mime_type(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def _encode_image_as_data_url(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    mime = _guess_mime_type(path)
    return f"data:{mime};base64,{encoded}"


async def async_request_chat_completions(
    input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    output = RequestFuncOutput()
    output.start_time = time.perf_counter()

    extra_body = dict(input.extra_body)
    if input.width and input.height:
        extra_body.setdefault("height", input.height)
        extra_body.setdefault("width", input.width)
    if input.num_frames:
        extra_body.setdefault("num_frames", input.num_frames)
    if input.num_inference_steps:
        extra_body.setdefault("num_inference_steps", input.num_inference_steps)
    if input.seed is not None:
        extra_body.setdefault("seed", input.seed)
    if input.fps:
        extra_body.setdefault("fps", input.fps)

    if input.image_paths and len(input.image_paths) > 0:
        content = []
        if input.prompt:
            content.append({"type": "text", "text": input.prompt})
        for img_path in input.image_paths:
            if not os.path.exists(img_path):
                output.error = f"Image file not found: {img_path}"
                output.success = False
                if pbar:
                    pbar.update(1)
                return output
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image_as_data_url(img_path)},
                }
            )
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": input.prompt}]

    payload = {
        "model": input.model,
        "messages": messages,
    }
    if extra_body:
        payload["extra_body"] = extra_body

    try:
        async with session.post(input.api_url, json=payload, headers=build_delay_x_hint_headers(input)) as response:
            if response.status == 200:
                resp_json = await response.json()
                output.response_body = resp_json
                output.success = True
                if "peak_memory_mb" in resp_json:
                    output.peak_memory_mb = resp_json["peak_memory_mb"]
            else:
                output.error = f"HTTP {response.status}: {await response.text()}"
                output.success = False
    except Exception as e:
        output.error = str(e)
        output.success = False

    output.latency = time.perf_counter() - output.start_time

    if output.success and input.slo_ms is not None:
        output.slo_achieved = (output.latency * 1000.0) <= float(input.slo_ms)

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_images(
    input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    """
    Send request to OpenAI's /v1/images/generations endpoint.
    """
    output = RequestFuncOutput()
    output.start_time = time.perf_counter()

    # Build size string from width/height
    width = input.width or 1024
    height = input.height or 1024
    size = f"{width}x{height}"

    payload: dict[str, Any] = {
        "model": input.model,
        "prompt": input.prompt,
        "n": 1,
        "size": size,
        "response_format": "b64_json",
    }

    # Add optional parameters
    if input.seed is not None:
        payload["seed"] = input.seed
    if input.num_inference_steps is not None:
        payload["num_inference_steps"] = input.num_inference_steps

    # Add any extra body parameters
    if input.extra_body:
        for key, value in input.extra_body.items():
            if key not in payload:
                payload[key] = value

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY",
    }
    headers.update(build_delay_x_hint_headers(input))

    try:
        async with session.post(input.api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                resp_json = await response.json()
                output.response_body = resp_json
                output.success = True
                # Check for usage/memory info if available
                if "usage" in resp_json and "peak_memory_mb" in resp_json.get("usage", {}):
                    output.peak_memory_mb = resp_json["usage"]["peak_memory_mb"]
            else:
                output.error = f"HTTP {response.status}: {await response.text()}"
                output.success = False
    except Exception as e:
        output.error = str(e)
        output.success = False

    output.latency = time.perf_counter() - output.start_time

    if output.success and input.slo_ms is not None:
        output.slo_achieved = (output.latency * 1000.0) <= float(input.slo_ms)

    if pbar:
        pbar.update(1)
    return output


async def async_request_v1_videos(
    input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    output = RequestFuncOutput()
    output.start_time = time.perf_counter()

    files = dict(input.extra_body)
    if input.prompt:
        files.setdefault("prompt", input.prompt)
    if input.width and input.height:
        files.setdefault("height", input.height)
        files.setdefault("width", input.width)
    if input.num_frames:
        files.setdefault("num_frames", input.num_frames)
    if input.num_inference_steps:
        files.setdefault("num_inference_steps", input.num_inference_steps)
    if input.seed is not None:
        files.setdefault("seed", input.seed)
    if input.fps:
        files.setdefault("fps", input.fps)

    form = aiohttp.FormData()
    for k, v in files.items():
        form.add_field(k, str(v))

    image_file = None
    if input.image_paths and len(input.image_paths) > 0:
        image_path = input.image_paths[0]
        image_file = open(image_path, "rb")
        form.add_field(
            "input_reference",
            image_file,
            filename=os.path.basename(image_path),
            content_type="application/octet-stream",
        )

    try:
        async with session.post(input.api_url, data=form, headers=build_delay_x_hint_headers(input)) as response:
            if response.status == 200:
                resp_json = await response.json()
                output.response_body = resp_json
                output.success = True
                if "peak_memory_mb" in resp_json:
                    output.peak_memory_mb = resp_json["peak_memory_mb"]
            else:
                output.error = f"HTTP {response.status}: {await response.text()}"
                output.success = False
    except Exception as e:
        output.error = str(e)
        output.success = False
    finally:
        if image_file is not None:
            image_file.close()

    output.latency = time.perf_counter() - output.start_time

    if output.success and input.slo_ms is not None:
        output.slo_achieved = (output.latency * 1000.0) <= float(input.slo_ms)

    if pbar:
        pbar.update(1)
    return output


backends_function_mapping = {
    "vllm-omni": (async_request_chat_completions, "/v1/chat/completions"),
    "openai": (async_request_openai_images, "/v1/images/generations"),
    "v1/videos": (async_request_v1_videos, "/v1/videos"),
}
