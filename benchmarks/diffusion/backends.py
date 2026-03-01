import base64
import mimetypes
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from tqdm import tqdm


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
        async with session.post(input.api_url, json=payload) as response:
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
    """Send request to vLLM-Omni's /v1/videos endpoint."""
    output = RequestFuncOutput()
    output.start_time = time.perf_counter()

    form = aiohttp.FormData()
    form.add_field("model", input.model)
    form.add_field("prompt", input.prompt)

    scalar_fields = {
        "width": input.width,
        "height": input.height,
        "num_frames": input.num_frames,
        "num_inference_steps": input.num_inference_steps,
        "seed": input.seed,
        "fps": input.fps,
    }
    for key, value in scalar_fields.items():
        if value is not None:
            form.add_field(key, str(value))

    for key, value in input.extra_body.items():
        if value is not None:
            form.add_field(key, str(value))

    if input.image_paths and len(input.image_paths) > 0:
        image_path = input.image_paths[0]
        if not os.path.exists(image_path):
            output.error = f"Image file not found: {image_path}"
            output.success = False
            if pbar:
                pbar.update(1)
            return output
        with open(image_path, "rb") as image_file:
            form.add_field(
                "input_reference",
                image_file,
                filename=os.path.basename(image_path),
                content_type=_guess_mime_type(image_path),
            )
            try:
                async with session.post(input.api_url, data=form) as response:
                    if response.status == 200:
                        output.response_body = await response.json()
                        output.success = True
                    else:
                        output.error = f"HTTP {response.status}: {await response.text()}"
                        output.success = False
            except Exception as e:
                output.error = str(e)
                output.success = False
    else:
        try:
            async with session.post(input.api_url, data=form) as response:
                if response.status == 200:
                    output.response_body = await response.json()
                    output.success = True
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


backends_function_mapping = {
    "vllm-omni": (async_request_chat_completions, "/v1/chat/completions"),
    "openai": (async_request_openai_images, "/v1/images/generations"),
}


def get_backend_request_config(backend: str, task: str) -> tuple[Any, str]:
    """Resolve request function and endpoint by backend and task."""
    if backend == "vllm-omni" and task in {"t2v", "i2v", "ti2v"}:
        return async_request_v1_videos, "/v1/videos"
    return backends_function_mapping[backend]
