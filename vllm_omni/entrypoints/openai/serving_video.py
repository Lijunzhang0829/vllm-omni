# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import os
import time
import uuid
from http import HTTPStatus
from typing import Any, cast

from fastapi import HTTPException, Request
from PIL import Image
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.image_api_utils import parse_size
from vllm_omni.entrypoints.openai.protocol.videos import (
    VideoData,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from vllm_omni.entrypoints.openai.video_api_utils import decode_input_reference, encode_video_base64
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniSamplingParams, OmniTextPrompt
from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id

logger = init_logger(__name__)

_VIDEO_GENERATION_TIMEOUT_ENV = "VLLM_OMNI_VIDEO_GENERATION_TIMEOUT_S"
_VIDEO_ENCODING_TIMEOUT_ENV = "VLLM_OMNI_VIDEO_ENCODING_TIMEOUT_S"
_DEFAULT_VIDEO_ENCODING_TIMEOUT_S = 300.0


class OmniOpenAIServingVideo:
    """OpenAI-style video generation handler for omni diffusion models."""

    def __init__(
        self,
        engine_client: EngineClient,
        model_name: str | None = None,
        stage_configs: list[Any] | None = None,
    ) -> None:
        self._engine_client = engine_client
        self._model_name = model_name
        self._stage_configs = stage_configs

    @staticmethod
    def _get_timeout_seconds(env_var: str, default: float | None = None) -> float | None:
        raw_value = os.environ.get(env_var)
        if raw_value is None or raw_value == "":
            return default
        try:
            timeout_s = float(raw_value)
        except ValueError:
            logger.warning("Invalid %s=%r; ignoring timeout override.", env_var, raw_value)
            return default
        if timeout_s <= 0:
            return None
        return timeout_s

    @staticmethod
    def _timeout_detail(stage_name: str, request_id: str, timeout_s: float) -> str:
        return f"Video request {request_id} timed out during {stage_name} after {timeout_s:.1f}s."

    async def _run_generation_with_timeout(
        self,
        prompt: OmniTextPrompt,
        gen_params: OmniDiffusionSamplingParams,
        request_id: str,
        raw_request: Request | None,
    ) -> Any:
        timeout_s = self._get_timeout_seconds(_VIDEO_GENERATION_TIMEOUT_ENV)
        if timeout_s is None:
            return await self._run_generation(prompt, gen_params, request_id, raw_request)

        logger.info(
            "Applying generation timeout %.2fs for video request %s via %s.",
            timeout_s,
            request_id,
            _VIDEO_GENERATION_TIMEOUT_ENV,
        )
        try:
            return await asyncio.wait_for(
                self._run_generation(prompt, gen_params, request_id, raw_request),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as exc:
            logger.error(
                "Video request %s timed out during generation after %.2fs.",
                request_id,
                timeout_s,
            )
            raise HTTPException(
                status_code=HTTPStatus.GATEWAY_TIMEOUT.value,
                detail=self._timeout_detail("generation", request_id, timeout_s),
            ) from exc

    async def _encode_video_data(self, video: Any, fps: int, request_id: str, video_index: int) -> VideoData:
        if isinstance(video, str):
            logger.info(
                "Video encoding skipped for request %s item=%d; payload already base64-encoded.",
                request_id,
                video_index,
            )
            return VideoData(b64_json=video)
        start = time.perf_counter()
        logger.info(
            "Video encoding started for request %s item=%d fps=%d.",
            request_id,
            video_index,
            fps,
        )
        encoded = await asyncio.to_thread(encode_video_base64, video, fps)
        logger.info(
            "Video encoding finished for request %s item=%d in %.2fs.",
            request_id,
            video_index,
            time.perf_counter() - start,
        )
        return VideoData(b64_json=encoded)

    async def _encode_videos(self, videos: list[Any], fps: int, request_id: str) -> list[VideoData]:
        timeout_s = self._get_timeout_seconds(
            _VIDEO_ENCODING_TIMEOUT_ENV,
            default=_DEFAULT_VIDEO_ENCODING_TIMEOUT_S,
        )
        encode_tasks = [
            self._encode_video_data(video, fps=fps, request_id=request_id, video_index=idx)
            for idx, video in enumerate(videos)
        ]
        try:
            if timeout_s is None:
                return await asyncio.gather(*encode_tasks)
            logger.info(
                "Applying encoding timeout %.2fs for video request %s via %s.",
                timeout_s,
                request_id,
                _VIDEO_ENCODING_TIMEOUT_ENV,
            )
            return await asyncio.wait_for(asyncio.gather(*encode_tasks), timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            logger.error(
                "Video request %s timed out during encoding after %.2fs.",
                request_id,
                timeout_s,
            )
            raise HTTPException(
                status_code=HTTPStatus.GATEWAY_TIMEOUT.value,
                detail=self._timeout_detail("encoding", request_id, timeout_s),
            ) from exc

    @classmethod
    def for_diffusion(
        cls,
        diffusion_engine: EngineClient,
        model_name: str,
        stage_configs: list[Any] | None = None,
    ) -> OmniOpenAIServingVideo:
        return cls(
            diffusion_engine,
            model_name=model_name,
            stage_configs=stage_configs,
        )

    async def generate_videos(
        self,
        request: VideoGenerationRequest,
        raw_request: Request | None = None,
        *,
        input_reference_bytes: bytes | None = None,
    ) -> VideoGenerationResponse:
        if request.stream:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Streaming video generation is not supported yet.",
            )

        request_id = f"video_gen_{uuid.uuid4().hex}"
        model_name = self._resolve_model_name(raw_request)

        if request.model is not None and model_name is not None and request.model != model_name:
            logger.warning(
                "Model mismatch: request specifies '%s' but server is running '%s'. Using server model.",
                request.model,
                model_name,
            )

        prompt: OmniTextPrompt = {"prompt": request.prompt}
        if request.negative_prompt is not None:
            prompt["negative_prompt"] = request.negative_prompt

        input_image = None
        try:
            input_image = decode_input_reference(request.input_reference, input_reference_bytes)
        except ValueError as exc:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=str(exc),
            ) from exc
        if input_image is not None:
            prompt["multi_modal_data"] = {"image": input_image}

        gen_params = OmniDiffusionSamplingParams(num_outputs_per_prompt=request.n)

        width, height, num_frames, fps = self._resolve_video_params(request)
        if width is not None and height is not None:
            gen_params.width = width
            gen_params.height = height
        if num_frames is not None:
            gen_params.num_frames = num_frames
        if fps is not None:
            gen_params.fps = fps

        if request.num_inference_steps is not None:
            gen_params.num_inference_steps = request.num_inference_steps
        if request.guidance_scale is not None:
            gen_params.guidance_scale = request.guidance_scale
        if request.guidance_scale_2 is not None:
            gen_params.guidance_scale_2 = request.guidance_scale_2
        if request.true_cfg_scale is not None:
            gen_params.true_cfg_scale = request.true_cfg_scale
        if request.seed is not None:
            gen_params.seed = request.seed
        if request.boundary_ratio is not None:
            gen_params.boundary_ratio = request.boundary_ratio

        logger.info(
            "Boundary ratio parse: request=%s gen_params=%s",
            request.boundary_ratio,
            gen_params.boundary_ratio,
        )
        if request.flow_shift is not None:
            gen_params.extra_args["flow_shift"] = request.flow_shift

        self._apply_lora(request.lora, gen_params)

        logger.info(
            "Video sampling params: steps=%s guidance=%s guidance_2=%s seed=%s",
            gen_params.num_inference_steps,
            gen_params.guidance_scale,
            gen_params.guidance_scale_2,
            gen_params.seed,
        )

        generation_start = time.perf_counter()
        result = await self._run_generation_with_timeout(prompt, gen_params, request_id, raw_request)
        logger.info(
            "Video generation completed for request %s in %.2fs",
            request_id,
            time.perf_counter() - generation_start,
        )

        extract_start = time.perf_counter()
        videos = self._extract_video_outputs(result)
        logger.info(
            "Extracted %d video payload(s) for request %s in %.2fs",
            len(videos),
            request_id,
            time.perf_counter() - extract_start,
        )
        output_fps = fps or 24

        encode_start = time.perf_counter()
        video_data = await self._encode_videos(videos, fps=output_fps, request_id=request_id)
        logger.info(
            "Encoded %d video payload(s) for request %s in %.2fs",
            len(video_data),
            request_id,
            time.perf_counter() - encode_start,
        )
        return VideoGenerationResponse(created=int(time.time()), data=video_data)

    def _resolve_model_name(self, raw_request: Request | None) -> str | None:
        if self._model_name:
            return self._model_name
        if raw_request is None:
            return None
        serving_models = getattr(raw_request.app.state, "openai_serving_models", None)
        if serving_models and getattr(serving_models, "base_model_paths", None):
            base_paths = serving_models.base_model_paths
            if base_paths:
                return base_paths[0].name
        return None

    @staticmethod
    def _resolve_video_params(request: VideoGenerationRequest) -> tuple[int | None, int | None, int | None, int | None]:
        width = request.width or (request.video_params.width if request.video_params else None)
        height = request.height or (request.video_params.height if request.video_params else None)
        num_frames = request.num_frames or (request.video_params.num_frames if request.video_params else None)
        fps = request.fps or (request.video_params.fps if request.video_params else None)
        seconds = request.seconds

        if request.size:
            width, height = parse_size(request.size)

        if fps is None:
            fps = 24  # Default FPS if not specified

        if num_frames is None and seconds is not None:
            num_frames = int(seconds) * int(fps)

        return width, height, num_frames, fps

    @staticmethod
    def _apply_lora(lora_body: Any, gen_params: OmniDiffusionSamplingParams) -> None:
        if lora_body is None:
            return
        if not isinstance(lora_body, dict):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Invalid lora field: expected an object.",
            )
        lora_name = lora_body.get("name") or lora_body.get("lora_name") or lora_body.get("adapter")
        lora_path = (
            lora_body.get("local_path")
            or lora_body.get("path")
            or lora_body.get("lora_path")
            or lora_body.get("lora_local_path")
        )
        lora_scale = lora_body.get("scale")
        if lora_scale is None:
            lora_scale = lora_body.get("lora_scale")
        lora_int_id = lora_body.get("int_id")
        if lora_int_id is None:
            lora_int_id = lora_body.get("lora_int_id")
        if lora_int_id is None and lora_path:
            lora_int_id = stable_lora_int_id(str(lora_path))

        if not lora_name or not lora_path:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Invalid lora object: both name and path are required.",
            )

        gen_params.lora_request = LoRARequest(str(lora_name), int(lora_int_id), str(lora_path))
        if lora_scale is not None:
            gen_params.lora_scale = float(lora_scale)

    async def _run_generation(
        self,
        prompt: OmniTextPrompt,
        gen_params: OmniDiffusionSamplingParams,
        request_id: str,
        raw_request: Request | None,
    ) -> Any:
        has_stage_list = hasattr(self._engine_client, "stage_list")
        logger.info(
            "Video generation routing: stage_configs=%s, has_stage_list=%s, engine_type=%s",
            "present"
            if (self._stage_configs or (getattr(raw_request.app.state, "stage_configs", None) if raw_request else None))
            else "missing",
            has_stage_list,
            type(self._engine_client).__name__,
        )
        stage_configs = (
            self._stage_configs
            or (getattr(raw_request.app.state, "stage_configs", None) if raw_request else None)
            or getattr(self._engine_client, "stage_configs", None)
        )

        if not stage_configs:
            if not hasattr(self._engine_client, "stage_list"):
                raise HTTPException(
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                    detail="Stage configs not found. Start server with an omni diffusion model.",
                )

        # Video generation endpoint only supports diffusion stages.
        if stage_configs:
            for stage in stage_configs:
                # Extract stage_type: dicts and OmegaConf objects use .get(), others use getattr
                if hasattr(stage, "get"):
                    stage_type = stage.get("stage_type", "llm")
                else:
                    stage_type = getattr(stage, "stage_type", "llm")

                if stage_type != "diffusion":
                    raise HTTPException(
                        status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                        detail=f"Video generation only supports diffusion stages, found '{stage_type}' stage.",
                    )

        # Common generation logic for both paths
        engine_client = cast(AsyncOmni, self._engine_client)
        stage_list = getattr(engine_client, "stage_list", None)
        if isinstance(stage_list, list):
            sampling_params_list: list[OmniSamplingParams] = [gen_params for _ in stage_list]
        else:
            sampling_params_list = [gen_params]

        result = None
        async for output in engine_client.generate(
            prompt=prompt,
            request_id=request_id,
            sampling_params_list=sampling_params_list,
        ):
            result = output
            if getattr(output, "finished", False):
                return output

        if result is None:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail="No output generated from video generation pipeline.",
            )
        return result

    @staticmethod
    def _normalize_video_outputs(videos: Any) -> list[Any]:
        if videos is None:
            return []
        if hasattr(videos, "ndim") and videos.ndim == 5:
            return [videos[i] for i in range(videos.shape[0])]
        if isinstance(videos, list):
            if not videos:
                return []
            first = videos[0]
            if hasattr(first, "ndim") and first.ndim == 5:
                flattened: list[Any] = []
                for item in videos:
                    if hasattr(item, "ndim") and item.ndim == 5:
                        flattened.extend([item[i] for i in range(item.shape[0])])
                    else:
                        flattened.append(item)
                return flattened
            if isinstance(first, list):
                return videos
            if hasattr(first, "ndim") and first.ndim == 3:
                return [videos]
            if isinstance(first, Image.Image):
                return [videos]
            return videos
        return [videos]

    def _extract_video_outputs(self, result: Any) -> list[Any]:
        videos = None
        if hasattr(result, "images") and result.images:
            videos = result.images
        elif hasattr(result, "request_output"):
            request_output = result.request_output
            if (
                isinstance(request_output, dict)
                and isinstance(request_output.get("multimodal_output"), dict)
                and request_output["multimodal_output"].get("video_b64")
            ):
                videos = request_output["multimodal_output"]["video_b64"]
            elif isinstance(request_output, dict) and request_output.get("images"):
                videos = request_output["images"]
            elif (
                hasattr(request_output, "multimodal_output")
                and request_output.multimodal_output
                and request_output.multimodal_output.get("video_b64")
            ):
                videos = request_output.multimodal_output.get("video_b64")
            elif hasattr(request_output, "images") and request_output.images:
                videos = request_output.images
            elif hasattr(request_output, "multimodal_output") and request_output.multimodal_output:
                videos = request_output.multimodal_output.get("video")
        if videos is None and hasattr(result, "multimodal_output") and result.multimodal_output:
            videos = result.multimodal_output.get("video_b64") or result.multimodal_output.get("video")

        normalized = self._normalize_video_outputs(videos)
        if not normalized:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail="No video outputs found in generation result.",
            )
        return normalized
