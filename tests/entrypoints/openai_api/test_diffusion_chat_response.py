# SPDX-License-Identifier: Apache-2.0
"""Regression tests for diffusion chat response construction."""

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_diffusion_chat_response_accepts_dumped_usage_with_image_content():
    import time

    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionResponseChoice,
        ChatMessage,
    )
    from vllm.entrypoints.openai.engine.protocol import UsageInfo

    from vllm_omni.entrypoints.openai.protocol.chat_completion import (
        OmniChatCompletionResponse,
    )

    message = ChatMessage.model_construct(role="assistant")
    object.__setattr__(
        message,
        "content",
        [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc"},
                "stage_durations": {},
                "peak_memory_mb": 0,
            }
        ],
    )
    if hasattr(message, "__pydantic_fields_set__"):
        message.__pydantic_fields_set__.add("content")

    choice = ChatCompletionResponseChoice.model_construct(
        index=0,
        message=message,
        finish_reason="stop",
        logprobs=None,
        stop_reason=None,
    )
    usage = UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    response = OmniChatCompletionResponse(
        id="req-test",
        created=int(time.time()),
        model="Qwen/Qwen-Image",
        choices=[choice],
        usage=usage.model_dump(exclude_none=True),
    )

    assert response.usage.prompt_tokens == 1
    assert response.choices[0].message.content[0]["type"] == "image_url"
