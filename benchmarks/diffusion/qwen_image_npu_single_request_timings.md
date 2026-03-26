# Qwen-Image NPU Single-Request Timings

Date: 2026-03-26

## Setup

- Model: `Qwen/Qwen-Image`
- Device: single NPU, `ASCEND_RT_VISIBLE_DEVICES=0`
- Platform plugin: `ascend`
- Server flags: `--vae-use-slicing --vae-use-tiling`
- Prompt: `a small red cabin in snowy mountains, cinematic lighting`
- Seed: `0`
- Timing method: wall-clock elapsed time around one client request
- Note: the input `weight` field was ignored on purpose; it is not part of request execution
- Note: local requests must bypass the machine's proxy, otherwise `127.0.0.1:8091` is routed to `http_proxy` and returns `502`

## Server Command

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_PLUGINS=ascend
vllm serve Qwen/Qwen-Image --omni --port 8091 --vae-use-slicing --vae-use-tiling
```

## Client Command Template

```bash
env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost bash -lc "
TIMEFORMAT='elapsed_seconds=%R'
time python examples/online_serving/text_to_image/openai_chat_client.py \
  --server http://127.0.0.1:8091 \
  --prompt 'a small red cabin in snowy mountains, cinematic lighting' \
  --width <WIDTH> \
  --height <HEIGHT> \
  --steps <STEPS> \
  --seed 0 \
  --output /tmp/qwen_image_<WIDTH>x<HEIGHT>_<STEPS>.png
"
```

## Results

| width | height | num_inference_steps | weight_ignored | elapsed_seconds |
| --- | --- | --- | --- | --- |
| 512 | 512 | 20 | 0.15 | 8.930 |
| 768 | 768 | 20 | 0.25 | 9.438 |
| 1024 | 1024 | 25 | 0.45 | 13.411 |
| 1536 | 1536 | 35 | 0.15 | 43.895 |

## Output Files

- `/tmp/qwen_image_512x512_20.png`
- `/tmp/qwen_image_768x768_20.png`
- `/tmp/qwen_image_1024x1024_25.png`
- `/tmp/qwen_image_1536x1536_35.png`

## Notes

- These numbers were measured sequentially against the same already-warmed server instance.
- Before measuring, I fixed a regression where diffusion warmup/preemption state handling could require a `request_id` on warmup requests and break server startup.
