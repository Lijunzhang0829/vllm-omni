# Qwen-Image Preemption Overhead A/B

Date: 2026-03-26

## Goal

Measure the overhead introduced by diffusion preemption recovery on the same Qwen-Image serving workload, and compare the old fixed chunk-step policy with the newer arrival-triggered async preemption policy.

## Environment

- Model: `Qwen/Qwen-Image`
- Device: single NPU, `ASCEND_RT_VISIBLE_DEVICES=0`
- Platform plugin: `ascend`
- Server flags: `--omni --vae-use-slicing --vae-use-tiling`
- Benchmark entry: `benchmarks/diffusion/diffusion_benchmark_serving.py`
- Backend: `vllm-omni`
- Task: `t2i`
- Number of measured requests: `5`
- `--max-concurrency 1000`
- `--request-rate 1`
- Diffusion seed: `--seed 0`
- Request sampling seed: `--random-request-seed 8`

## Workload Definition

Input request profile set:

```json
[
  {"width":512,"height":512,"num_inference_steps":20,"weight":0.15},
  {"width":768,"height":768,"num_inference_steps":20,"weight":0.25},
  {"width":1024,"height":1024,"num_inference_steps":25,"weight":0.45},
  {"width":1536,"height":1536,"num_inference_steps":35,"weight":0.15}
]
```

The `weight` field was only used for request sampling. It was not sent to the server.

Actual sampled 5-request sequence for `--random-request-seed 8`:

```json
[
  {"width": 768, "height": 768, "num_inference_steps": 20},
  {"width": 1536, "height": 1536, "num_inference_steps": 35},
  {"width": 512, "height": 512, "num_inference_steps": 20},
  {"width": 1024, "height": 1024, "num_inference_steps": 25},
  {"width": 512, "height": 512, "num_inference_steps": 20}
]
```

## Server Commands

Preemption disabled:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_PLUGINS=ascend
export VLLM_OMNI_ENABLE_DIFFUSION_PREEMPTION=0
vllm serve Qwen/Qwen-Image --omni --port 8091 --vae-use-slicing --vae-use-tiling
```

Preemption enabled:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_PLUGINS=ascend
export VLLM_OMNI_ENABLE_DIFFUSION_PREEMPTION=1
vllm serve Qwen/Qwen-Image --omni --port 8091 --vae-use-slicing --vae-use-tiling
```

## Benchmark Command

```bash
env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python benchmarks/diffusion/diffusion_benchmark_serving.py \
  --model Qwen/Qwen-Image \
  --backend vllm-omni \
  --dataset random \
  --task t2i \
  --num-prompts 5 \
  --max-concurrency 1000 \
  --request-rate 1 \
  --seed 0 \
  --random-request-seed 8 \
  --random-request-config '[
    {"width":512,"height":512,"num_inference_steps":20,"weight":0.15},
    {"width":768,"height":768,"num_inference_steps":20,"weight":0.25},
    {"width":1024,"height":1024,"num_inference_steps":25,"weight":0.45},
    {"width":1536,"height":1536,"num_inference_steps":35,"weight":0.15}
  ]' \
  --disable-tqdm \
  --output-file <OUTPUT_JSON>
```

## Results

### Fixed Chunk-Step Preemption

This was the earlier implementation that periodically yielded with fixed chunk sizes.

| Metric | Preemption Off | Preemption On | Delta | Relative Overhead |
| --- | ---: | ---: | ---: | ---: |
| benchmark duration (s) | 80.575 | 84.589 | +4.014 | +4.98% |
| throughput (req/s) | 0.0621 | 0.0591 | -0.0029 | -4.75% |
| latency mean (s) | 52.802 | 55.426 | +2.624 | +4.97% |
| latency median (s) | 57.415 | 60.382 | +2.967 | +5.17% |
| latency p95 (s) | 75.139 | 79.055 | +3.916 | +5.21% |
| latency p99 (s) | 76.283 | 80.277 | +3.994 | +5.24% |

### Arrival-Triggered Async Preemption

This is the current implementation:

- preemption decision only happens on request arrival
- the active request keeps running until a step boundary sees the shared preemption event
- there is no fixed periodic chunk budget anymore

| Metric | Baseline Off | Async Preemption On | Delta | Relative Change |
| --- | ---: | ---: | ---: | ---: |
| benchmark duration (s) | 80.575 | 79.851 | -0.724 | -0.90% |
| throughput (req/s) | 0.0621 | 0.0626 | +0.0006 | +0.91% |
| latency mean (s) | 52.802 | 52.309 | -0.493 | -0.93% |
| latency median (s) | 57.415 | 56.849 | -0.566 | -0.99% |
| latency p95 (s) | 75.139 | 74.444 | -0.695 | -0.92% |
| latency p99 (s) | 76.283 | 75.565 | -0.718 | -0.94% |

## Raw Metric Files

- `/tmp/qwen_image_preemption_off_metrics.json`
- `/tmp/qwen_image_preemption_on_metrics.json`
- `/tmp/qwen_image_preemption_on_async_event_metrics.json`

## Notes

- All runs used the exact same sampled 5-request sequence and the same diffusion seed.
- To make this A/B run reproducible, the benchmark CLI was extended with `--random-request-seed`.
- The earlier fixed chunk-step policy showed a consistent ~5% penalty on this workload.
- The newer arrival-triggered async event policy removed that penalty on the same workload and was slightly faster than baseline in this single run.
- For this benchmark only, the diffusion scheduler was given a minimal environment-variable switch: `VLLM_OMNI_ENABLE_DIFFUSION_PREEMPTION=0/1`.
