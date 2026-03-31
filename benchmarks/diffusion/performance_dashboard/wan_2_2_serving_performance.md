# Wan2.2 Serving Performance Dashboard

This document describes how to deploy and benchmark **Wan-AI/Wan2.2-T2V-A14B-Diffusers** using vLLM-Omni. It includes service startup configuration, acceleration-related options, benchmark methodology, dataset settings, and performance results.

---

# 1. Overview

Wan-AI/Wan2.2-T2V-A14B-Diffusers is a multimodal text-to-video generation model served through the vLLM-Omni infrastructure.

This document covers:

* Service launch configuration (including acceleration options)
* Benchmark scripts and usage
* Dataset and workload settings
* Performance measurement results
* Reproducibility guidelines

---

# 2. Test Environment
| Component | Specification |
|------------|----------------|
| GPU | NVIDIA A100-SXM4-80GB |
| Diffusion Attention Backend | FlashAttention |

# 3. Service Launch Configuration

## 3.1 Basic Serving Command

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni \
    --port 8091
```

### 3.1.1 NPU 4-card Serving Command (910B2)

Current NPU validation was done on four `910B2` cards with:

- `--usp 4`
- `--enable-layerwise-offload`
- `--boundary-ratio 0.875`

```bash
export ASCEND_RT_VISIBLE_DEVICES=1,2,3,5
export VLLM_PLUGINS=ascend
export HF_HUB_OFFLINE=1
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni \
    --port 8091 \
    --usp 4 \
    --enable-layerwise-offload \
    --boundary-ratio 0.875
```

Notes:

* This model starts much more slowly than Qwen-Image. Wait for `/health` to return `200`.
* In this environment, the visible physical NPU ids are `1,2,3,5`, while the stage runtime uses local logical ids `0,1,2,3`.
* For online T2V requests, use `flow_shift=12.0` for `480p` and `flow_shift=5.0` for `720p`.

### 3.1.2 Current 4-card Validation Modes

Current `Wan2.2` validation uses a single 4-card server instance. The root script
[`eval_wan22.sh`](/root/vllm-omni/eval_wan22.sh) only launches the client benchmark. Start the
matching server manually first.

Baseline:

```bash
export ASCEND_RT_VISIBLE_DEVICES=1,2,3,5
export VLLM_PLUGINS=ascend
export HF_HUB_OFFLINE=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export VLLM_OMNI_ENABLE_DIFFUSION_SERVER_SCHEDULING=0
export VLLM_OMNI_ENABLE_DIFFUSION_PREEMPTION=0
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni \
    --port 8091 \
    --usp 4 \
    --enable-layerwise-offload \
    --boundary-ratio 0.875 \
    --super-p95-hardware-profile 910B2
```

Async-preemption / local `super_p95` validation:

```bash
export ASCEND_RT_VISIBLE_DEVICES=1,2,3,5
export VLLM_PLUGINS=ascend
export HF_HUB_OFFLINE=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export VLLM_OMNI_ENABLE_DIFFUSION_SERVER_SCHEDULING=1
export VLLM_OMNI_ENABLE_DIFFUSION_PREEMPTION=1
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni \
    --port 8091 \
    --usp 4 \
    --enable-layerwise-offload \
    --boundary-ratio 0.875 \
    --super-p95-hardware-profile 910B2 \
    --vae-use-slicing
```

Notes:

* `--vae-use-slicing` is currently required for the async-preemption path; without it, the `1280x720, 6 steps, 80 frames` request can still OOM in `vae.decode(...)`.
* `vae tiling` remains enabled by the current Wan registry path and should stay enabled.

## 3.2 Key Parameters

| Parameter             | Description              |
| --------------------- | ------------------------ |
| `--cfg-parallel-size` | CFG parallelism degree   |
| `--ulysses-degree`    | Ulysses parallel degree  |
| `--vae-patch-parallel-size`    | VAE parallel degree  |
| `--tensor-parallel-size` | Tensor parallelism degree |
| `--use-hsdp` | Enable Hybrid Sharded Data Parallel to shard model weights across GPUs |

Record these parameters when reporting performance results.

---

# 4. Benchmark Script

## 4.1 Benchmark Entry

```bash
python benchmarks/diffusion/diffusion_benchmark_serving.py \
    --backend v1/videos \
    --dataset <DATASET_NAME> \
    --task t2v \
    --num-prompts <N> \
    --max-concurrency <C> \
    --enable-negative-prompt \
    --random-request-config <CFG>
```

## 4.2 Key Benchmark Arguments

| Parameter              | Description                       |
| ---------------------- | --------------------------------- |
| `--backend`            | Serving backend (use `v1/videos`) |
| `--dataset`            | Dataset name (`random` or custom) |
| `--task`               | Task type (e.g., `t2v`)           |
| `--num-prompts`        | Total number of requests          |
| `--max-concurrency`    | Client-side concurrency           |
| `--random-request-config`| JSON string defining random request |

---

# 5. Dataset & Workload Settings

## 5.1 Recommended Evaluation Configurations

### Dataset A (480p)

* Dataset: `random`
* Task: t2v
* Concurrency: 1
* Mix Resolution
```
[
    {"width":854,"height":480,"num_inference_steps":3,"num_frames":80,"fps":16,"weight":1}
]
```
### Dataset B (720p)

* Dataset: `random`
* Task: t2v
* Concurrency: 1
* Mix Resolution
```
[
    {"width":1280,"height":720,"num_inference_steps":6,"num_frames":80,"fps":16,"weight":1}
]
```
### Dataset C (Mix Resolution)

* Dataset: `random`
* Task: t2v
* Concurrency: 1
* Mix Resolution
```
[
 {"width":854,"height":480,"num_inference_steps":3,"num_frames":80,"fps":16,"weight":0.15},
 {"width":854,"height":480,"num_inference_steps":4,"num_frames":120,"fps":24,"weight":0.25},
 {"width":1280,"height":720,"num_inference_steps":6,"num_frames":80,"fps":16,"weight":0.6}
]
```
---

## 5.2 Example Benchmark Command

```bash
python benchmarks/diffusion/diffusion_benchmark_serving.py \
    --backend v1/videos \
    --dataset random \
    --task t2v \
    --num-prompts 1 \
    --max-concurrency 1 \
    --enable-negative-prompt \
    --random-request-config '[
        {"width":854,"height":480,"num_inference_steps":3,"num_frames":80,"fps":16,"boundary_ratio":0.875,"flow_shift":12.0,"weight":1}
    ]'
```

---

## 5.3 Minimal Client Validation Command

The current `/v1/videos` serving path is synchronous: a single `POST /v1/videos` returns `200` with the generated video in `data[0].b64_json`.

```bash
env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
curl --noproxy "*" -sS -X POST http://127.0.0.1:8091/v1/videos \
  -F "prompt=A calm river flowing through a forest in spring" \
  -F "width=854" \
  -F "height=480" \
  -F "num_frames=80" \
  -F "fps=16" \
  -F "num_inference_steps=3" \
  -F "boundary_ratio=0.875" \
  -F "flow_shift=12.0" \
  -F "seed=0" \
  -o /tmp/wan22_req_480_3_80.json
```

The benchmark client also supports these request-level fields through `--random-request-config`.

---

# 6. Performance Metrics

The following metrics are collected during benchmarking:

| Metric             | Description                   | Unit    |
| ------------------ | ----------------------------- | ------- |
| Mean Latency        | Mean of latency       | seconds |
| P99 Latency        | P99 of latency             | seconds |

---

# 7. Performance Results

| Dataset Configuration | Max Concur. | CFG | Usp | Tp | Hsdp | VAE Parallel | Mean Latency (s) | P99 Latency (s) |
|-----------------------|-----|-----|-----|-----|----|--------------|------------------|------------------|
| Dataset A | 1 | 2 | 2 | 1 | On | 1          | 24.6766          | 24.6766          |
| Dataset A | 1 | 2 | 2 | 1 | On | 4          | 21.6810          | 21.6810          |
| Dataset B | 1 | 2 | 2 | 1 | On | 1          | 124.6639         | 124.6639          |
| Dataset B | 1 | 2 | 2 | 1 | On | 4          | 117.44          | 117.44          |
| Dataset C | 1 | 2 | 2 | 1 | On | 1          | 79.2175        | 124.2565 |
| Dataset C | 1 | 2 | 2 | 1 | On | 4          | 74.4977        | 117.710 |
---

## 7.1 Current NPU 910B2 Single-Request Timings

Measured on the 4-card NPU serving command above, with:

* model: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
* hardware: `910B2`
* serving: `--usp 4 --enable-layerwise-offload`
* client: `backend=v1/videos`, `num-prompts=1`, `max-concurrency=1`, `warmup-requests=0`

| Width | Height | Steps | Frames | FPS | flow_shift | Latency (s) |
|-------|--------|-------|--------|-----|------------|-------------|
| 854 | 480 | 3 | 80 | 16 | 12.0 | 43.61 |
| 854 | 480 | 4 | 120 | 24 | 12.0 | 90.17 |
| 1280 | 720 | 6 | 80 | 16 | 5.0 | 164.38 |

These are the current `910B2` service-time anchors for Wan2.2.
`910B3` remains unset for now and should be filled with measured data later.

For `super_p95` estimation, the current implementation uses:

* exact-match anchors for the three profiles above on `910B2`
* a `1280x720, 6 steps, 80 frames` normalized fallback for other Wan2.2 request shapes
* `910B3 -> 910B2` fallback, with a warning, until dedicated `910B3` measurements are available

## 7.2 Current 4-card 910B2 Async-Preemption Timings

Measured with:

* `VLLM_OMNI_ENABLE_DIFFUSION_SERVER_SCHEDULING=1`
* `VLLM_OMNI_ENABLE_DIFFUSION_PREEMPTION=1`
* `--vae-use-slicing`

| Width | Height | Steps | Frames | FPS | flow_shift | Latency (s) |
|-------|--------|-------|--------|-----|------------|-------------|
| 854 | 480 | 3 | 80 | 16 | 12.0 | 44.97 |
| 854 | 480 | 4 | 120 | 24 | 12.0 | 92.94 |
| 1280 | 720 | 6 | 80 | 16 | 5.0 | 170.51 |

Relative to the current baseline anchors, the async-preemption path with `--vae-use-slicing`
adds roughly:

* `+3.12%` on `854x480, 3 steps, 80 frames`
* `+3.07%` on `854x480, 4 steps, 120 frames`
* `+3.73%` on `1280x720, 6 steps, 80 frames`

---

# 8. Reproducibility Checklist

To ensure consistent and comparable benchmark results:

* Record GPU type
* Record parallel configuration
* Record benchmark parameters (resolution, concurrency, number of prompts)
* Ensure no background workload on GPUs during testing

---

This document serves as the official Wan2.2 serving performance reference under vLLM-Omni.
