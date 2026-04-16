# Third-Party Test Commands

All commands assume the repo root is:

```bash
cd /vllm-workspace/vllm-omni
```

## 1. Environment Cleanup

```bash
pkill -f 'super_p95_dispatcher.py|vllm_omni.entrypoints.cli.main serve Qwen/Qwen-Image|vllm_omni.entrypoints.cli.main serve Wan-AI/Wan2.2-T2V-A14B-Diffusers' || true
rm -f /dev/shm/* 2>/dev/null || true
npu-smi info
```

Check:

- all 8 NPUs show `No running processes found`
- `/dev/shm` is empty or nearly empty

## 2. Qwen-Image

### 2.1 Qwen-Image Baseline Server

```bash
env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python3 benchmarks/diffusion/super_p95_dispatcher.py \
  --host 127.0.0.1 \
  --port 8080 \
  --num-servers 8 \
  --device-ids 0,1,2,3,4,5,6,7 \
  --model Qwen/Qwen-Image \
  --backend-start-port 8091 \
  --backend-hardware-profiles 910B3 \
  --backend-scheduler step_baseline \
  --backend-args=--omni \
  --backend-args=--vae-use-slicing \
  --backend-args=--vae-use-tiling
```

### 2.2 Qwen-Image Super P95 Server

```bash
env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python3 benchmarks/diffusion/super_p95_dispatcher.py \
  --host 127.0.0.1 \
  --port 8080 \
  --num-servers 8 \
  --device-ids 0,1,2,3,4,5,6,7 \
  --model Qwen/Qwen-Image \
  --backend-start-port 8091 \
  --backend-hardware-profiles 910B3 \
  --backend-scheduler super_p95_step \
  --quota-every 20 \
  --quota-amount 1 \
  --threshold-ratio 0.8 \
  --sacrificial-load-factor 0.1 \
  --backend-args=--omni \
  --backend-args=--vae-use-slicing \
  --backend-args=--vae-use-tiling
```

### 2.3 Qwen-Image Benchmark

Baseline:
```bash
BASE_URL=http://127.0.0.1:8080 \
NUM_PROMPTS=500 \
MAX_CONCURRENCY=1000 \
REQUEST_RATES=0.8 \
SEED=0 \
RANDOM_REQUEST_SEED=8 \
ARRIVAL_SEED=8 \
bash eval_qwen_image.sh baseline
```

Super P95:
```bash
BASE_URL=http://127.0.0.1:8080 \
NUM_PROMPTS=500 \
MAX_CONCURRENCY=1000 \
REQUEST_RATES=0.8 \
SEED=0 \
RANDOM_REQUEST_SEED=8 \
ARRIVAL_SEED=8 \
bash eval_qwen_image.sh super_p95
```

## 3. Wan2.2

### 3.1 Wan2.2 Baseline Server

```bash
env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python3 benchmarks/diffusion/super_p95_dispatcher.py \
  --host 127.0.0.1 \
  --port 8080 \
  --num-servers 1 \
  --device-ids 0,1,2,3,4,5,6,7 \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --backend-start-port 8091 \
  --backend-hardware-profiles 910B3 \
  --backend-scheduler step_baseline \
  --backend-log-dir /tmp/wan22_t2v_baseline_usp8 \
  --backend-env VLLM_OMNI_MASTER_PORT=30191 \
  --request-timeout-s 1000000 \
  --backend-health-timeout-s 1800 \
  --backend-health-poll-interval-s 10 \
  --backend-args=--omni \
  --backend-args=--usp \
  --backend-args=8 \
  --backend-args=--enable-layerwise-offload \
  --backend-args=--boundary-ratio \
  --backend-args=0.875 \
  --backend-args=--flow-shift \
  --backend-args=5.0 \
  --backend-args=--vae-use-slicing \
  --backend-args=--vae-use-tiling
```

### 3.2 Wan2.2 Super P95 Server

```bash
bash benchmarks/diffusion/run_wan22_super_p95_dispatcher.sh
```

### 3.3 Wan2.2 Benchmark

Baseline:

```bash
env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
  --base-url http://127.0.0.1:8080 \
  --backend v1/videos \
  --dataset random \
  --task t2v \
  --num-prompts 160 \
  --max-concurrency 160 \
  --request-rate 0.02 \
  --enable-negative-prompt \
  --seed 42 \
  --random-request-seed 42 \
  --arrival-seed 42 \
  --random-request-config '[
    {"width":854,"height":480,"num_inference_steps":3,"num_frames":80,"fps":16,"weight":0.15},
    {"width":854,"height":480,"num_inference_steps":4,"num_frames":120,"fps":24,"weight":0.25},
    {"width":1280,"height":720,"num_inference_steps":6,"num_frames":80,"fps":16,"weight":0.6}
  ]' \
  --output-file /tmp/wan22_t2v_baseline_usp8/run_160req_mix3_rps0.02_seed42.json
```

Super P95:

```bash
env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
  --base-url http://127.0.0.1:8080 \
  --backend v1/videos \
  --dataset random \
  --task t2v \
  --num-prompts 160 \
  --max-concurrency 160 \
  --request-rate 0.02 \
  --enable-negative-prompt \
  --seed 42 \
  --random-request-seed 42 \
  --arrival-seed 42 \
  --random-request-config '[
    {"width":854,"height":480,"num_inference_steps":3,"num_frames":80,"fps":16,"weight":0.15},
    {"width":854,"height":480,"num_inference_steps":4,"num_frames":120,"fps":24,"weight":0.25},
    {"width":1280,"height":720,"num_inference_steps":6,"num_frames":80,"fps":16,"weight":0.6}
  ]' \
  --output-file /tmp/wan22_t2v_super_p95_usp8/run_160req_mix3_rps0.02_seed42.json
```

## 4. Readiness Check

Wan2.2 is ready when the backend log contains:

```text
Starting vLLM API server (pure diffusion mode)
Application startup complete.
```

Qwen-Image is ready when:

- `/health` returns `200`
- backend startup log reaches `Application startup complete`
