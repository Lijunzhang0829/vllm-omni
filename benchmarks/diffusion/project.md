##  Qwen-Image

### Baseline (without preempt)

Server

```
python benchmarks/diffusion/launch_qwen_image_servers.py \
  --model Qwen/Qwen-Image \
  --num-servers 8 \
  --devices 0,1,2,3,4,5,6,7 \
  --host 0.0.0.0 \
  --dispatcher-host 0.0.0.0 \
  --base-port 8091 \
  --dispatcher-port 8090 \
  --disable-diffusion-preemption \
  --extra-server-args "--dtype bfloat16 --vae-use-slicing --vae-use-tiling"
```

Client

```
python benchmarks/diffusion/run_qwen_image_req_rate_sweep.py \
  --base-url http://127.0.0.1:8090 \
  --model Qwen/Qwen-Image \
  --backend vllm-omni \
  --dataset random \
  --task t2i \
  --num-prompts 500 \
  --seed 42   --random-request-seed 42 \
  --max-concurrency 1000 \
  --warmup-requests 1 \
  --enable-negative-prompt \
  --request-rates 0.1,0.25,0.4,0.55,0.7,0.85,1 \
  --output-dir benchmarks/diffusion/results/qwen_image_baseline_no_preempt \
  --random-request-config '[
    {"width":512,"height":512,"num_inference_steps":20,"weight":0.15},
    {"width":768,"height":768,"num_inference_steps":20,"weight":0.25},
    {"width":1024,"height":1024,"num_inference_steps":25,"weight":0.45},
    {"width":1536,"height":1536,"num_inference_steps":35,"weight":0.15}
  ]'
```

```
python benchmarks/diffusion/report_qwen_image_req_rate_sweep.py \
  --input-dir benchmarks/diffusion/results/qwen_image_baseline_no_preempt
```

### Ours (with preempt)

Server

```
python benchmarks/diffusion/launch_qwen_image_servers.py \
  --model Qwen/Qwen-Image \
  --num-servers 8 \
  --devices 0,1,2,3,4,5,6,7 \
  --host 0.0.0.0 \
  --dispatcher-host 0.0.0.0 \
  --base-port 8091 \
  --dispatcher-port 8090 \
  --extra-server-args "--dtype bfloat16 --vae-use-slicing --vae-use-tiling"
```

Client

```
python benchmarks/diffusion/run_qwen_image_req_rate_sweep.py \
  --base-url http://127.0.0.1:8090 \
  --model Qwen/Qwen-Image \
  --backend vllm-omni \
  --dataset random \
  --task t2i \
  --num-prompts 500 \
  --seed 42   --random-request-seed 42 \
  --max-concurrency 1000 \
  --warmup-requests 1 \
  --enable-negative-prompt \
  --request-rates 0.1,0.25,0.4,0.55,0.7,0.85,1 \
  --output-dir benchmarks/diffusion/results/qwen_image_preempt \
  --random-request-config '[
    {"width":512,"height":512,"num_inference_steps":20,"weight":0.15},
    {"width":768,"height":768,"num_inference_steps":20,"weight":0.25},
    {"width":1024,"height":1024,"num_inference_steps":25,"weight":0.45},
    {"width":1536,"height":1536,"num_inference_steps":35,"weight":0.15}
  ]'
```
