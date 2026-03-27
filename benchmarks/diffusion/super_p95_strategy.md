# Super P95 Strategy

本文档描述当前 Qwen-Image 在线调度策略的设计目标、公式、数据结构和实现语义。

旧名字：
- `delay_x`

新名字：
- `super_p95`

目标：
- 直接优化 `P95 E2E latency`
- 允许一小部分“重请求”被延后执行
- 正常请求优先保证尾延迟


## 1. 总体思路

`super_p95` 是一个两层策略：

1. `dispatcher` 层：
- 对新到达请求做静态分类
- 决定它是不是 `sacrificial`（放弃保护的请求）
- 决定它派发到哪张卡

2. `server` 层：
- 对本机已有请求做本地调度
- 正常请求按“预计延迟最大优先”
- `sacrificial` 请求整体尾排

一句话：

```text
dispatcher 负责分类和分卡；
server 负责本地优先级和抢占。
```


## 2. 请求耗时估计

策略依赖一个粗粒度 service-time 估计器。

当前实现支持按硬件 profile 选择不同 anchor：

- `910B2`
- `910B3`

如果 server / dispatcher 没有显式指定 hardware profile，会 warning 并默认回落到 `910B2`。

### 2.1 已知 profile

Qwen-Image 当前使用这 4 个 profile anchor。

`910B2`：

- `512x512x20x1 -> 8.60s`
- `768x768x20x1 -> 8.94s`
- `1024x1024x25x1 -> 14.22s`
- `1536x1536x35x1 -> 43.22s`

`910B3`：

- `512x512x20x1 -> 8.64s`
- `768x768x20x1 -> 8.64s`
- `1024x1024x25x1 -> 14.22s`
- `1536x1536x35x1 -> 49.34s`

当前单卡 NPU 实测值（2026-03-26，`Qwen/Qwen-Image`，`--vae-use-slicing --vae-use-tiling`，warm server，单请求 wall-clock）：

- `512x512x20x1 -> 8.93s`
- `768x768x20x1 -> 9.44s`
- `1024x1024x25x1 -> 13.41s`
- `1536x1536x35x1 -> 43.90s`

说明：

- 这组值是当前环境下的实测观测值，不是算法定义的一部分
- `weight` 字段在这次测量中被忽略，没有参与请求执行
- 对应更完整的测量记录见 `benchmarks/diffusion/qwen_image_npu_single_request_timings.md`

这里四元组含义是：

```text
(width, height, num_inference_steps, num_frames)
```

### 2.2 同分辨率估计公式

先按 hardware profile 选 anchor 表。

若分辨率命中以下 4 档之一：

- `512x512`
- `768x768`
- `1024x1024`
- `1536x1536`

则先预处理：

```text
per_pixel_step_ms
= anchor_latency_s * 1000 / (anchor_width * anchor_height * anchor_steps)
```

然后估计：

```text
estimated_service_s
= per_pixel_step_ms * width * height * num_inference_steps * num_frames / 1000
```

### 2.3 未知分辨率 fallback

若分辨率未知，则 fallback 到当前 hardware profile 下的 `1024x1024` 基准：

```text
estimated_service_s
= per_pixel_step_ms_1024
 * width * height * num_inference_steps * num_frames
 / 1000
```

### 2.4 剩余耗时估计

定义：

```text
total_cost     = width * height * total_steps * num_frames
remaining_cost = width * height * remaining_steps * num_frames
```

则：

```text
remaining_service_s
= total_service_s * (remaining_cost / total_cost)
```


## 3. Dispatcher 层：Super P95 分类

### 3.1 状态

dispatcher 维护以下轻量全局状态：

- `arrival_counter`
- `credits`
- `global_max_service_s`

含义：

- `arrival_counter`：到达请求计数
- `credits`：当前剩余的 sacrificial 名额
- `global_max_service_s`：历史上出现过的最大请求预计耗时

### 3.2 credits 更新

每收到 `quota_every` 个请求，增加 `quota_amount` 个名额：

```text
if arrival_counter % quota_every == 0:
    credits += quota_amount
```

### 3.3 global max 更新

新请求到达时更新：

```text
global_max_service_s = max(global_max_service_s, estimated_service_s)
```

### 3.4 sacrificial 判定

新请求 arrival 时直接决定它自己是不是 `sacrificial`：

```text
mark_sacrificial =
    (credits > 0)
    and (global_max_service_s > 0)
    and (estimated_service_s >= threshold_ratio * global_max_service_s)
```

若命中，则：

```text
credits -= 1
```

然后给该请求打 `sacrificial` 标记。

### 3.5 当前默认参数

典型参数是：

```text
quota_every = 20
quota_amount = 1
threshold_ratio = 0.8
```

解释：

- 每 20 个请求放出 1 个名额
- 只有“接近历史最大档”的请求才会吃到名额


## 4. Dispatcher 层：分卡规则

dispatcher 为每张卡维护两类负载：

- `normal_load_s`
- `sacrificial_load_s`

并维护一个 sacrificial 轻量负载系数：

```text
sacrificial_load_factor = α
```

当前推荐经验值：

```text
α = 0.1
```

### 4.1 主分数

当前主分卡分数是：

```text
weighted_total_load_s
= normal_load_s + α * sacrificial_load_s
```

dispatcher 选：

```text
argmin(weighted_total_load_s)
```

### 4.2 当前实现中的 tie-break

当前代码里 tie-break 还会附带：

- `inflight_normal_requests + α * inflight_sacrificial_requests`
- `latency_ema_s`
- `backend_name`

但主导项仍然是：

```text
normal_load_s + α * sacrificial_load_s
```

### 4.3 负载纠偏

dispatcher 自己会先按估计值增量更新负载。

backend 在请求返回时，会把本机当前权威负载通过 header 回传：

- `X-Super-P95-Normal-Load-S`
- `X-Super-P95-Sacrificial-Load-S`

dispatcher 收到响应后，用 server 的值覆盖本地账本，避免累计误差。


## 5. Server 层：本地调度目标

server 负责本地排序和抢占。

核心目标：

- 正常请求：优先保证 `P95`
- `sacrificial` 请求：整体延后

### 5.1 本地逻辑时间

server 不用墙钟时间做调度，而是维护一个逻辑时间：

```text
current_time_s
```

它通过已完成的 work 推进：

```text
elapsed_service_s
= total_service_s * (work_done / total_cost)

current_time_s += elapsed_service_s
```

这里：

- `work_done = before_remaining_cost - after_remaining_cost`

### 5.2 arrival time

每个请求第一次进入 policy 时，记录：

```text
arrival_time_s = current_time_s
```

注意这里是逻辑时间，不是墙钟时间。


## 6. Server 层：正常请求优先级公式

正常请求的核心分数是：

```text
predicted_latency_s
= current_time_s + remaining_service_s - arrival_time_s
```

等价写法：

```text
predicted_latency_s
= waited_service_s + remaining_service_s
```

其中：

```text
waited_service_s = current_time_s - arrival_time_s
```

server 总是优先执行 `predicted_latency_s` 最大的正常请求。

即：

```text
argmax(predicted_latency_s)
```


## 7. Server 层：sacrificial 请求规则

`sacrificial` 请求不参与正常请求竞争。

规则是：

1. 正常请求整体优先于 `sacrificial`
2. `sacrificial` 只有在 normal 队列为空时才会被取出
3. `sacrificial` 内部仍保留自己的顺序

### 7.1 当前组内规则

当前实现里，`sacrificial` 组内优先级由 `tail_penalty` 控制。

定义：

```text
tail_penalty = β
```

典型值：

```text
β = 100
```

当前 priority tuple 近似可写为：

正常请求：

```text
(0, -predicted_latency_s, arrival_seq, request_key)
```

`sacrificial` 请求：

```text
(1, predicted_latency_s / β, arrival_seq, request_key)
```

也就是：

- 第一位 `0/1` 保证 normal 永远先于 sacrificial
- 第二位只用于组内排序


## 8. 数据结构

当前 server 本地使用两个 heap：

- `normal_heap`
- `sacrificial_heap`

### 8.1 为什么能用 heap

因为当前 waiting 请求的排序键不依赖前缀和，不需要：

```text
prefix_wait_before_me
```

waiting 请求内部比较时，`current_time_s` 是公共项，所以只需比较：

```text
remaining_service_s - arrival_time_s
```

### 8.2 heap key

对 waiting 请求，当前 key 近似为：

正常 heap：

```text
key_normal = -(remaining_service_s - arrival_time_s)
```

`sacrificial` heap：

```text
key_sacrificial = +(remaining_service_s - arrival_time_s)
```

再配合：

- `arrival_seq`
- `heap_version`
- `request_key`

做 tie-break 和 lazy delete。


## 9. 抢占语义

### 9.1 事件点

server 的重排发生在这些事件点：

1. 请求到达
2. 当前请求完成
3. abort / shutdown / rpc 等外部事件

### 9.2 到达时抢占

arrival 时：

1. 新请求先进入 policy
2. policy 比较：
   - `current_req`
   - 当前最优 waiting 候选
3. 若候选优先级更高，则抢占当前请求

### 9.3 完成时切换

finish 时：

1. 当前请求出队
2. 先从 `normal_heap` 取
3. normal 空了再从 `sacrificial_heap` 取

### 9.4 当前 Qwen 实现状态

为了避免单请求性能回退，当前分支里：

- `QwenImage` 默认 **关闭** step-level preemption
- 即：
  - 仍保留 `super_p95` 的 arrival/finish 重排策略
  - 但不默认走 step-resume 的细粒度抢占路径

这个是当前代码的**实现状态**，不是算法定义本身。

如果要重写 `super_p95`，建议把 step-level preemption 做成：

- 明确 opt-in
- 或仅在高竞争场景开启

而不要作为默认路径。


## 10. 协议与元数据

### 10.1 Dispatcher -> Server

当前 dispatcher 会通过 header 给请求打标：

- `X-Super-P95-Sacrificial: 1`

server 收到后，把该请求视为 `sacrificial`。

### 10.2 Benchmark -> Dispatcher

benchmark client 会通过 header 提供请求 service hint：

- `X-Super-P95-Estimated-Service-S`

dispatcher 优先使用这个值，避免逐请求解析 JSON body。

### 10.3 Server -> Dispatcher

server 返回时会带回本机负载：

- `X-Super-P95-Normal-Load-S`
- `X-Super-P95-Sacrificial-Load-S`

dispatcher 用于纠偏本地估计。


## 11. 推荐的重写边界

如果要让同事重写，建议严格按下面边界拆分：

### Dispatcher 负责

- 估计新请求 `estimated_service_s`
- 维护：
  - `arrival_counter`
  - `credits`
  - `global_max_service_s`
- 决定该新请求是否 `sacrificial`
- 用 `weighted_total_load_s` 分卡
- 在响应返回后，用 server header 纠偏负载

### Server 负责

- 维护：
  - `arrival_time_s`
  - `current_time_s`
  - `normal_heap`
  - `sacrificial_heap`
- 正常请求按 `predicted_latency_s` 最大优先
- `sacrificial` 整体尾排
- 在 arrival / finish 时做重排


## 12. 当前实现中最容易踩坑的点

### 12.1 不要把 `super_p95` 错写成“从老队列里挑一个请求再改成 sacrificial”

这不是当前主策略。

当前主策略是：

```text
新请求 arrival 时，dispatcher 直接决定这个新请求自己是否 sacrificial。
```

### 12.2 不要把 `predicted_latency` 写成 `prefix_wait + remaining`

当前 heap 版主路径不需要：

```text
prefix_wait_before_me
```

因为一旦把前缀等待加进去，就很难继续用 heap 维护。

当前主路径是：

```text
predicted_latency_s = current_time_s + remaining_service_s - arrival_time_s
```

### 12.3 dispatcher 不要维护 server 的真实队列

dispatcher 只维护轻量负载账本，不托管 server 真实等待队列。


## 13. 一页纸公式汇总

### 耗时估计

```text
known profile:
  910B2:
    (512,512,20,1)   -> 8.60
    (768,768,20,1)   -> 8.94
    (1024,1024,25,1) -> 14.22
    (1536,1536,35,1) -> 43.22
  910B3:
    (512,512,20,1)   -> 8.64
    (768,768,20,1)   -> 8.64
    (1024,1024,25,1) -> 14.22
    (1536,1536,35,1) -> 49.34
```

```text
same-resolution estimate:
per_pixel_step_ms
= anchor_latency_s * 1000 / (anchor_width * anchor_height * anchor_steps)

estimated_service_s
= per_pixel_step_ms * width * height * steps * frames / 1000
```

```text
fallback:
estimated_service_s
= per_pixel_step_ms_1024 * width * height * steps * frames / 1000
```

### Dispatcher sacrificial 规则

```text
if arrival_counter % quota_every == 0:
    credits += quota_amount
```

```text
global_max_service_s = max(global_max_service_s, estimated_service_s)
```

```text
mark_sacrificial =
    credits > 0
    and estimated_service_s >= threshold_ratio * global_max_service_s
```

### Dispatcher 分卡

```text
weighted_total_load_s
= normal_load_s + α * sacrificial_load_s
```

```text
choose backend = argmin(weighted_total_load_s)
```

### Server 剩余耗时

```text
remaining_service_s
= total_service_s * remaining_cost / total_cost
```

### Server 逻辑时间推进

```text
current_time_s += total_service_s * work_done / total_cost
```

### Server 主优先级

```text
predicted_latency_s
= current_time_s + remaining_service_s - arrival_time_s
```

### Server 调度

```text
normal first
then sacrificial
```


## 14. 建议给重写同事的话

如果目标是“先验证策略本身，而不是优化到极限”，建议第一版只保留：

- dispatcher：
  - `credits`
  - `global_max_service_s`
  - `weighted_total_load_s`
- server：
  - `normal_heap`
  - `sacrificial_heap`
  - `predicted_latency_s = current_time + remaining - arrival`

先不要上：

- prefix wait
- 老队列选 sacrificial
- 复杂全局 regret
- 默认 step-level preemption

这样更容易复现策略本身，也更容易排查性能问题。


## 15. Qwen-Image P95 Sweep Commands

目标：

- 横轴：`request_rate = 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8`
- 纵轴：`latency_p95`
- 对比：`baseline` vs `super_p95`

说明：

- 当前 benchmark 的 `--request-rate` 实现是固定间隔发请求，不是 Poisson。
- 建议两组实验都使用相同 workload、相同 seed、相同 `random_request_seed`。
- 默认 sweep 配置使用 `500` 条请求。
- `baseline` 和 `super_p95` 都通过同一个 dispatcher 统一对外暴露 `8080`。
- `baseline` 关闭 sacrificial 分类，关闭 server 异步抢占，并关闭 backend local scheduler，尽量贴近 `v0.16.0` 的直通执行路径。
- `super_p95` 开启 sacrificial 分类，并开启 server 异步抢占。
- dispatcher managed launch 在检测到 `numactl` 和有效的 NPU NUMA node 后，会自动为每个 backend 做 `cpunodebind/membind` 绑定。

### 15.1 固定 workload

```bash
export QWEN_IMAGE_RANDOM4='[
  {"width":512,"height":512,"num_inference_steps":20,"weight":0.15},
  {"width":768,"height":768,"num_inference_steps":20,"weight":0.25},
  {"width":1024,"height":1024,"num_inference_steps":25,"weight":0.45},
  {"width":1536,"height":1536,"num_inference_steps":35,"weight":0.15}
]'
```

### 15.2 Baseline Server Command

```bash
env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python benchmarks/diffusion/super_p95_dispatcher.py \
  --host 127.0.0.1 \
  --port 8080 \
  --num-servers 2 \
  --device-ids 0,1 \
  --backend-hardware-profiles 910B2,910B2 \
  --model Qwen/Qwen-Image \
  --backend-start-port 8091 \
  --backend-args "--omni --vae-use-slicing --vae-use-tiling" \
  --backend-env VLLM_PLUGINS=ascend \
  --backend-env HF_HUB_OFFLINE=1 \
  --backend-env VLLM_OMNI_ENABLE_DIFFUSION_SERVER_SCHEDULING=0 \
  --backend-env VLLM_OMNI_ENABLE_DIFFUSION_PREEMPTION=0 \
  --request-timeout-s 10000 \
  --quota-every 20 \
  --quota-amount 0
```

### 15.3 Super-P95 Server Command

```bash
env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python benchmarks/diffusion/super_p95_dispatcher.py \
  --host 127.0.0.1 \
  --port 8080 \
  --num-servers 2 \
  --device-ids 0,1 \
  --backend-hardware-profiles 910B2,910B2 \
  --model Qwen/Qwen-Image \
  --backend-start-port 8091 \
  --backend-args "--omni --vae-use-slicing --vae-use-tiling" \
  --backend-env VLLM_PLUGINS=ascend \
  --backend-env HF_HUB_OFFLINE=1 \
  --backend-env VLLM_OMNI_ENABLE_DIFFUSION_PREEMPTION=1 \
  --request-timeout-s 10000 \
  --quota-every 20 \
  --quota-amount 1 \
  --threshold-ratio 0.8 \
  --sacrificial-load-factor 0.1
```

### 15.4 Baseline Benchmark Command

```bash
mkdir -p /tmp/super_p95_plot/baseline

for rate in 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
  env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  python benchmarks/diffusion/diffusion_benchmark_serving.py \
    --base-url http://127.0.0.1:8080 \
    --model Qwen/Qwen-Image \
    --backend vllm-omni \
    --dataset random \
    --task t2i \
    --num-prompts 500 \
    --max-concurrency 1000 \
    --request-rate "${rate}" \
    --warmup-requests 1 \
    --warmup-num-inference-steps 1 \
    --seed 0 \
    --random-request-seed 8 \
    --random-request-config "${QWEN_IMAGE_RANDOM4}" \
    --run-label baseline \
    --output-file "/tmp/super_p95_plot/baseline/rate_${rate}.json"
done
```

### 15.5 Super-P95 Benchmark Command

```bash
mkdir -p /tmp/super_p95_plot/super_p95

for rate in 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
  env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  python benchmarks/diffusion/diffusion_benchmark_serving.py \
    --base-url http://127.0.0.1:8080 \
    --model Qwen/Qwen-Image \
    --backend vllm-omni \
    --dataset random \
    --task t2i \
    --num-prompts 500 \
    --max-concurrency 1000 \
    --request-rate "${rate}" \
    --warmup-requests 1 \
    --warmup-num-inference-steps 1 \
    --seed 0 \
    --random-request-seed 8 \
    --random-request-config "${QWEN_IMAGE_RANDOM4}" \
    --run-label super_p95 \
    --output-file "/tmp/super_p95_plot/super_p95/rate_${rate}.json"
done
```

### 15.6 Smoke Test

先做命令连通性验证时，可以把 `num-prompts` 降到 `4`，只跑单个 `request_rate`，例如 `0.2`：

```bash
env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python benchmarks/diffusion/diffusion_benchmark_serving.py \
  --base-url http://127.0.0.1:8080 \
  --model Qwen/Qwen-Image \
  --backend vllm-omni \
  --dataset random \
  --task t2i \
  --num-prompts 4 \
  --max-concurrency 1000 \
  --request-rate 0.2 \
  --warmup-requests 1 \
  --warmup-num-inference-steps 1 \
  --seed 0 \
  --random-request-seed 8 \
  --random-request-config "${QWEN_IMAGE_RANDOM4}" \
  --run-label smoke \
  --output-file /tmp/super_p95_plot/smoke_rate_0.2.json
```

### 15.7 Output JSON Fields

当前 `--output-file` 会写出：

- 聚合指标：
  - `duration`
  - `completed_requests`
  - `failed_requests`
  - `throughput_qps`
  - `latency_mean`
  - `latency_median`
  - `latency_p95`
  - `latency_p99`
- 复现元数据：
  - `run_label`
  - `generated_at_utc`
  - `base_url`
  - `request_rate`
  - `traffic_schedule`
  - `num_prompts`
  - `max_concurrency`
  - `warmup_requests`
  - `warmup_num_inference_steps`
  - `seed`
  - `random_request_seed`
  - `random_request_config`
  - `backend`
  - `model`
  - `dataset`
  - `task`

绘图时直接读取每个 JSON 的：

```text
x = request_rate
y = latency_p95
group = run_label
```
