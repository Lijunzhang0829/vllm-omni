# Super P95 Strategy

本文档描述当前 diffusion `super_p95` 系统的真实实现，而不是早期设想版。

适用范围：
- `Qwen-Image`
- `Wan2.2`

目标：
- 优化 `P95 E2E latency`
- 允许一小部分重请求被“牺牲”
- 在高压区优先保护大多数请求的尾延迟


## 1. 系统结构

当前实现是两层：

1. `dispatcher`
- 负责估计新请求 service time
- 负责把新请求标成 `sacrificial` 或 normal
- 负责选择 backend
- 维护轻量负载账本，并用 backend 回传 header 纠偏

2. `server`
- 负责本机 waiting queue 的排序
- 负责 arrival / finish 时的重排
- 在支持异步抢占的模型上，负责 arrival-time preemption

一句话：

```text
dispatcher 做分类和分卡；
server 做本地优先级和抢占。
```


## 2. 当前实现状态

当前实现已经和早期文档有几处重要偏移，下面这些以代码为准：

- dispatcher 启动参数是 `--num-servers`
  - 不是旧文档里的 `--managed-backends`
- dispatcher 对所有请求都按
  - `normal_load_s + α * sacrificial_load_s`
  做主分卡
  - 不是“normal 请求只看 normal load，sacrificial 才看加权 load”
- server 里：
  - normal 组内是“更大的 predicted latency 先跑”
  - sacrificial 组内是“更小的 tail key 先跑”
- `Wan2.2` 的 `910B2` / `910B3` anchor 现在都使用同一组 8 卡 `usp=8` 实测值
- `Wan2.2` preemption-capable path 里，non-expand timestep 构造已经修成 baseline 同款
  - 这是近期性能修复的关键变更


## 3. 请求耗时估计

策略依赖一个粗粒度 service-time 估计器。

当前支持的 hardware profile：
- `910B2`
- `910B3`

如果 dispatcher / server 没有显式指定 profile：
- 会 warning
- 并默认回落到 `910B2`


### 3.1 Qwen-Image anchor

四元组含义：

```text
(width, height, num_inference_steps, num_frames)
```

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

估计公式：

```text
per_pixel_step_ms
= anchor_latency_s * 1000 / (anchor_width * anchor_height * anchor_steps)

estimated_service_s
= per_pixel_step_ms * width * height * num_inference_steps * num_frames / 1000
```

未知分辨率 fallback 到当前 profile 下的 `1024x1024x25` anchor。


### 3.2 Wan2.2 anchor

当前 `Wan2.2` 使用的视频 anchor 是：

`910B2`：
- `854x480x3x80 -> 38.07s`
- `854x480x4x120 -> 71.34s`
- `1280x720x6x80 -> 119.71s`

`910B3`：
- `854x480x3x80 -> 38.07s`
- `854x480x4x120 -> 71.34s`
- `1280x720x6x80 -> 119.71s`

说明：
- 这不是说两种硬件天然相同
- 而是当前本地 `super_p95` 估时为了和最近的 8 卡 `usp=8` serving 实测对齐，临时共用同一组 anchor

精确命中时直接用 exact match。

未知组合时 fallback：

```text
per_pixel_frame_step_ms
= anchor_latency_s * 1000 / (width * height * steps * frames)

estimated_service_s
= per_pixel_frame_step_ms * width * height * steps * frames / 1000
```

当前 fallback 基准是：

```text
(1280, 720, 6, 80)
```


### 3.3 剩余耗时估计

server 侧 remaining service 仍按工作量比例缩放：

```text
total_cost     = width * height * total_steps * num_frames
remaining_cost = width * height * remaining_steps * num_frames

remaining_service_s
= total_service_s * remaining_cost / total_cost
```


## 4. Dispatcher：sacrificial 分类

dispatcher 维护：
- `arrival_counter`
- `credits`
- `global_max_service_s`


### 4.1 credits 规则

```text
if arrival_counter % quota_every == 0:
    credits += quota_amount
```


### 4.2 global max 更新

```text
global_max_service_s = max(global_max_service_s, estimated_service_s)
```


### 4.3 sacrificial 判定

新请求 arrival 时，dispatcher 直接决定该请求自己是不是 `sacrificial`：

```text
mark_sacrificial =
    (credits > 0)
    and (global_max_service_s > 0)
    and (estimated_service_s >= threshold_ratio * global_max_service_s)
```

若命中：

```text
credits -= 1
```

当前常用参数：

```text
quota_every = 20
quota_amount = 1
threshold_ratio = 0.8
```


## 5. Dispatcher：分卡规则

每个 backend 维护：
- `normal_load_s`
- `sacrificial_load_s`
- `inflight_normal_requests`
- `inflight_sacrificial_requests`
- `latency_ema_s`

主分数是：

```text
weighted_total_load_s
= normal_load_s + α * sacrificial_load_s
```

当前经验值：

```text
α = 0.1
```

dispatcher 选择：

```text
argmin(score_tuple)
```

当前 `score_tuple(alpha)` 实现是：

```text
(
  normal_load_s + α * sacrificial_load_s,
  inflight_normal_requests + α * inflight_sacrificial_requests,
  latency_ema_s,
  normal_load_s,
  backend_name,
)
```

注意：
- 这里所有请求都走同一套主分数
- normal 请求并不会绕开 `α * sacrificial_load_s`


### 5.1 响应反馈

server 在响应头回传：
- `X-Super-P95-Normal-Load-S`
- `X-Super-P95-Sacrificial-Load-S`

dispatcher 收到后：
- 用 server 的权威值覆盖本地账本
- 如果 header 缺失，就退回到“减掉本次 estimated load”的 fallback 逻辑


## 6. Server：本地调度目标

server 维护逻辑时间：

```text
current_time_s
```

它不是墙钟时间，而是“已完成 service work”驱动的逻辑时间。

请求第一次进入 policy 时，记录：

```text
arrival_time_s = current_time_s
```


### 6.1 正常请求优先级

normal 请求的核心分数：

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

server 选择：

```text
argmax(predicted_latency_s)
```

在 heap 里等价成：

```text
key_normal = -(remaining_service_s - arrival_time_s)
```


### 6.2 sacrificial 规则

规则是：

1. normal 永远先于 sacrificial
2. 只有 normal 为空时，才会从 sacrificial 里取请求
3. sacrificial 组内按自己的 key 排序

当前 sacrificial heap key 是：

```text
key_sacrificial = +(remaining_service_s - arrival_time_s)
```

所以组内语义是：

```text
更小的 tail key 先跑
```

这一点最近修过，不能再写反。


### 6.3 当前数据结构

当前 server 维护两个 heap：

- `normal_heap`
- `sacrificial_heap`

实现上对应：
- `_normal_pending`
- `_sacrificial_pending`


## 7. Server：抢占语义

当前重排事件点：

1. 请求到达
2. 当前请求完成
3. abort / shutdown 等外部事件

arrival 时：
- 新请求先进入 policy
- 若 waiting 中的最优候选优先级高于 incumbent，则可以请求抢占

finish 时：
- 当前请求移除
- 先从 normal 取
- normal 空了再从 sacrificial 取


### 7.1 当前实现里的 preemption

当前 `Qwen-Image` 和 `Wan2.2` 都支持 step-boundary async preemption。

也就是说：
- 不是任意时刻硬中断
- 而是在 denoise step 边界检查 `interrupt`
- 触发后返回 `finished=False` 和恢复状态

近期 `Wan2.2` 的性能问题说明：
- preemption-capable path 即使在“没真正发生抢占”时，也可能带来固定额外成本
- 因此实现上必须特别注意“未抢占 fast path”和“真正 yield/resume path”之间的开销差异


## 8. 协议与元数据

### 8.1 Dispatcher -> Server

dispatcher 会通过 header 传：

- `X-Super-P95-Estimated-Service-S`
- `X-Super-P95-Sacrificial: 1`（仅 sacrificial 请求）

server 通过 header / `extra_args` 读取这些元数据。


### 8.2 Server -> Dispatcher

server 返回：

- `X-Super-P95-Normal-Load-S`
- `X-Super-P95-Sacrificial-Load-S`

dispatcher 用于负载纠偏。


## 9. 启动与部署

### 9.1 当前推荐拓扑

对外统一走 dispatcher：

```text
:8080  ->  managed backends :8091, :8092, ...
```

即使只有一个 backend，也建议通过 dispatcher 走一遍，避免：
- baseline / super 拓扑不一致
- sacrificial metadata 不生效


### 9.2 当前 managed launch 参数

当前 dispatcher 支持的是：

```text
--num-servers
```

不是旧文档里的：

```text
--managed-backends
```

`backend` 额外参数要用重复的：

```text
--backend-args=...
```

不是单个拼接字符串后再二次拆分的旧假设。


### 9.3 NUMA 行为

managed launch 会根据 NPU PCI Bus-Id 推断 NUMA node，并尝试：

```text
numactl --cpunodebind <node>
```

当前默认行为：
- 单设备 backend：保留历史 `membind`
- 多设备 `--usp > 1` backend：默认关闭 `membind`

原因：
- `Wan2.2 usp=8` 启动时，若强行 `--membind` 到单个 NUMA node，容易出现宿主机 OOM killer
- 这是最近踩到并修过的坑


## 10. 当前最容易踩坑的点

### 10.1 sacrificial 组内顺序不要写反

当前正确语义是：

```text
key_sacrificial = +(remaining_service_s - arrival_time_s)
```

即：
- sacrificial 组内“小的先跑”

不是：
- 和 normal 一样“大 predicted latency 先跑”


### 10.2 dispatcher 分卡主分数对所有请求都一样

当前实现不是：
- normal 只看 `normal_load`
- sacrificial 才看 `normal + α * sacrificial`

而是：

```text
所有请求都按 normal_load_s + α * sacrificial_load_s 选 backend
```


### 10.3 `Wan2.2` 不经过 dispatcher 跑出的 super 结果不可信

如果 client 直接打 backend：
- sacrificial 分类不会真正生效
- 只能得到 local scheduling 的局部效果

所以近期 `Wan2.2` 的 baseline / super 对比，都要求统一走：

```text
dispatcher :8080
```


### 10.4 本地 localhost 调试要关代理

若环境里有代理，health 检查和本地 benchmark 很容易被错误路由到代理地址。

建议统一带：

```bash
NO_PROXY=127.0.0.1,localhost
no_proxy=127.0.0.1,localhost
```


### 10.5 `Wan2.2` 的 8 卡估时 anchor 已经更新

不要再用旧的：
- 4 卡
- 910B2-only

那套 anchor 去解释当前 8 卡 `usp=8` 结果。

当前 server / dispatcher 两边都已经对齐到：
- 8 卡 `usp=8`
- 910B2/910B3 同一组本地实测值


### 10.6 `Wan2.2` preemption path 里 non-expand timestep 构造不能再退回旧实现

近期最重要的性能修复就是：
- preemption-capable path 在普通 T2V/non-expand 分支里
- 不再每个 denoise step 现造 `mask -> flatten -> expand` 的 device-side timestep
- 改回和 baseline 一致的：

```text
timestep = t.expand(batch)
```

这个修复显著收敛了 `super_p95` 和 baseline 的系统性性能差。


## 11. 当前 benchmark 结论摘要

### 11.1 Qwen-Image

当前观察：
- 低压区：baseline 与 `super_p95` 接近，甚至 baseline 略好
- 高压区：`super_p95` 对 `P95` 有明显收益

### 11.2 Wan2.2

近期修复后，当前观察是：
- `0.012`：还没明显打满，baseline 略好或接近
- `0.020`：高压下 `super_p95` 开始体现 `P95` 收益

也就是说：
- `Wan2.2` 的 `super_p95` 价值主要体现在高压区
- 没打满时，baseline 的简单路径有时反而更占便宜


## 12. 一页纸公式汇总

### 耗时估计

```text
Qwen:
estimated_service_s
= per_pixel_step_ms * width * height * steps * frames / 1000
```

```text
Wan:
estimated_service_s
= per_pixel_frame_step_ms * width * height * steps * frames / 1000
```

### Dispatcher sacrificial

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
    and global_max_service_s > 0
    and estimated_service_s >= threshold_ratio * global_max_service_s
```

### Dispatcher 分卡

```text
score_backend
= normal_load_s + α * sacrificial_load_s
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

### Server normal key

```text
key_normal = -(remaining_service_s - arrival_time_s)
```

### Server sacrificial key

```text
key_sacrificial = +(remaining_service_s - arrival_time_s)
```


## 13. 推荐给后续维护者的话

如果你要继续改 `super_p95`，优先遵守这三条：

1. 先保证 dispatcher / server / benchmark 三层拓扑一致，再谈性能。
2. 先检查是不是实现和文档不一致，再怀疑策略本身。
3. 对 `Wan2.2`，优先怀疑 preemption-capable denoise path 的实现开销，而不是先怀疑调度公式。


## 14. 最新实验记录

这一节记录最近一次已经对齐实现后的 benchmark 结果，避免后续重复踩坑。


### 14.1 实验 setup

#### Qwen-Image

- 模型：
  - `Qwen/Qwen-Image`
- server 拓扑：
  - dispatcher 对外 `:8080`
  - 多个 managed backend
- workload：
  - `QWEN_IMAGE_RANDOM4`
  - `500` requests
- 对比：
  - `baseline`
  - `super_p95`
- 指标：
  - 重点看 `latency_p95`

#### Wan2.2

- 模型：
  - `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- server 拓扑：
  - dispatcher `:8080`
  - 单个 managed backend `:8091`
- backend 配置：
  - `--usp 8`
  - `--enable-layerwise-offload`
  - `--boundary-ratio 0.875`
  - `--vae-use-slicing`
  - `--vae-use-tiling`
- workload：
  - `WAN22_RANDOM3`
  - `160` requests
  - 固定 `seed=0`
  - 固定 `random_request_seed=8`
- 对比：
  - `baseline`
  - `super_p95`
- rate：
  - `0.012`
  - `0.020`

#### 统一注意事项

- baseline 和 super 都统一通过 dispatcher 暴露 `:8080`
- localhost 调试时必须带：

```bash
NO_PROXY=127.0.0.1,localhost
no_proxy=127.0.0.1,localhost
```

- 启动前先检查：

```bash
npu-smi info
```

确认没有残留运行进程


### 14.2 实验结果

#### Qwen-Image：baseline vs super_p95

| Rate | Baseline P95 (s) | Super P95 (s) | Speedup |
|---|---:|---:|---:|
| 0.2 | `49.79` | `49.35` | `1.009x` |
| 0.3 | `49.57` | `49.64` | `0.999x` |
| 0.4 | `55.14` | `56.83` | `0.970x` |
| 0.5 | `141.63` | `95.30` | `1.486x` |
| 0.6 | `276.89` | `181.82` | `1.523x` |
| 0.7 | `391.25` | `291.50` | `1.342x` |
| 0.8 | `461.92` | `374.87` | `1.232x` |

#### Wan2.2：baseline vs super_p95

| Rate | Baseline Duration | Super Duration | Baseline P95 | Super P95 | Speedup |
|---|---:|---:|---:|---:|---:|
| 0.012 | `13485.18s` | `13485.96s` | `190.11s` | `195.32s` | `0.973x` |
| 0.020 | `12641.97s` | `12609.25s` | `4343.80s` | `3890.77s` | `1.116x` |

补充指标：

`Wan2.2 @ 0.012`
- baseline:
  - `mean = 126.147s`
  - `median = 124.569s`
  - `P99 = 204.776s`
- super:
  - `mean = 126.387s`
  - `median = 119.364s`
  - `P99 = 343.511s`

`Wan2.2 @ 0.020`
- baseline:
  - `mean = 2288.197s`
  - `median = 2311.784s`
  - `P99 = 4605.198s`
- super:
  - `mean = 2205.855s`
  - `median = 1981.146s`
  - `P99 = 9906.762s`


### 14.3 实验分析

#### Qwen-Image

当前结论很稳定：

- 低压区：
  - `0.2 ~ 0.4`
  - baseline 与 `super_p95` 接近
  - 有时 baseline 略好
- 高压区：
  - `0.5 ~ 0.8`
  - `super_p95` 对 `P95` 有明显收益

这说明：
- `Qwen-Image` 上，`super_p95` 的主要价值在高压区
- 低压时系统还没形成明显 waiting set，策略收益有限

#### Wan2.2

当前结论是：

- `0.012`
  - 还没明显打满
  - baseline 与 `super_p95` 非常接近
  - baseline 的 `P95` 略好
- `0.020`
  - 明显进入高压区
  - `super_p95` 开始体现 `P95` 收益
  - 相比 baseline 改善约 `453s`

也就是说：

- `Wan2.2` 的 `super_p95` 不是在所有负载下都占优
- 它的价值主要体现在高压区

#### Wan2.2 近期关键性能修复

最近这一轮定位到并修复的主问题是：

- `Wan2.2` preemption-capable denoise path 在普通 non-expand T2V 分支里
- 每个 step 都在 device 上现造一套不必要的 `mask -> flatten -> expand` timestep
- baseline 路径并不需要这套额外工作

修复后：
- `super_p95` 的系统性慢问题已经大幅收敛
- 当前剩余的主要现象已经不是“整体都慢”，而更多是高压下的 tail tradeoff

#### 当前还需记住的结论

- `super_p95` 是策略名
- `P95` 是指标名
- 讨论“好不好”时，先明确是在说：
  - 策略
  - 还是指标

对当前项目来说，优先级最高的指标仍然是：

```text
latency_p95
```

不是：
- `duration`
- `mean`
- `P99`

这些指标仍然值得看，但不应该盖过主目标。
