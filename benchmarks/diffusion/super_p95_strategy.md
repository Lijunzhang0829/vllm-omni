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

### 2.1 已知 profile

Qwen-Image 当前使用这 4 个锚点：

- `512x512x20x1 -> 22.35s`
- `768x768x20x1 -> 20.62s`
- `1024x1024x25x1 -> 33.90s`
- `1536x1536x35x1 -> 102.66s`

这里四元组含义是：

```text
(width, height, num_inference_steps, num_frames)
```

### 2.2 同分辨率估计公式

若分辨率命中以下 4 档之一：

- `512x512`
- `768x768`
- `1024x1024`
- `1536x1536`

则使用该分辨率锚点做缩放：

```text
estimated_service_s
= anchor_latency_s
 * (num_inference_steps / anchor_steps)
 * (num_frames / anchor_frames)
```

### 2.3 未知分辨率 fallback

若分辨率未知，则 fallback 到总 work 线性公式：

```text
estimated_service_s
= 33.90
 * (width * height * num_inference_steps * num_frames)
 / (1024 * 1024 * 25 * 1)
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

- `X-DelayX-Normal-Load-S`
- `X-DelayX-Sacrificial-Load-S`

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

- `X-DelayX-Sacrificial: 1`

server 收到后，把该请求视为 `sacrificial`。

### 10.2 Benchmark -> Dispatcher

benchmark client 会通过 header 提供请求 service hint：

- `X-DelayX-Estimated-Service-S`

dispatcher 优先使用这个值，避免逐请求解析 JSON body。

### 10.3 Server -> Dispatcher

server 返回时会带回本机负载：

- `X-DelayX-Normal-Load-S`
- `X-DelayX-Sacrificial-Load-S`

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
  (512,512,20,1)   -> 22.35
  (768,768,20,1)   -> 20.62
  (1024,1024,25,1) -> 33.90
  (1536,1536,35,1) -> 102.66
```

```text
same-resolution estimate:
estimated_service_s
= anchor_latency_s
 * (steps / anchor_steps)
 * (frames / anchor_frames)
```

```text
fallback:
estimated_service_s
= 33.90 * (width * height * steps * frames) / (1024 * 1024 * 25)
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
