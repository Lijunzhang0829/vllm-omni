# Super P95 Technical Plan

## Outline

1. Background and target
2. Current implementation architecture
3. Two-level scheduling model
4. Inter-card scheduling formulas
5. In-card scheduling formulas
6. Service-time estimation formulas
7. Request metadata and load feedback protocol
8. Startup and benchmark procedure
9. Known limitations and risks
10. Recommended rewrite plan
11. Validation plan and acceptance criteria

## Core Points

- `super_p95` is a diffusion-serving latency policy aimed at reducing tail
  latency, especially P95, by routing and scheduling high-cost requests as
  sacrificial work.
- There are two schedulers. The inter-card scheduler lives in the dispatcher and
  chooses which card/backend receives each request. The in-card scheduler lives
  inside each backend and chooses which local request runs at each diffusion
  step boundary.
- Inter-card scheduling is responsible for global admission decisions: request
  service-time estimation, sacrificial classification, backend scoring, backend
  selection, and forwarding metadata through HTTP headers.
- In-card scheduling is responsible for local execution order: normal requests
  keep FIFO ordering, sacrificial requests are deprioritized, and a normal
  request may preempt a running sacrificial request at a step boundary.
- The current branch already has a `v0.18`-style step scheduler
  (`SuperP95StepScheduler`), but the rewrite notes warn that the broader branch
  still mixed older request-mode recovery ideas with the newer runtime shape.
- The next robust implementation should start from a clean `v0.18.0` base and
  build `super_p95` on top of native stepwise diffusion execution.
- Benchmark results are only meaningful after environment cleanup and health
  checks. Dirty NPU state has previously caused false regressions.

## 1. Background And Target

Diffusion image and video requests have highly variable service times. Large
resolution, more denoising steps, and more frames can create long-running
requests that inflate P95 latency for normal requests arriving behind them.

`super_p95` treats part of the long-request population as a controlled cost
sink. The policy allows a limited number of expensive requests to be marked as
sacrificial. Sacrificial requests are still served, but backend scheduling gives
normal requests priority over them. This trades a moderate throughput cost and
longer latency for sacrificial requests for lower P95 latency for the main
request population.

The target behavior is:

- preserve correctness for all requests
- improve median and P95 latency for normal workload mixes
- keep throughput degradation bounded and measurable
- avoid restarting completed diffusion steps after preemption
- keep baseline behavior unchanged when `super_p95` is disabled

## 2. Current Implementation Architecture

The implementation has four main layers.

### Dispatcher

File:

- `benchmarks/diffusion/super_p95_dispatcher.py`

Responsibilities:

- start and monitor managed backend processes when requested
- expose a single frontend HTTP endpoint
- estimate per-request service time
- maintain dispatcher-side backend load state
- decide whether a request is normal or sacrificial
- choose a backend
- forward metadata to backend through `X-Super-P95-*` headers
- consume backend load feedback headers after each response

The dispatcher supports these request paths:

- `/v1/chat/completions`
- `/v1/images/generations`
- `/v1/videos`

### Shared Protocol And Estimation Helpers

File:

- `vllm_omni/diffusion/super_p95.py`

Responsibilities:

- define request headers
- define internal `extra_args` keys
- parse request metadata from headers
- build backend load feedback headers
- estimate service time for Qwen-Image and Wan2.2-style workloads
- normalize hardware profiles such as `910B2` and `910B3`

### Backend API Integration

File:

- `vllm_omni/entrypoints/openai/api_server.py`

Responsibilities:

- read `super_p95` headers from incoming requests
- write sacrificial and estimated-service metadata into sampling
  `extra_args`
- expose scheduler load snapshots back to dispatcher through response headers

### Backend Scheduler

File:

- `vllm_omni/diffusion/sched/super_p95_step_scheduler.py`

Responsibilities:

- keep separate normal and sacrificial queues
- preserve FIFO ordering for normal requests
- prioritize newer sacrificial requests within the sacrificial queue
- preempt sacrificial work when a normal request is available
- track remaining estimated load for response feedback
- operate on the stepwise execution path

## 3. Two-Level Scheduling Model

`super_p95` uses two independent but cooperating schedulers.

### Inter-Card Scheduler

Location:

- `benchmarks/diffusion/super_p95_dispatcher.py`

The inter-card scheduler runs once per incoming user request. It decides:

- estimated service time of the request
- whether the request is normal or sacrificial
- which backend/card should receive the request
- what metadata should be attached to the forwarded request

It maintains global state across all managed backends:

- arrival counter
- sacrificial credits
- global maximum observed service time
- per-backend normal load
- per-backend sacrificial load
- per-backend inflight request counts
- per-backend latency EMA

### In-Card Scheduler

Location:

- `vllm_omni/diffusion/sched/super_p95_step_scheduler.py`

The in-card scheduler runs inside a backend/card. It decides:

- which local request should run next
- whether a running sacrificial request should yield to a normal request
- how much estimated remaining work is currently queued on the card
- what load snapshot should be returned to the dispatcher

It maintains local state:

- normal pending heap
- sacrificial pending heap
- currently running request
- per-request total steps
- per-request completed steps
- per-request estimated service time

## 4. Inter-Card Scheduling Formulas

The dispatcher increments an arrival counter for every request. Every
`quota_every` arrivals, it adds `quota_amount` sacrificial credits. A request can
be marked sacrificial only when:

- at least one credit is available
- the request estimated service time is near the largest service time observed
  so far
- specifically, `estimated_service_s >= threshold_ratio * global_max_service_s`

If a request is marked sacrificial, one credit is consumed. Otherwise it remains
normal.

Backend selection uses the dispatcher-side load model. Each backend tracks:

- estimated normal load
- estimated sacrificial load
- inflight normal request count
- inflight sacrificial request count
- latency EMA

The selection score discounts sacrificial load by
`sacrificial_load_factor`. This prevents sacrificial work from dominating load
balancing while still reflecting that a backend is not idle.

On backend response, the dispatcher prefers authoritative scheduler load
headers. If those headers are absent, it falls back to subtracting the original
estimated service time for the completed request.

### Symbols

For request `r` and backend/card `b`:

- `A`: global arrival counter
- `Q`: current sacrificial credit count
- `E_r`: estimated service time of request `r`, in seconds
- `G`: global maximum observed request service time
- `T`: sacrificial threshold ratio, configured by `threshold_ratio`
- `K`: quota interval, configured by `quota_every`
- `M`: quota refill amount, configured by `quota_amount`
- `alpha`: sacrificial load factor, configured by `sacrificial_load_factor`
- `N_b`: backend `b` normal remaining load, in seconds
- `S_b`: backend `b` sacrificial remaining load, in seconds
- `I^N_b`: backend `b` inflight normal request count
- `I^S_b`: backend `b` inflight sacrificial request count
- `L_b`: backend `b` latency EMA, in seconds

### Arrival Counter

For every incoming request:

```text
A <- A + 1
```

### Quota Refill

If `A` is divisible by `K`:

```text
Q <- Q + M
```

Otherwise:

```text
Q <- Q
```

### Global Maximum Service Time

After estimating the new request:

```text
G <- max(G, E_r)
```

### Sacrificial Classification

A request is sacrificial only when credit is available and the request is close
to the largest observed request:

```text
is_sacrificial(r) = (Q > 0) and (G > 0) and (E_r >= T * G)
```

Credit consumption:

```text
if is_sacrificial(r):
    Q <- Q - 1
```

If the condition is false, `r` is normal and no credit is consumed.

### Backend Weighted Load

The dispatcher discounts sacrificial load by `alpha`:

```text
W_b = N_b + alpha * S_b
```

### Backend Inflight Score

The inflight score uses the same sacrificial discount:

```text
F_b = I^N_b + alpha * I^S_b
```

### Backend Selection Score

The current implementation compares backends lexicographically:

```text
score_b = (W_b, F_b, L_b, N_b, backend_name_b)
```

The selected backend is:

```text
b* = argmin_b score_b
```

Tie-breaking order is:

1. lower weighted total load
2. lower weighted inflight count
3. lower latency EMA
4. lower normal load
5. deterministic backend name

### Dispatcher Load Update On Assignment

For the selected backend `b*`, before forwarding the request:

```text
if is_sacrificial(r):
    S_b* <- S_b* + E_r
    I^S_b* <- I^S_b* + 1
else:
    N_b* <- N_b* + E_r
    I^N_b* <- I^N_b* + 1
```

### Latency EMA Update

After the backend response returns with observed elapsed time `D_r`:

```text
if L_b == 0:
    L_b <- D_r
else:
    L_b <- 0.9 * L_b + 0.1 * D_r
```

### Inflight Decrement

After response or failure:

```text
if is_sacrificial(r):
    I^S_b <- max(I^S_b - 1, 0)
else:
    I^N_b <- max(I^N_b - 1, 0)
```

### Authoritative Load Feedback

If the backend response includes load headers:

```text
N_b <- X-Super-P95-Normal-Load-S
S_b <- X-Super-P95-Sacrificial-Load-S
```

These backend-reported values replace dispatcher estimates.

### Fallback Load Removal

If backend load headers are missing:

```text
if is_sacrificial(r):
    S_b <- max(S_b - E_r, 0)
else:
    N_b <- max(N_b - E_r, 0)
```

## 5. In-Card Scheduling Formulas

`SuperP95StepScheduler` assumes the diffusion engine can execute one denoising
step per scheduling cycle. Preemption is therefore represented by selecting a
different request at the next scheduling cycle, rather than interrupting a
request in the middle of a step.

### Symbols

For local request `r`:

- `seq_r`: local arrival sequence
- `E_r`: estimated total service time, in seconds
- `C_r`: completed diffusion steps
- `P_r`: total diffusion steps
- `R_r`: remaining diffusion steps
- `rem_r`: remaining estimated service time, in seconds
- `s_r`: whether request `r` is sacrificial

### Remaining Steps

```text
R_r = max(P_r - C_r, 0)
```

### Remaining Estimated Service Time

```text
rem_r = E_r * R_r / max(P_r, 1)
```

### Normal Queue Sort Key

Normal requests preserve FIFO:

```text
sort_key_normal(r) = (seq_r)
```

Lower `seq_r` runs earlier.

### Sacrificial Queue Sort Key

Sacrificial requests use newest-first order:

```text
sort_key_sacrificial(r) = (-seq_r)
```

Higher `seq_r` runs earlier among sacrificial requests.

### Queue Priority

The scheduler always checks queues in this order:

```text
normal_pending first
sacrificial_pending second
```

Therefore normal requests dominate sacrificial requests regardless of
sacrificial arrival order.

### Preemption Rule

Let `c` be the best pending candidate and `x` be the currently running request.
The current implementation preempts only when:

```text
preempt(c, x) = (s_c == false) and (s_x == true)
```

That is, a normal candidate can preempt a sacrificial incumbent. A sacrificial
candidate cannot preempt a normal incumbent.

### Step Progress Update

When runner output reports `step_index`, completed steps are clamped:

```text
C_r <- min(max(step_index, C_r), P_r)
```

Delta steps:

```text
Delta_r = max(C_r(new) - C_r(old), 0)
```

Scheduler logical time advances by estimated service for completed steps:

```text
time <- time + E_r * Delta_r / max(P_r, 1)
```

When a request finishes:

```text
C_r <- P_r
rem_r <- 0
```

### Backend Load Snapshot

For all unfinished local requests:

```text
normal_load = sum(rem_r for r where s_r == false)
sacrificial_load = sum(rem_r for r where s_r == true)
```

The backend reports these values through:

```text
X-Super-P95-Normal-Load-S = normal_load
X-Super-P95-Sacrificial-Load-S = sacrificial_load
```

## 6. Service-Time Estimation Formulas

The dispatcher and backend use the same service-time estimator.

### Image Requests

For Qwen-Image-like requests:

- `w`: width
- `h`: height
- `p = (w, h)`: resolution
- `steps`: denoising step count
- `A_p = (anchor_steps_p, anchor_latency_p)`: measured anchor for resolution
  `p`

If an exact resolution anchor exists:

```text
per_pixel_step_ms_p = anchor_latency_p * 1000 / (w * h * anchor_steps_p)
E_r = per_pixel_step_ms_p * w * h * steps / 1000
```

If no exact anchor exists, use the fallback resolution `(1024, 1024)`:

```text
per_pixel_step_ms_fallback =
    anchor_latency_fallback * 1000
    / (1024 * 1024 * anchor_steps_fallback)

E_r = per_pixel_step_ms_fallback * w * h * steps / 1000
```

Default image values:

```text
w = width or resolution or 1024
h = height or resolution or 1024
steps = max(num_inference_steps or 25, 1)
frames = max(num_frames or 1, 1)
```

If `frames > 1`, the request is treated as a video request.

### Video Requests

For Wan2.2-like video requests:

- `w`: width
- `h`: height
- `steps`: denoising step count
- `frames`: frame count
- `key = (w, h, steps, frames)`
- fallback key is `(1280, 720, 6, 80)`

If an exact anchor exists:

```text
E_r = anchor_latency_key
```

If no exact anchor exists:

```text
frames_norm = normalize_frames(frames)

per_pixel_frame_step_ms =
    anchor_latency_fallback * 1000
    / (1280 * 720 * 6 * 80)

E_r = per_pixel_frame_step_ms * w * h * steps * frames_norm / 1000
```

Default video values:

```text
w = width or 854
h = height or 480
steps = max(num_inference_steps or 1, 1)
frames = max(num_frames or 1, 1)
```

### Wan2.2 Frame Normalization

The current VAE temporal scale factor is:

```text
q = 4
```

Frame normalization:

```text
frames = max(num_frames or 1, 1)

if frames % q != 1:
    frames = floor(frames / q) * q + 1

frames_norm = max(frames, 1)
```

### Request-Path Parameter Mapping

For `/v1/chat/completions`:

```text
width = extra_body.width
height = extra_body.height
steps = extra_body.num_inference_steps
frames = extra_body.num_frames
```

For `/v1/images/generations`:

```text
(width, height) = parse_size(body.size)
steps = body.num_inference_steps
frames = body.num_frames
```

For `/v1/videos`:

```text
(width_from_size, height_from_size) = parse_size(body.size)
width = body.width or width_from_size
height = body.height or height_from_size
steps = body.num_inference_steps
```

If `num_frames` is absent but `seconds` is present:

```text
fps = body.fps or 24
frames = seconds * fps
```

Otherwise:

```text
frames = body.num_frames
```

Known profiles:

- `910B2`
- `910B3`

Unknown profiles normalize to the default profile.

## 7. Metadata And Feedback Protocol

Dispatcher-to-backend request headers:

- `X-Super-P95-Estimated-Service-S`
- `X-Super-P95-Sacrificial`

Backend-to-dispatcher response headers:

- `X-Super-P95-Normal-Load-S`
- `X-Super-P95-Sacrificial-Load-S`

Internal sampling `extra_args` keys:

- `_super_p95_sacrificial`
- `_super_p95_estimated_service_s`

This protocol keeps the dispatcher and backend loosely coupled. The dispatcher
does not need to inspect backend scheduler internals; the backend only needs to
understand a small amount of metadata and report load snapshots.

## 8. Startup And Benchmark Procedure

Before every benchmark:

1. Kill dispatcher and backend processes.
2. Clear `/dev/shm` if safe for the environment.
3. Run `npu-smi info`.
4. Verify every NPU reports no running process.
5. Start dispatcher and managed backend services.
6. Wait until every backend `/health` returns `200`.
7. Wait until dispatcher `/health` returns `200`.
8. Run the benchmark script.

The reliable measurement path is:

- `benchmarks/diffusion/diffusion_benchmark_serving.py`
- `--backend openai` or the specific OpenAI-compatible path
- managed dispatcher/backend topology

Direct ad hoc HTTP calls are useful for debugging API behavior, but should not
be treated as the canonical performance measurement path.

## 9. Known Limitations And Risks

### Dirty Runtime State

Dirty NPU state has previously changed results from healthy throughput to bad
throughput without a meaningful code change. Benchmark comparison is invalid
unless both runs start from a clean environment.

### Startup Instability

Managed multi-backend startup has shown two modes:

- all backends become healthy and dispatcher health succeeds
- processes exist but `/health` hangs, returns empty replies, or never reaches
  startup completion

Process liveness is not readiness. Health endpoints must be used.

### Estimation Drift

The service-time estimator is anchor-based and workload-specific. New models,
hardware, resolution buckets, or video settings need calibration. Estimation
errors can cause poor sacrificial classification and backend imbalance.

### Throughput Tax

The policy intentionally changes scheduling order. Lower P95 is expected to
come with some throughput and sacrificial-latency cost. The acceptable tax must
be defined per workload.

### Architecture Drift

The rewrite notes warn that the current branch mixed old request-mode recovery
with the newer `v0.18` runtime. Future work should avoid preserving accidental
structure from the old branch when a cleaner stepwise-native design is
available.

## 10. Recommended Rewrite Plan

### Milestone 1: Clean Baseline

- branch from clean upstream `v0.18.0`
- keep baseline diffusion execution behavior unchanged
- validate single-backend and 8-backend startup
- validate baseline throughput against known clean numbers

Acceptance criteria:

- all health endpoints reliably reach `200`
- baseline benchmark throughput is stable across repeated clean runs
- no `super_p95` behavior is enabled yet

### Milestone 2: Dispatcher Metadata Only

- port dispatcher request parsing and service-time estimation
- add request headers for estimated service time and sacrificial marker
- add backend response headers for load snapshots
- do not enable backend preemption yet

Acceptance criteria:

- dispatcher can route to all backends
- headers are visible at backend API layer
- baseline behavior remains equivalent when scheduler is baseline

### Milestone 3: Stepwise Backend Scheduler

- implement or port `SuperP95StepScheduler`
- force native stepwise execution when scheduler is `super_p95_step`
- keep normal FIFO behavior
- allow normal request to preempt sacrificial request at step boundary
- report accurate remaining-load snapshots

Acceptance criteria:

- unit tests cover queue ordering, preemption, and load snapshots
- single-backend preemption resumes from completed step state
- completed steps are not repeated

### Milestone 4: Multi-Backend Policy Validation

- enable quota-based sacrificial classification
- tune `quota_every`, `quota_amount`, `threshold_ratio`, and
  `sacrificial_load_factor`
- compare baseline and `super_p95` on clean 8-backend runs

Acceptance criteria:

- P95 improves on target workload
- throughput regression remains within the chosen bound
- backend load feedback is stable and does not drift indefinitely

### Milestone 5: Production Hardening

- add structured trace events behind a low-overhead flag
- add startup failure diagnostics
- add estimator calibration documentation
- add benchmark result templates
- document unsupported request shapes and fallback behavior

Acceptance criteria:

- failed backend startup produces actionable diagnostics
- benchmark output includes enough metadata to reproduce the run
- estimator changes require tests and calibration notes

## 11. Validation Plan

Unit tests:

- service-time estimation for images and videos
- header parsing and response header generation
- dispatcher classification and credit consumption
- backend selection with normal and sacrificial load
- scheduler ordering for normal FIFO
- scheduler ordering for sacrificial newest-first
- preemption of sacrificial by normal request
- load snapshot after partial step completion

Integration tests:

- single backend baseline request
- single backend `super_p95_step` request
- single backend preemption/resume correctness
- dispatcher with one managed backend
- dispatcher with multiple managed backends
- health endpoint behavior during startup and shutdown

Benchmark stages:

1. clean baseline startup stability
2. clean baseline throughput
3. single-backend preemption correctness
4. two-backend sanity
5. eight-backend 100-request smoke benchmark
6. eight-backend 500-request comparison benchmark

Metrics to track:

- throughput
- mean latency
- median latency
- P95 latency
- P99 latency
- normal-request latency distribution
- sacrificial-request latency distribution
- backend load snapshots
- preemption count
- failed request count
- startup time and health-check time

## Proposed Default Tuning

For Qwen-Image 8-backend validation, use the current documented defaults as the
starting point:

- `quota_every = 20`
- `quota_amount = 1`
- `threshold_ratio = 0.8`
- `sacrificial_load_factor = 0.1`
- hardware profile `910B3` when running on 910B3

For Wan2.2 validation, keep the existing single-managed-backend topology first,
then expand only after startup and single-backend correctness are stable.

## Final Recommendation

The current technical direction is sound at the policy level: dispatcher-side
classification plus backend-side stepwise scheduling is the right decomposition.
The main engineering risk is not the policy itself, but carrying over too much
old request-mode recovery machinery.

The next implementation should therefore preserve the semantics, not the old
shape:

- preserve quota-based sacrificial classification
- preserve estimated-service metadata
- preserve backend load feedback
- preserve normal-over-sacrificial scheduling semantics
- rebuild execution around native `v0.18` stepwise diffusion scheduling
