# Super P95 v0.18 Rewrite Notes

This document records the main lessons from the current `super_p95` port attempt.
It is intended to guide a fresh implementation starting from clean `v0.18.0`,
using the upstream `v0.18` diffusion runtime as the source of truth.

## Scope

This note focuses on:

- environment and benchmark hygiene
- pitfalls that consumed time but were not root causes
- confirmed or strongly-supported findings
- concrete guardrails for the next rewrite

It does not try to preserve the current branch design.

## High-Level Conclusion

The current branch mixed:

- `v0.16`-style request-mode preemption/recovery semantics
- `v0.18` runtime, scheduler, and executor structure

That combination led to:

- hard-to-reason-about control flow
- unstable startup behavior in managed multi-backend mode
- repeated false attributions during performance debugging

The next attempt should start from clean `v0.18.0` and implement `super_p95`
on top of the native `v0.18` diffusion execution model, especially its
stepwise/stateful execution path, instead of porting the old request-mode
recovery flow directly.

## Environment And Measurement Hygiene

### 1. Always clean the environment before startup

This was the single biggest source of confusion.

Observed behavior:

- dirty NPU state could degrade results from normal (`~0.41-0.45 qps`) to bad
  (`~0.27-0.28 qps`)
- startup success itself could become non-deterministic if old contexts were not
  fully released

Required startup procedure:

1. kill dispatcher and backend processes
2. run `npu-smi info`
3. verify every card shows:
   - `No running processes found`
4. only then start services

If `npu-smi info` still shows anonymous PIDs with large HBM usage and `ps`
cannot find them, do not continue. Wait for Ascend-side context cleanup.

### 2. Do not compare clean and dirty runs

Several large apparent regressions were later explained by environment state,
not code changes.

At least one clean 8-GPU `super_p95` run reached:

- `duration ~= 240.74s`
- `throughput ~= 0.415 qps`
- `P95 ~= 100.11s`

The same code later produced:

- `duration ~= 366s`
- `throughput ~= 0.27 qps`

This was not a code diff. It was runtime-state instability.

### 3. Use the benchmark script as the source of truth

The stable and reproducible fixed-case measurements came from:

- [diffusion_benchmark_serving.py](/vllm-workspace/vllm-omni/benchmarks/diffusion/diffusion_benchmark_serving.py)
- `--backend openai`
- managed dispatcher/backend service

Directly hand-crafting `/v1/images/generations` requests against single-backend
services produced unstable `502` failures and was not a reliable measurement
path.

For fixed-case profiling, use the benchmark script with:

- `--dataset random`
- `--num-prompts 1`
- `--max-concurrency 1`
- `--random-request-config '[{"width":...,"height":...,"num_inference_steps":...,"weight":1}]'`

## Confirmed Findings

### 1. Baseline performance can be healthy on clean runs

Clean 8-GPU baseline, 500 requests, rate 0.8:

- `duration ~= 1105.06s`
- `throughput ~= 0.4525 qps`
- `P95 ~= 407.83s`

This means the branch is not fundamentally incapable of normal baseline
throughput. Bad `0.27-0.28` baseline results were at least partly due to dirty
environment or unstable startup state.

### 2. `super_p95` improves median/P95, but costs some throughput

Clean 8-GPU, 500 requests:

- baseline:
  - `throughput ~= 0.4525 qps`
  - `P95 ~= 407.83s`
- super:
  - `throughput ~= 0.4139 qps`
  - `P95 ~= 321.00s`

So the remaining problem is not catastrophic regression. It is a real but
moderate throughput tax.

### 3. Recovery does not restart from scratch

Single-backend direct tracing showed:

- a preempted request resumed from the correct `start_index`
- completed diffusion steps were not re-run from zero

This rules out a major class of false explanations.

### 4. `restore_scheduler_state()` itself is not the heavy part

Tracing showed:

- restore logic was tiny
- no evidence that resume-time scheduler/timestep rebuild was the dominant cost

This should not be treated as the main throughput bottleneck.

### 5. The current implementation does not use native `v0.18` recovery

Current `super_p95` still uses:

- scheduler-managed pending/active/requeue logic
- worker-side resident scheduler state
- pipeline request-mode resume flow

It does **not** use the native `v0.18` diffusion step-execution path as the
main preemption/recovery mechanism.

This is the most important architectural lesson for the rewrite.

## Pitfalls And Dead Ends

### 1. Do not over-attribute regressions to trace overhead

Heavy trace can slow things down and should not be used for final performance
comparison.

But if a no-trace run is already bad, trace is not the root cause.

Use trace only for diagnosis, then validate with no-trace clean runs.

### 2. Do not trust single-backend direct HTTP path as a canonical test path

We repeatedly saw:

- backend ready
- direct `/v1/images/generations` returns `502`

while benchmark-script-driven measurements on the stable path remained usable.

Treat the benchmark path as authoritative unless the direct API path is itself
being debugged.

### 3. Do not mutate too many variables at once

The following were all explored at different times and created confusion:

- `max_workers=1` vs `32`
- control-pipe vs `result_mq`
- worker payload CPU conversion
- extra trace hooks
- fast-path scheduler experiments
- NUMA binding changes

Several of these were later proven not to be primary causes.

Future work should change one variable at a time and validate on a clean run.

### 4. Do not assume queue wait implies throughput loss

Long request queue time in local scheduler traces mainly explains latency.

If a backend is still busy running other requests, queue wait alone does not
explain lower throughput. This was an important reasoning correction.

### 5. Do not assume startup delay means normal cold start

Observed bad-state startup symptoms:

- `/health` stays `000`
- `curl` gets `Empty reply from server`
- dispatcher remains in startup hook

That is not “still loading”. Treat it as failed startup.

## Specific Implementation Lessons

### 1. Fixed master port assignment is required

Managed backend launch must use deterministic rendezvous/master ports.

Random or semi-random port assignment caused avoidable instability and made
startup debugging much harder.

### 2. NUMA binding is not the first thing to chase

Restoring NUMA binding did not explain the major regressions we observed.

NUMA can still matter, but it should not be the first hypothesis unless there
is direct evidence.

### 3. `max_workers=32` was not the baseline regression root cause

Changing the async entrypoint thread pool between `32` and `1` did not explain
the baseline degradation that was initially suspected.

Do not spend another cycle re-litigating this unless new evidence appears.

### 4. `control_pipe` result routing was not proven to be the baseline root cause

We tested rollback-style variations of result publication:

- control pipe
- `result_mq`
- `_prepare_result_for_scheduler()` on/off in some paths

These did not produce a clean causal explanation for the main observed
throughput gap.

Do not start the rewrite by replaying those local experiments.

## Startup Stability Lessons

### 1. 8-backend managed startup is non-deterministic today

The same command can enter two modes:

- good mode:
  - all backends become healthy
  - dispatcher `/health` becomes `200`
- bad mode:
  - startup hook never completes
  - backend ports return timeout or `Empty reply from server`

This instability must be treated as a first-class problem.

### 2. Ready means `/health = 200`, not “process exists”

Use:

- dispatcher `/health`
- backend `/health`

Process liveness is not enough. We repeatedly saw processes alive while services
were not actually healthy.

## What To Preserve From This Attempt

These conclusions are still useful:

- clean-environment discipline matters
- benchmark-script path is the reliable measurement path
- fixed-case Qwen timings on the stable path are approximately:
  - `512x512_s20 ~= 8.59s`
  - `768x768_s20 ~= 8.96s`
  - `1024x1024_s25 ~= 14.83s`
  - `1536x1536_s35 ~= 49.86s`
- `super_p95` can improve median/P95 on clean large-scale runs
- resume does not restart completed steps

## Recommendations For The Rewrite

### 1. Start from clean `v0.18.0`

Create a fresh branch from clean upstream `v0.18.0`.

Do not port the current branch structure forward.

### 2. Reuse only semantic requirements from the old squashed commit

Carry forward:

- dispatcher-side policy semantics
- backend selection and sacrificial metadata semantics
- preemption ordering semantics

Do **not** carry forward:

- old request-mode recovery ownership model
- worker resident-state design as the default implementation shape

### 3. Use native `v0.18` stepwise diffusion execution as the recovery base

This is the biggest architectural recommendation.

Implement preemption/recovery on top of `v0.18`’s native diffusion execution
model rather than embedding the `v0.16` request-mode flow into `v0.18`.

### 4. Revalidate in this order

1. clean baseline startup stability
2. clean baseline throughput
3. single-backend preemption correctness
4. 2-backend sanity
5. 8-backend `100` requests
6. 8-backend `500` requests

Do not jump to large-scale super benchmarks before startup stability is proven.

### 5. Keep a strict startup checklist in the new branch

Before every run:

1. kill dispatcher/backends
2. confirm `npu-smi info` shows no running processes
3. start services
4. confirm all `/health` endpoints return `200`
5. only then benchmark

## Suggested First Rewrite Milestone

The first milestone for the new branch should be:

- clean `v0.18` baseline unchanged
- managed 8-backend startup stable and reproducible
- dispatcher can annotate requests with `super_p95` metadata
- no recovery/preemption yet

Only after that should backend-side preemption/recovery be implemented.
