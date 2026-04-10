# v0.18.0 Super P95 Port Plan

## Goal

Port the diffusion `super_p95` scheduling and async preemption design from the
historical `v0.16.0` implementation onto the official `v0.18.0` codebase.

This branch intentionally starts from a clean `v0.18.0` baseline and does not
apply the old implementation by cherry-pick. The port will be re-implemented in
small steps to match the new runtime structure.

Important scope clarification:

- this plan is about porting the feature set onto `v0.18.0`
- it is not a claim that the current branch is still a clean upstream baseline
- before each major experiment, we must distinguish:
  - clean upstream `v0.18.0`
  - `v0.18.0 + ported super_p95 feature`
  - `v0.18.0 + runtime workaround`

## Non-Goals

- Do not change unrelated `v0.18.0` behavior unless required for the port.
- Do not optimize benchmark scripts or experiment defaults before the core port
  is stable.
- Do not attempt to preserve the old file layout if `v0.18.0` has already
  replaced it with new abstractions.

## Why Re-Implement Instead of Cherry-Pick

The old `v0.16.0` implementation was built on runtime pieces that no longer
match `v0.18.0`:

- old diffusion scheduler/runtime layout
- old executor / worker interaction pattern
- old entrypoint plumbing
- old request metadata flow

Trying to replay the large `v0.16.0` squash commit directly would mix together:

- true feature logic
- outdated runtime assumptions
- benchmark/documentation changes

That would make review and debugging much harder than a structured port.

## Porting Principles

1. Keep the `v0.18.0` runtime model first-class.
2. Port the smallest viable feature set first.
3. Separate feature porting from experiment/documentation updates.
4. Verify each layer independently before moving upward.
5. Prefer explicit compatibility shims over invasive rewrites.
6. Separate algorithm parity from runtime stabilization.

## Definition Of Done

The port is only considered complete when all of the following are true:

1. Clean `v0.18.0` baseline can run target Qwen-Image and Wan2.2 experiments.
2. `super_p95` can be toggled on and off without changing unrelated behavior.
3. Dispatcher-side formulas match the old implementation or documented
   replacement formulas.
4. Backend-side local scheduling and preemption semantics match the documented
   design.
5. Qwen-Image preemption/recovery tests pass.
6. Wan2.2 preemption/recovery tests pass.
7. Benchmark scripts and docs clearly distinguish:
   - baseline
   - `super_p95`
   - runtime workaround configuration

## Target Feature Set

The port should eventually restore the following capabilities:

1. Dispatcher-side `super_p95` backend selection and request annotation.
2. Backend-side diffusion server scheduling.
3. Async preemption at diffusion step boundaries.
4. Request state save/load needed for preemption recovery.
5. Qwen-Image support.
6. Wan2.2 support.
7. Benchmark scripts and docs aligned with the new implementation.

## Proposed Work Breakdown

### Phase 1: Runtime Mapping

Map old concepts to the `v0.18.0` runtime:

- request lifecycle entrypoints
- diffusion engine scheduling entrypoints
- worker RPC path
- result publication path
- request metadata plumbing from HTTP layer to diffusion runtime

Deliverable:

- a short design note appended to this file with exact file mappings
- a list of old files with no direct `v0.18.0` equivalent

### Phase 2: Dispatcher Port

Port only the dispatcher-side pieces:

- backend load estimation
- hardware profile resolution
- request classification metadata
- header propagation format

Do not enable backend scheduling yet.

Deliverable:

- dispatcher can run against unmodified `v0.18.0` backends
- request metadata arrives at serving layer

### Phase 3: Backend Metadata Plumbing

Port the metadata path into diffusion requests:

- request model fields
- sampling params metadata
- API server / serving layer propagation

Do not schedule on it yet.

Deliverable:

- backend receives `super_p95` metadata with no behavior change

### Phase 4: Minimal Server Scheduling

Introduce backend scheduling with the smallest possible scope:

- local waiting queue
- active request selection
- request completion publication

At this phase, prefer correctness and observability over aggressive preemption.

Deliverable:

- server scheduling can be toggled on/off
- baseline path remains intact when disabled

### Phase 5: Async Preemption Recovery

Add step-boundary async preemption:

- interrupt signalling
- per-request resumable state
- save/load of minimum recovery state
- resumed execution correctness

Deliverable:

- Qwen-Image async preemption passes targeted tests

### Phase 6: Wan2.2 Port

Port Wan2.2-specific stepwise/preemption behavior carefully:

- match `v0.18.0` pipeline structure
- avoid re-introducing known high-overhead timestep logic
- keep non-preempted hot path close to baseline behavior

Deliverable:

- Wan2.2 baseline and `super_p95` both run

### Phase 7: Validation and Experiment Layer

Only after the feature path is stable:

- restore benchmark scripts
- restore experiment docs
- compare baseline vs `super_p95`
- investigate high-load runtime issues specific to `v0.18.0`

## Initial File Candidates

Likely primary files for the port:

- `benchmarks/diffusion/super_p95_dispatcher.py`
- `vllm_omni/diffusion/diffusion_engine.py`
- `vllm_omni/diffusion/request.py`
- `vllm_omni/diffusion/executor/multiproc_executor.py`
- `vllm_omni/diffusion/worker/diffusion_worker.py`
- `vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py`
- `vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py`
- `vllm_omni/entrypoints/openai/api_server.py`
- `vllm_omni/entrypoints/openai/serving_chat.py`
- `vllm_omni/entrypoints/openai/serving_video.py`

## Explicit File-Level Mapping Status

This section exists to avoid vague "we will port it later" planning. Each old
area must end up in one of these categories:

- direct mapping
- semantic mapping
- no direct equivalent, redesign required

### Dispatcher Layer

| Old source | v0.18.0 target | Status | Notes |
|---|---|---|---|
| `benchmarks/diffusion/super_p95_dispatcher.py` | `benchmarks/diffusion/super_p95_dispatcher.py` | direct mapping | Can be reintroduced largely as feature code, but managed-backend startup must respect new runtime behavior. |
| `vllm_omni/diffusion/super_p95.py` | same path | direct mapping | Metadata/header helpers are largely runtime-agnostic. |

### Backend Scheduling Layer

| Old source | v0.18.0 target | Status | Notes |
|---|---|---|---|
| `vllm_omni/diffusion/server_scheduling.py` | same path or split across `sched/*` | semantic mapping | Formulas reusable, scheduler ownership not reusable verbatim. |
| `vllm_omni/diffusion/scheduler.py` | `vllm_omni/diffusion/sched/*` | no direct equivalent | Must not be restored as a parallel scheduler stack. |

### Engine / Executor Layer

| Old source | v0.18.0 target | Status | Notes |
|---|---|---|---|
| `vllm_omni/diffusion/diffusion_engine.py` | same path | semantic mapping | Integrate through existing `SchedulerInterface` loop. |
| `vllm_omni/diffusion/executor/multiproc_executor.py` | same path | semantic mapping | Old control-path assumptions do not hold. |
| `vllm_omni/diffusion/worker/diffusion_worker.py` | same path | semantic mapping | Stepwise execution hooks must match new worker contract. |

### Request / API Plumbing

| Old source | v0.18.0 target | Status | Notes |
|---|---|---|---|
| `vllm_omni/diffusion/request.py` | same path | semantic mapping | Request metadata likely portable, but request lifecycle differs. |
| `vllm_omni/entrypoints/openai/api_server.py` | same path | semantic mapping | Must re-hook header propagation carefully. |
| `vllm_omni/entrypoints/openai/serving_chat.py` | same path | semantic mapping | Needed for image-generation requests routed via chat path. |
| `vllm_omni/entrypoints/openai/serving_video.py` | same path | semantic mapping | Needed for video request metadata. |

### Pipeline Layer

| Old source | v0.18.0 target | Status | Notes |
|---|---|---|---|
| `pipeline_qwen_image.py` | same path | semantic mapping | Preemption semantics portable, helper structure likely not. |
| `pipeline_wan2_2.py` | same path | semantic mapping | Requires fresh performance review to avoid old hot-path regressions. |

### Old Files With No Acceptable Direct Port

These old files or structures should not be recreated mechanically:

- `vllm_omni/diffusion/scheduler.py`
- old scheduler-owned result publication flow
- old pipe-assisted executor result/control path

Reason:

- each conflicts with an already-established `v0.18.0` ownership model

## Known Risks

1. `v0.18.0` runtime behavior differs materially from `v0.16.0`.
2. Shared-memory / worker coordination behavior may dominate high-load results.
3. Wan2.2 may require separate handling from Qwen-Image even if both share the
   same scheduling framework.
4. Benchmark comparability can be invalidated by silent changes in arrival
   process, timeout, or concurrency limits.

## Guardrails

Before claiming parity with the old implementation:

- baseline must remain reproducible on clean `v0.18.0`
- all ported behavior must be behind explicit toggles
- benchmark defaults must match documented experiment settings
- temporary debug hooks must not become default behavior

## Immediate Next Step

Next change after this document:

- inspect `v0.18.0` diffusion runtime and write a concrete old-to-new component
  mapping before porting any feature code

## Old-to-New Conflict Map

This section records the concrete structural conflicts between the historical
`v0.16.0` implementation and the current `v0.18.0` runtime. These are the main
reasons we are not attempting a large cherry-pick.

### 1. Scheduler Topology Conflict

`v0.16.0` introduced a custom scheduler stack centered around:

- `vllm_omni/diffusion/scheduler.py`
- `vllm_omni/diffusion/server_scheduling.py`
- `Scheduler`-owned request/result publication
- direct integration in the multiprocess executor

`v0.18.0` already has a different scheduler abstraction:

- `vllm_omni/diffusion/sched/base_scheduler.py`
- `vllm_omni/diffusion/sched/request_scheduler.py`
- `SchedulerInterface`
- `DiffusionSchedulerOutput`

Conflict:

- the old implementation replaces the scheduling model
- the new runtime expects a `sched/*` interface and a vLLM-style waiting/running
  state machine

Resolution strategy:

- do not restore the old monolithic `scheduler.py`
- port `super_p95` into the `sched/*` interface
- any server-side scheduling feature must appear as a new scheduler
  implementation or a scheduler extension, not as a parallel scheduler stack

### 2. Executor Communication Conflict

`v0.16.0` executor behavior:

- broadcast requests through a main queue
- return generation results partly through worker control pipes
- maintain extra scheduler-side reader thread logic

`v0.18.0` executor behavior:

- use `_broadcast_mq` for worker RPC
- use `_result_mq` for replies
- no `v0.16.0`-style control-pipe result publication path

Conflict:

- the old implementation assumes a mixed `mq + pipe` control model
- the new implementation centralizes control flow through `MessageQueue`

Resolution strategy:

- adapt server scheduling and preemption to the existing `_broadcast_mq /
  _result_mq` model
- do not port the old control-pipe design literally
- treat `v0.18.0` message-queue behavior as the source of truth

### 3. Engine Execution Model Conflict

`v0.16.0` old ported code changed `DiffusionEngine.add_req()` and related logic
to support:

- direct execution path
- server scheduling path
- synchronous waiting on `done_event`
- result publication through scheduler-owned state

`v0.18.0` current engine model:

- owns a `SchedulerInterface`
- calls `schedule()`
- submits request(s) to executor
- calls `update_from_output()`

Conflict:

- the old engine embeds scheduling decisions in a different control loop
- the new engine already has an explicit schedule/update loop

Resolution strategy:

- preserve the `v0.18.0` engine contract
- port `super_p95` as a scheduler/runtime extension behind explicit toggles
- avoid introducing a second engine execution loop

### 4. Request Metadata Conflict

Old implementation adds request metadata via:

- `X-Super-P95-*` headers
- `sampling_params.extra_args`
- request-level interpretation in backend scheduling

`v0.18.0` still supports `sampling_params.extra_args`, but the request / API /
serving layers are not wired for `super_p95` by default.

Conflict:

- metadata format is conceptually reusable
- metadata plumbing entrypoints changed

Resolution strategy:

- preserve the header names and `extra_args` names exactly where possible
- port only the plumbing, not a second metadata schema

### 5. Pipeline Preemption Conflict

Old implementation adds async preemption behavior into:

- `pipeline_qwen_image.py`
- `pipeline_wan2_2.py`

`v0.18.0` pipeline code differs substantially:

- function boundaries changed
- internal runner / scheduler output structures changed
- some pipeline-specific optimizations differ from `v0.16.0`

Conflict:

- a mechanical transplant would almost certainly reintroduce outdated control
  flow and performance bugs

Resolution strategy:

- port preemption behavior at the semantic level:
  - step-boundary interrupt
  - minimal resumable state
  - resumed execution correctness
- do not copy the old internal helper structure blindly

## Formula and Semantics Consistency Checklist

The port must preserve these semantics unless we explicitly decide otherwise.

### A. Dispatcher Load Formula

Old dispatcher scoring:

```text
weighted_total_load = normal_load + alpha * sacrificial_load
score_tuple = (
  weighted_total_load,
  inflight_normal + alpha * inflight_sacrificial,
  latency_ema,
  normal_load,
  backend_name,
)
```

Consistency requirement:

- backend selection in the port must preserve this ordering unless the design
  review explicitly changes it

Open decision:

- whether `latency_ema` should remain as a late tie-breaker
- whether `normal_load` should remain ahead of backend name

Missing detail to preserve:

- response feedback loop must also remain consistent:
  - decrement inflight counts on response
  - authoritative backend load headers override local estimates
  - fallback removes estimated load when authoritative headers are absent

### B. Sacrificial Classification Rule

Old rule:

```text
credits += quota_amount every quota_every arrivals
is_sacrificial =
  credits > 0
  and global_max_service > 0
  and estimated_service >= threshold_ratio * global_max_service
if sacrificial:
  credits -= 1
```

Consistency requirement:

- preserve the quota/credit semantics
- preserve the threshold against `global_max_service`

Open decision:

- whether `global_max_service` remains process lifetime state
- or should be windowed / decayed in `v0.18.0`

Missing detail to preserve:

- sacrificial classification happens before backend selection in the old code
- the selected backend then accounts the request into either normal or
  sacrificial load buckets

### C. Service-Time Estimation Formula

Old code uses:

- exact anchor lookup where possible
- fallback scaling from per-pixel-step or per-pixel-frame-step anchors

For Qwen-Image:

```text
service_s = per_pixel_step_ms * width * height * steps / 1000
```

For Wan2.2 fallback:

```text
service_s = per_pixel_frame_step_ms * width * height * steps * frames / 1000
```

Consistency requirement:

- preserve exact anchors
- preserve fallback scaling logic
- keep hardware-profile normalization behavior consistent

Open decision:

- whether anchors should remain embedded in backend-side code
- or be moved to benchmark / config data

Missing detail to preserve:

- exact-match anchor lookup must take precedence over fallback scaling
- hardware-profile normalization must produce identical fallback behavior on
  unknown profiles

### D. Server-Side Local Scheduling Rule

Old `PredictedLatencyPolicy` uses:

```text
priority = remaining_service_s - arrival_time_s

normal: sort_key = (-priority, arrival_seq)
sacrificial: sort_key = (priority, arrival_seq)
```

Interpretation:

- normal queue prefers larger `remaining_service - arrival_time`
- sacrificial queue prefers smaller `remaining_service - arrival_time`

Consistency requirement:

- preserve this exact ordering if we claim parity with the old strategy

Open decision:

- whether this local queue behavior should remain unchanged once mapped into
  `v0.18.0` scheduler abstractions

Missing detail to preserve:

- `request_outranks(candidate, incumbent)` semantics must remain exactly
  compatible if we want parity with old preemption behavior
- update-after-quantum accounting must advance both:
  - `current_time_s`
  - `remaining_service_s`

### E. Preemption Boundary Semantics

Old intended semantics:

- preempt only at diffusion step boundaries
- save/load only minimum recovery state
- no arbitrary mid-kernel interruption

Consistency requirement:

- preserve step-boundary-only preemption semantics
- do not introduce a stronger or more invasive interruption model

Missing detail to preserve:

- "supports preemption" must not imply "always pay full resumable-state cost"
- non-preempted hot paths should remain close to baseline where possible

## Experiment Consistency Guardrail

The old implementation history showed that experiment conclusions can be invalid
even when the feature code is correct. Therefore benchmark methodology is part
of migration correctness.

The port must keep these concerns explicit:

1. arrival process semantics must be documented
2. timeout policy must be documented
3. `max_concurrency` must match the intended experiment contract
4. baseline must not accidentally route through feature logic
5. benchmark scripts must distinguish:
   - debugging runs
   - formal comparison runs

## Fine-Grained Porting Questions

These must be answered explicitly during implementation.

### 1. What is the `v0.18.0` equivalent of old scheduler-owned request completion?

In `v0.16.0`, completion and publish flow were tightly coupled to the custom
scheduler. In `v0.18.0`, request completion flows through:

- `SchedulerInterface.schedule()`
- executor execution
- `update_from_output()`

Decision needed:

- whether server scheduling lives entirely inside a new scheduler
- or partly in engine / executor cooperation

Current preferred answer:

- keep ownership centered in the scheduler contract as much as possible
- only extend engine/executor where the scheduler contract is insufficient

### 2. Where should stepwise execution live?

Old implementation added stepwise behavior to executor/worker/pipeline together.

Decision needed:

- whether stepwise execution should be represented in `DiffusionSchedulerOutput`
- whether `RequestScheduler` needs a new scheduler output subtype
- whether executor APIs should branch on request mode vs stepwise mode

Current preferred answer:

- extend scheduler output rather than introducing a second external control loop
- keep executor branching explicit instead of implicit request-type checks

### 3. How should result publication work under preemption?

Old implementation used explicit scheduler-owned result publication state.

Decision needed:

- whether `DiffusionOutput` should remain the only engine-facing completion
  object
- or whether resumed / partial execution needs an intermediate output type

Current preferred answer:

- keep `DiffusionOutput` as the terminal completion type
- introduce a clearly separate partial/stepwise runtime type if needed, instead
  of overloading terminal outputs

### 4. How much of the old header protocol should be preserved?

Header names currently provide a good compatibility boundary:

- `X-Super-P95-Sacrificial`
- `X-Super-P95-Estimated-Service-S`
- response load headers

Decision needed:

- preserve them exactly for compatibility
- or simplify them if no external dependency exists

Current recommendation:

- preserve them exactly unless a later migration problem forces change

### 5. What is part of the strategy, and what is a runtime workaround?

This was under-specified in the first draft and must be explicit.

Examples of strategy logic:

- dispatcher classification
- backend local queue ordering
- async preemption semantics

Examples of runtime workaround:

- fixed rendezvous/master-port assignment
- shared-memory queue sizing changes
- timeout values used only to stop experiments from deadlocking

Decision needed:

- every such workaround must be tracked separately from algorithm commits and
  experiment claims

## Places Likely To Require Redesign, Not Porting

These are the areas most likely to require fresh implementation decisions.

### 1. Scheduler Integration

The old `scheduler.py` should not be recreated verbatim.

Reason:

- it conflicts with `v0.18.0` `sched/*` ownership

### 2. Executor Control Path

The old pipe-assisted result/control model should not be restored directly.

Reason:

- `v0.18.0` executor already consolidated around `MessageQueue`

### 3. High-Load Runtime Behavior

Recent `v0.18.0` experiments exposed:

- shared-memory broadcast stalls
- backend startup rendezvous-port collisions

These are runtime issues, not part of the strategy definition itself.

Decision needed:

- keep the strategy port isolated from runtime mitigations
- track runtime mitigations separately so they are not mistaken for algorithm
  changes

Additional note:

- high-load runtime mitigations may be required before any algorithm comparison
  is meaningful on `v0.18.0`
- this means "algorithm parity" and "experiment parity" are separate milestones

### 4. Wan2.2 Internal Fast Path

The old Wan2.2 code included performance-sensitive fixes specific to its
preemption path. Those should be re-evaluated against the new pipeline layout,
not transplanted wholesale.

## Required Verification Before Each Porting Step

For each migrated piece, verify:

1. Formula equality against the old implementation.
2. Header names and metadata field names.
3. Request ordering semantics.
4. Whether baseline-disabled mode stays identical to clean `v0.18.0`.
5. Whether any new behavior is actually strategy logic or an unrelated runtime
   workaround.
6. Whether the same benchmark script and experimental contract still mean the
   same thing after the code change.

## Next Concrete Planning Task

Before touching code, the next document update should add a file-by-file mapping:

- old file
- new file or "no direct equivalent"
- port strategy
- expected conflict type

After that, implementation should begin with the smallest low-risk layer:

- metadata/header portability first
- scheduler integration second
- preemption recovery third
- benchmark/docs last

## Concrete Resolution Plan For The Two Core Conflicts

This section turns the two most important migration conflicts into explicit
implementation decisions.

### Core Conflict 1: Scheduler Ownership

#### Problem

The `v0.16.0` implementation assumes that the feature can own the diffusion
scheduler end-to-end. That is no longer true in `v0.18.0`, where the runtime
already owns:

- request state storage
- waiting/running lifecycle
- scheduler output contract
- engine schedule/update loop

If we recreate the old scheduler stack verbatim, we would end up with:

- duplicate request state ownership
- duplicate completion semantics
- a baseline path that is no longer a clean `v0.18.0` baseline

#### Resolution

The port must implement `super_p95` as a `v0.18.0` scheduler-compatible
extension, not as a second scheduler universe.

#### Concrete Implementation Path

1. Keep `SchedulerInterface` as the external contract.
2. Add a new scheduler implementation, tentatively:
   - `SuperP95RequestScheduler`
3. Reuse the old formulas and policy semantics inside that scheduler:
   - service-time estimation
   - normal vs sacrificial classification
   - local queue ordering
   - preemption ordering rule
4. Keep the existing engine control loop:
   - `add_request()`
   - `schedule()`
   - executor call
   - `update_from_output()`
5. Do not reintroduce old `scheduler.py` as a parallel owner of runtime state.

#### Required Design Constraint

When `super_p95` is disabled:

- engine behavior must remain identical to clean `v0.18.0`
- no alternate request lifecycle is allowed

#### What Must Be Reused Exactly

The following old semantics should be preserved exactly unless explicitly
reviewed and changed:

- dispatcher load accounting formula
- sacrificial credit/threshold rule
- backend local queue ordering
- preemption outrank rule

### Core Conflict 2: Executor / Runtime Communication

#### Problem

The old implementation relied on a different communication topology:

- request broadcast path
- worker control pipe path
- scheduler-owned publication path

`v0.18.0` instead relies much more directly on:

- `_broadcast_mq`
- `_result_mq`
- `MessageQueue`
- shared-memory broadcast semantics

Recent experiments already showed that the runtime layer can dominate the
observed behavior under high load:

- startup rendezvous port collisions
- shared-memory broadcast stalls

Therefore, if we mix runtime rewrites into the feature port, we will not be
able to tell whether an observed behavior comes from:

- the strategy
- the preemption implementation
- or the message-passing runtime

#### Resolution

Adopt the `v0.18.0` communication model first. Port the feature onto it before
making any structural runtime redesign.

#### Concrete Implementation Path

1. Treat `_broadcast_mq / _result_mq` as fixed runtime contracts.
2. Introduce stepwise execution through executor/worker APIs that fit that
   model.
3. Represent preemption state transitions explicitly in scheduler/runtime data,
   not through side channels.
4. Track runtime workarounds separately from algorithm work.

#### Runtime Workarounds That Must Stay Separate

The following are not part of `super_p95` semantics and must be tracked as
runtime-only work:

- fixed rendezvous port assignment
- startup serialization or parallel-start safety logic
- `shm_broadcast` buffer sizing changes
- benchmark timeout policy used only to avoid deadlock-like stalls

#### Required Design Constraint

Any runtime mitigation must satisfy both:

1. baseline behavior stays attributable to upstream runtime + mitigation
2. `super_p95` claims do not depend on undocumented runtime hacks

### Practical Order Of Work

The first real implementation work should follow this order:

1. Dispatcher metadata path
2. Backend metadata plumbing
3. Scheduler integration on top of `sched/*`
4. Stepwise execution contract
5. Async preemption recovery for Qwen-Image
6. Wan2.2-specific adaptation
7. Runtime mitigation only where experiments prove it is necessary

### Stop Conditions

If any implementation step attempts to:

- recreate old `scheduler.py` ownership
- reintroduce old pipe-based result/control flow
- or silently change baseline execution semantics

then the port should stop and be redesigned before more code is added.
