#!/bin/bash
# Wan2.2 online serving startup script

set -euo pipefail

MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
PORT="${PORT:-8098}"
BOUNDARY_RATIO="${BOUNDARY_RATIO:-0.875}"
FLOW_SHIFT="${FLOW_SHIFT:-5.0}"
CACHE_BACKEND="${CACHE_BACKEND:-none}"
ENABLE_CACHE_DIT_SUMMARY="${ENABLE_CACHE_DIT_SUMMARY:-0}"
ULYSSES_DEGREE="${ULYSSES_DEGREE:-1}"
RING_DEGREE="${RING_DEGREE:-1}"
CFG_PARALLEL_SIZE="${CFG_PARALLEL_SIZE:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VAE_PATCH_PARALLEL_SIZE="${VAE_PATCH_PARALLEL_SIZE:-1}"
USE_HSDP="${USE_HSDP:-0}"
HSDP_SHARD_SIZE="${HSDP_SHARD_SIZE:-}"
ENABLE_CPU_OFFLOAD="${ENABLE_CPU_OFFLOAD:-0}"
ENABLE_LAYERWISE_OFFLOAD="${ENABLE_LAYERWISE_OFFLOAD:-0}"
VAE_USE_SLICING="${VAE_USE_SLICING:-0}"
VAE_USE_TILING="${VAE_USE_TILING:-0}"

# Example presets:
#   SP=8:
#     ULYSSES_DEGREE=8 bash run_server.sh
#   SP=4 + CFG=2:
#     ULYSSES_DEGREE=4 CFG_PARALLEL_SIZE=2 bash run_server.sh
#   HSDP over 4 GPUs:
#     USE_HSDP=1 HSDP_SHARD_SIZE=4 bash run_server.sh
#   Single-card memory-saving:
#     ENABLE_CPU_OFFLOAD=1 ENABLE_LAYERWISE_OFFLOAD=1 VAE_USE_SLICING=1 VAE_USE_TILING=1 bash run_server.sh

echo "Starting Wan2.2 server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Boundary ratio: $BOUNDARY_RATIO"
echo "Flow shift: $FLOW_SHIFT"
echo "Cache backend: $CACHE_BACKEND"
echo "Ulysses degree: $ULYSSES_DEGREE"
echo "Ring degree: $RING_DEGREE"
echo "CFG parallel size: $CFG_PARALLEL_SIZE"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "VAE patch parallel size: $VAE_PATCH_PARALLEL_SIZE"
echo "Use HSDP: $USE_HSDP"
if [ -n "$HSDP_SHARD_SIZE" ]; then
    echo "HSDP shard size: $HSDP_SHARD_SIZE"
fi
echo "CPU offload: $ENABLE_CPU_OFFLOAD"
echo "Layerwise offload: $ENABLE_LAYERWISE_OFFLOAD"
echo "VAE slicing: $VAE_USE_SLICING"
echo "VAE tiling: $VAE_USE_TILING"
if [ "$ENABLE_CACHE_DIT_SUMMARY" != "0" ]; then
    echo "Cache-DiT summary: enabled"
fi

CACHE_BACKEND_FLAG=""
if [ "$CACHE_BACKEND" != "none" ]; then
    CACHE_BACKEND_FLAG="--cache-backend $CACHE_BACKEND"
fi

EXTRA_ARGS=(
    --port "$PORT"
    --boundary-ratio "$BOUNDARY_RATIO"
    --flow-shift "$FLOW_SHIFT"
    --ulysses-degree "$ULYSSES_DEGREE"
    --ring-degree "$RING_DEGREE"
    --cfg-parallel-size "$CFG_PARALLEL_SIZE"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --vae-patch-parallel-size "$VAE_PATCH_PARALLEL_SIZE"
)

if [ "$CACHE_BACKEND" != "none" ]; then
    EXTRA_ARGS+=(--cache-backend "$CACHE_BACKEND")
fi
if [ "$ENABLE_CACHE_DIT_SUMMARY" != "0" ]; then
    EXTRA_ARGS+=(--enable-cache-dit-summary)
fi
if [ "$USE_HSDP" != "0" ]; then
    EXTRA_ARGS+=(--use-hsdp)
fi
if [ -n "$HSDP_SHARD_SIZE" ]; then
    EXTRA_ARGS+=(--hsdp-shard-size "$HSDP_SHARD_SIZE")
fi
if [ "$ENABLE_CPU_OFFLOAD" != "0" ]; then
    EXTRA_ARGS+=(--enable-cpu-offload)
fi
if [ "$ENABLE_LAYERWISE_OFFLOAD" != "0" ]; then
    EXTRA_ARGS+=(--enable-layerwise-offload)
fi
if [ "$VAE_USE_SLICING" != "0" ]; then
    EXTRA_ARGS+=(--vae-use-slicing)
fi
if [ "$VAE_USE_TILING" != "0" ]; then
    EXTRA_ARGS+=(--vae-use-tiling)
fi

vllm serve "$MODEL" --omni "${EXTRA_ARGS[@]}"
