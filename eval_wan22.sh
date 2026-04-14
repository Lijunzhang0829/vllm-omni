#!/usr/bin/env bash
set -euo pipefail

# Wan2.2 server launch examples for this script.
#
# Notes:
# - This script only launches the client benchmark. Start the server manually first.
# - Wan2.2 validation is intentionally scoped to the old v0.16 squashed topology:
#   dispatcher on :8080 plus a single managed backend on :8091.
# - The supported backend config is fixed to:
#     --usp 8
#     --enable-layerwise-offload
#     --boundary-ratio 0.875
#     --vae-use-slicing
#     --vae-use-tiling
#
# Baseline via dispatcher with a single managed backend:
# env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
# VLLM_OMNI_ENABLE_DIFFUSION_SERVER_SCHEDULING=0 \
# VLLM_OMNI_ENABLE_DIFFUSION_PREEMPTION=0 \
# python3 benchmarks/diffusion/super_p95_dispatcher.py \
#   --host 0.0.0.0 \
#   --port 8080 \
#   --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
#   --num-servers 1 \
#   --backend-start-port 8091 \
#   --device-env-var ASCEND_RT_VISIBLE_DEVICES \
#   --backend-hardware-profiles 910B3 \
#   --backend-log-dir /tmp/super_p95_backends_wan22_baseline \
#   --request-timeout-s 1000000 \
#   --backend-args=--omni \
#   --backend-args=--usp \
#   --backend-args=8 \
#   --backend-args=--enable-layerwise-offload \
#   --backend-args=--boundary-ratio \
#   --backend-args=0.875 \
#   --backend-args=--vae-use-slicing \
#   --backend-args=--vae-use-tiling
#
# super_p95 via dispatcher with a single managed backend:
# env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
# VLLM_OMNI_ENABLE_DIFFUSION_SERVER_SCHEDULING=1 \
# VLLM_OMNI_ENABLE_DIFFUSION_PREEMPTION=1 \
# python3 benchmarks/diffusion/super_p95_dispatcher.py \
#   --host 0.0.0.0 \
#   --port 8080 \
#   --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
#   --num-servers 1 \
#   --backend-start-port 8091 \
#   --device-env-var ASCEND_RT_VISIBLE_DEVICES \
#   --backend-hardware-profiles 910B3 \
#   --backend-log-dir /tmp/super_p95_backends_wan22 \
#   --request-timeout-s 1000000 \
#   --sacrificial-load-factor 0.1 \
#   --backend-args=--omni \
#   --backend-args=--usp \
#   --backend-args=8 \
#   --backend-args=--enable-layerwise-offload \
#   --backend-args=--boundary-ratio \
#   --backend-args=0.875 \
#   --backend-args=--vae-use-slicing \
#   --backend-args=--vae-use-tiling

usage() {
  cat <<'EOF'
Usage:
  bash eval_wan22.sh baseline
  bash eval_wan22.sh super_p95

Description:
  Run the Wan2.2 client-side request-rate sweep against an already running server.
  This script does not launch the server. Start the matching baseline or super_p95
  server manually first, then run this script.

Environment overrides:
  BASE_URL            Default: http://127.0.0.1:8080
  MODEL               Default: Wan-AI/Wan2.2-T2V-A14B-Diffusers
  NUM_PROMPTS         Default: 160
  MAX_CONCURRENCY     Default: 1000
  REQUEST_RATES       Default: "0.012 0.020"
  WARMUP_REQUESTS     Default: 0
  SEED                Default: 0
  RANDOM_REQUEST_SEED Default: 8
  ARRIVAL_SEED        Default: 456
  CLIENT_TIMEOUT_S    Default: 1000000
  OUTPUT_ROOT         Default: benchmarks/diffusion/results/wan22_p95_sweep
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

RUN_LABEL="$1"
case "${RUN_LABEL}" in
  baseline|super_p95)
    ;;
  *)
    echo "Invalid mode: ${RUN_LABEL}" >&2
    usage
    exit 1
    ;;
esac

BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
NUM_PROMPTS="${NUM_PROMPTS:-160}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1000}"
REQUEST_RATES="${REQUEST_RATES:-0.012 0.020}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-0}"
SEED="${SEED:-0}"
RANDOM_REQUEST_SEED="${RANDOM_REQUEST_SEED:-8}"
ARRIVAL_SEED="${ARRIVAL_SEED:-456}"
CLIENT_TIMEOUT_S="${CLIENT_TIMEOUT_S:-1000000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmarks/diffusion/results/wan22_p95_sweep}"

DEFAULT_WAN22_RANDOM3="$(cat <<'EOF'
[
  {"width":854,"height":480,"num_inference_steps":3,"num_frames":80,"fps":16,"flow_shift":12.0,"boundary_ratio":0.875,"weight":0.15},
  {"width":854,"height":480,"num_inference_steps":4,"num_frames":120,"fps":24,"flow_shift":12.0,"boundary_ratio":0.875,"weight":0.25},
  {"width":1280,"height":720,"num_inference_steps":6,"num_frames":80,"fps":16,"flow_shift":5.0,"boundary_ratio":0.875,"weight":0.60}
]
EOF
)"
WAN22_RANDOM3="${WAN22_RANDOM3:-$DEFAULT_WAN22_RANDOM3}"

OUT_DIR="${OUTPUT_ROOT}/${RUN_LABEL}"
mkdir -p "${OUT_DIR}"

echo "Run label       : ${RUN_LABEL}"
echo "Base URL        : ${BASE_URL}"
echo "Model           : ${MODEL}"
echo "Num prompts     : ${NUM_PROMPTS}"
echo "Max concurrency : ${MAX_CONCURRENCY}"
echo "Request rates   : ${REQUEST_RATES}"
echo "Random seed     : ${RANDOM_REQUEST_SEED}"
echo "Arrival seed    : ${ARRIVAL_SEED}"
echo "Client timeout  : ${CLIENT_TIMEOUT_S}"
echo "Output dir      : ${OUT_DIR}"

python - <<'PY' <<<"${WAN22_RANDOM3}" >/dev/null
import json
import sys
json.loads(sys.stdin.read())
PY

HEALTH_CODE="$(curl --noproxy '*' -s -o /tmp/eval_wan22_health.json -w '%{http_code}' "${BASE_URL}/health" || true)"
if [[ "${HEALTH_CODE}" != "200" ]]; then
  echo "Server health check failed for ${BASE_URL}/health (status=${HEALTH_CODE})." >&2
  cat /tmp/eval_wan22_health.json 2>/dev/null || true
  exit 1
fi

for rate in ${REQUEST_RATES}; do
  output_file="${OUT_DIR}/rate_${rate}.json"
  echo
  echo "=== rate=${rate} -> ${output_file} ==="
  env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
    --base-url "${BASE_URL}" \
    --model "${MODEL}" \
    --backend v1/videos \
    --dataset random \
    --task t2v \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --request-rate "${rate}" \
    --warmup-requests "${WARMUP_REQUESTS}" \
    --seed "${SEED}" \
    --random-request-seed "${RANDOM_REQUEST_SEED}" \
    --arrival-seed "${ARRIVAL_SEED}" \
    --client-timeout-s "${CLIENT_TIMEOUT_S}" \
    --random-request-config "${WAN22_RANDOM3}" \
    --output-file "${output_file}"
done

echo
echo "Finished. Results written to ${OUT_DIR}"
