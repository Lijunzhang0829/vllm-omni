#!/usr/bin/env bash
set -euo pipefail

# Qwen-Image server launch examples for this script.
#
# Notes:
# - This script only launches the client benchmark. Start the server manually first.
# - Use the v0.18.0-super-p95 dispatcher for both baseline and super_p95 so the
#   serving topology stays identical.
# - TODO(super-p95-v018): Once diffusion_benchmark_serving.py regains
#   client-timeout / arrival-seed / random-request-seed parity, thread those
#   options through this script instead of relying on the benchmark defaults.
#
# Baseline, 8x 910B3:
# env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
# python3 benchmarks/diffusion/super_p95_dispatcher.py \
#   --host 127.0.0.1 \
#   --port 8080 \
#   --backend-url http://127.0.0.1:8091 \
#   --backend-url http://127.0.0.1:8092 \
#   --backend-url http://127.0.0.1:8093 \
#   --backend-url http://127.0.0.1:8094 \
#   --backend-url http://127.0.0.1:8095 \
#   --backend-url http://127.0.0.1:8096 \
#   --backend-url http://127.0.0.1:8097 \
#   --backend-url http://127.0.0.1:8098 \
#   --hardware-profile 910B3
#
# Super-P95, 8x 910B3:
# env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
# python3 benchmarks/diffusion/super_p95_dispatcher.py \
#   --host 127.0.0.1 \
#   --port 8080 \
#   --backend-url http://127.0.0.1:8091 \
#   --backend-url http://127.0.0.1:8092 \
#   --backend-url http://127.0.0.1:8093 \
#   --backend-url http://127.0.0.1:8094 \
#   --backend-url http://127.0.0.1:8095 \
#   --backend-url http://127.0.0.1:8096 \
#   --backend-url http://127.0.0.1:8097 \
#   --backend-url http://127.0.0.1:8098 \
#   --hardware-profile 910B3

usage() {
  cat <<'EOF'
Usage:
  bash eval_qwen_image.sh baseline
  bash eval_qwen_image.sh super_p95

Description:
  Run the Qwen-Image client-side request-rate sweep against an already running server.
  This script does not launch the server. Start the matching baseline or super_p95
  server manually first, then run this script.

Environment overrides:
  BASE_URL          Default: http://127.0.0.1:8080
  MODEL             Default: Qwen/Qwen-Image
  NUM_PROMPTS       Default: 500
  MAX_CONCURRENCY   Default: 1000
  REQUEST_RATES     Default: "0.2 0.3 0.4 0.5 0.6 0.7 0.8"
  WARMUP_REQUESTS   Default: 1
  WARMUP_STEPS      Default: 1
  SEED              Default: 0
  OUTPUT_ROOT       Default: benchmarks/diffusion/results/qwen_image_p95_sweep
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
MODEL="${MODEL:-Qwen/Qwen-Image}"
NUM_PROMPTS="${NUM_PROMPTS:-500}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1000}"
REQUEST_RATES="${REQUEST_RATES:-0.2 0.3 0.4 0.5 0.6 0.7 0.8}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmarks/diffusion/results/qwen_image_p95_sweep}"

DEFAULT_QWEN_IMAGE_RANDOM4="$(cat <<'EOF'
[
  {"width":512,"height":512,"num_inference_steps":20,"weight":0.15},
  {"width":768,"height":768,"num_inference_steps":20,"weight":0.25},
  {"width":1024,"height":1024,"num_inference_steps":25,"weight":0.45},
  {"width":1536,"height":1536,"num_inference_steps":35,"weight":0.15}
]
EOF
)"
QWEN_IMAGE_RANDOM4="${QWEN_IMAGE_RANDOM4:-$DEFAULT_QWEN_IMAGE_RANDOM4}"

OUT_DIR="${OUTPUT_ROOT}/${RUN_LABEL}"
mkdir -p "${OUT_DIR}"

echo "Run label       : ${RUN_LABEL}"
echo "Base URL        : ${BASE_URL}"
echo "Model           : ${MODEL}"
echo "Num prompts     : ${NUM_PROMPTS}"
echo "Max concurrency : ${MAX_CONCURRENCY}"
echo "Request rates   : ${REQUEST_RATES}"
echo "Output dir      : ${OUT_DIR}"

python - <<'PY' <<<"${QWEN_IMAGE_RANDOM4}" >/dev/null
import json
import sys
json.loads(sys.stdin.read())
PY

HEALTH_CODE="$(curl --noproxy '*' -s -o /tmp/eval_qwen_image_health.json -w '%{http_code}' "${BASE_URL}/health" || true)"
if [[ "${HEALTH_CODE}" != "200" ]]; then
  echo "Server health check failed for ${BASE_URL}/health (status=${HEALTH_CODE})." >&2
  cat /tmp/eval_qwen_image_health.json 2>/dev/null || true
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
    --backend vllm-omni \
    --dataset random \
    --task t2i \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --request-rate "${rate}" \
    --warmup-requests "${WARMUP_REQUESTS}" \
    --warmup-num-inference-steps "${WARMUP_STEPS}" \
    --seed "${SEED}" \
    --random-request-config "${QWEN_IMAGE_RANDOM4}" \
    --output-file "${output_file}"
done

echo
echo "Finished. Results written to ${OUT_DIR}"
