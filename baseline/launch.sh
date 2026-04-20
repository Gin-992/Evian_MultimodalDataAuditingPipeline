#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config
# =========================
PYTHON_SCRIPT=${PYTHON_SCRIPT:-"baseline/score_baseline.py"}
VLM_MODEL=${VLM_MODEL:-"/path/to/Qwen2.5-VL-7B-Instruct"}
INPUT_JSON=${INPUT_JSON:-"/path/to/combined_data.json"}
IMAGE_DIR=${IMAGE_DIR:-"/path/to/images"}
OUTPUT_DIR=${OUTPUT_DIR:-"./score_outputs"}
VLM_LOG_FILE="${OUTPUT_DIR}/vlm_server.log"

HOST=${HOST:-"127.0.0.1"}
PORT=${PORT:-"8001"}
CUDA_VISIBLE_DEVICES_VALUE=${CUDA_VISIBLE_DEVICES_VALUE:-"0,1"}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}

TOP_K=${TOP_K:-10000}
BATCH_SIZE=${BATCH_SIZE:-128}
SAVE_EVERY_N_BATCHES=${SAVE_EVERY_N_BATCHES:-5}

mkdir -p "${OUTPUT_DIR}"

VLM_PID=""

cleanup() {
  echo
  echo "================ Cleanup ================"
  if [[ -n "${VLM_PID}" ]] && ps -p "${VLM_PID}" > /dev/null 2>&1; then
    echo "Stopping VLM server PID=${VLM_PID}"
    kill "${VLM_PID}" || true
    sleep 5
    if ps -p "${VLM_PID}" > /dev/null 2>&1; then
      echo "Force killing VLM server PID=${VLM_PID}"
      kill -9 "${VLM_PID}" || true
    fi
  fi
  echo "Cleanup done."
}
trap cleanup EXIT INT TERM

echo "================ Launch VLM Server ================"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE} vllm serve "${VLM_MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --trust-remote-code \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  > "${VLM_LOG_FILE}" 2>&1 &

VLM_PID=$!
echo "VLM server PID: ${VLM_PID}"
sleep 15

if ! ps -p "${VLM_PID}" > /dev/null 2>&1; then
  echo "[CRITICAL ERROR] VLM server failed to start."
  cat "${VLM_LOG_FILE}"
  exit 1
fi

echo "Waiting for health endpoint..."
while true; do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 -m 10 "http://${HOST}:${PORT}/health" || echo "000")
  if [[ "${STATUS}" == "200" ]]; then
    echo "VLM server is ready."
    break
  fi

  printf "."
  sleep 5

  if ! ps -p "${VLM_PID}" > /dev/null 2>&1; then
    echo
    echo "[ERROR] VLM server died unexpectedly."
    cat "${VLM_LOG_FILE}"
    exit 1
  fi
done

echo
echo "================ Run Scoring Script ================"
export VLM_MODEL
export VLM_BASE_URL="http://${HOST}:${PORT}/v1"

python "${PYTHON_SCRIPT}" \
  --input_json "${INPUT_JSON}" \
  --image_dir "${IMAGE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --top_k "${TOP_K}" \
  --batch_size "${BATCH_SIZE}" \
  --save_every_n_batches "${SAVE_EVERY_N_BATCHES}" \
  --verbose

echo "Finished."