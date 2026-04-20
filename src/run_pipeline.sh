#!/bin/bash
set -euo pipefail

# =========================
# User-configurable paths
# =========================
REPO_DIR="/root/to/your_repo"
IMG_DIR="/root/to/DataPool/images"

GOOD_DATA_JSON="/root/to/MM-Score-Top50K-clean.json"
ORIGINAL_DATA_JSON="/root/to/DataPool/sampled_mmds_anno.json"
LOW_QUALITY_DATA_JSON="/root/to/DataPool/generated/low_quality_250k.json"

FINAL_OUTPUT_DIR="/root/to/DataPool/final_results"
COMBINED_DATA_JSON="${FINAL_OUTPUT_DIR}/combined_data_300k.json"
SCORED_DATA_JSON="${FINAL_OUTPUT_DIR}/scored_data_300k.json"
FINAL_TOP_N_JSON="${FINAL_OUTPUT_DIR}/top_10000_dataset.json"

LLM_LOG_FILE="${FINAL_OUTPUT_DIR}/llm_server.log"
VLM_LOG_FILE="${FINAL_OUTPUT_DIR}/vlm_server.log"

# =========================
# Model / API config
# =========================
export LLM_MODEL="/root/to/Qwen2.5-32B-Instruct-AWQ"
export VLM_MODEL="/root/to/Qwen2.5-VL-7B-Instruct-AWQ"
export LLM_BASE_URL="http://localhost:8000/v1"
export VLM_BASE_URL="http://localhost:8001/v1"
export OPENAI_API_KEY="vllm"

cleanup() {
    echo "[Cleanup] stopping background services..."

    if [[ -n "${LLM_PID:-}" ]] && ps -p "${LLM_PID}" > /dev/null 2>&1; then
        kill "${LLM_PID}" || true
    fi

    if [[ -n "${VLM_PID:-}" ]] && ps -p "${VLM_PID}" > /dev/null 2>&1; then
        kill "${VLM_PID}" || true
    fi

    sleep 3
    echo "[Cleanup] done."
}
trap cleanup EXIT

mkdir -p "${FINAL_OUTPUT_DIR}"
mkdir -p "$(dirname "${LOW_QUALITY_DATA_JSON}")"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

echo "=================================================="
echo "Pipeline start: $(date)"
echo "=================================================="

echo "[1/5] Launching LLM server..."
CUDA_VISIBLE_DEVICES=0 vllm serve "${LLM_MODEL}" \
    --max-model-len 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --port 8000 > "${LLM_LOG_FILE}" 2>&1 &
LLM_PID=$!
sleep 30

echo "[2/5] Launching VLM server..."
CUDA_VISIBLE_DEVICES=1 vllm serve "${VLM_MODEL}" \
    --max-model-len 12288 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --port 8001 > "${VLM_LOG_FILE}" 2>&1 &
VLM_PID=$!
sleep 50

echo "[3/5] Generating low-quality data..."
python scripts/generate_low_quality.py \
    --input_file "${ORIGINAL_DATA_JSON}" \
    --output_file "${LOW_QUALITY_DATA_JSON}" \
    --batch_size 128 \
    --num_samples 250000

echo "[4/5] Preparing combined dataset..."
python scripts/prepare_dataset.py \
    --original_data "${GOOD_DATA_JSON}" \
    --low_quality_data "${LOW_QUALITY_DATA_JSON}" \
    --output_file "${COMBINED_DATA_JSON}" \
    --num_original_samples 50000 \
    --seed 42

echo "[5/5] Scoring and aggregating..."
python scripts/score_dataset.py \
    --input_json "${COMBINED_DATA_JSON}" \
    --img_dir "${IMG_DIR}" \
    --output_json "${SCORED_DATA_JSON}" \
    --batch_size 128

python scripts/aggregate_results.py \
    --input_file "${SCORED_DATA_JSON}" \
    --output_file "${FINAL_TOP_N_JSON}" \
    --top_n 10000

echo "=================================================="
echo "Pipeline finished successfully."
echo "Final dataset: ${FINAL_TOP_N_JSON}"
echo "End time: $(date)"
echo "=================================================="