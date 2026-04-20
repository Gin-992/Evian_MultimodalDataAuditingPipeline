#!/bin/bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${CODE_DIR}/final_results"

export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

# 可按需覆盖
export LLM_MODEL="${LLM_MODEL:-/root/to/Qwen2.5-32B-Instruct-AWQ}"
export VLM_MODEL="${VLM_MODEL:-/root/to/Qwen2.5-VL-7B-Instruct-AWQ}"
export LLM_PORT="${LLM_PORT:-8000}"
export VLM_PORT="${VLM_PORT:-8001}"
export LLM_GPU="${LLM_GPU:-0}"
export VLM_GPU="${VLM_GPU:-1}"

mkdir -p "${OUTPUT_DIR}"

cleanup() {
    echo "Cleaning up background processes..."
    if [ -n "${LLM_PID:-}" ] && ps -p "${LLM_PID}" > /dev/null 2>&1; then
        kill "${LLM_PID}" || true
    fi
    if [ -n "${VLM_PID:-}" ] && ps -p "${VLM_PID}" > /dev/null 2>&1; then
        kill "${VLM_PID}" || true
    fi
    sleep 3
    echo "Cleanup done."
}
trap cleanup EXIT

LLM_LOG="${OUTPUT_DIR}/llm_server.log"
VLM_LOG="${OUTPUT_DIR}/vlm_server.log"

echo "============================================================"
echo "Launch LLM server"
echo "============================================================"
CUDA_VISIBLE_DEVICES="${LLM_GPU}" vllm serve "${LLM_MODEL}" \
    --max-model-len 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --port "${LLM_PORT}" > "${LLM_LOG}" 2>&1 &

LLM_PID=$!
sleep 10

if ! ps -p "${LLM_PID}" > /dev/null 2>&1; then
    echo "LLM server failed to start."
    cat "${LLM_LOG}"
    exit 1
fi

until curl --output /dev/null --silent --fail "http://localhost:${LLM_PORT}/health"; do
    printf '.'
    sleep 5
    if ! ps -p "${LLM_PID}" > /dev/null 2>&1; then
        echo
        echo "LLM server died unexpectedly."
        cat "${LLM_LOG}"
        exit 1
    fi
done
echo
echo "LLM server ready."

echo "============================================================"
echo "Launch VLM server"
echo "============================================================"
CUDA_VISIBLE_DEVICES="${VLM_GPU}" vllm serve "${VLM_MODEL}" \
    --max-model-len 12288 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --port "${VLM_PORT}" > "${VLM_LOG}" 2>&1 &

VLM_PID=$!
sleep 10

if ! ps -p "${VLM_PID}" > /dev/null 2>&1; then
    echo "VLM server failed to start."
    cat "${VLM_LOG}"
    exit 1
fi

until curl --output /dev/null --silent --fail "http://localhost:${VLM_PORT}/health"; do
    printf '.'
    sleep 5
    if ! ps -p "${VLM_PID}" > /dev/null 2>&1; then
        echo
        echo "VLM server died unexpectedly."
        cat "${VLM_LOG}"
        exit 1
    fi
done
echo
echo "VLM server ready."

run_one_mode() {
    local mode="$1"
    local scored_json="${OUTPUT_DIR}/scored_data_ablation_${mode}.json"
    local top_json="${OUTPUT_DIR}/top_10000_ablation_${mode}.json"

    echo "============================================================"
    echo "Running mode: ${mode}"
    echo "============================================================"

    python3 "${CODE_DIR}/run_ablation.py" \
        --mode "${mode}" \
        --output_json "${scored_json}" \
        --batch_size 128 \
        --num_workers 128

    python3 "${CODE_DIR}/aggregate_results.py" \
        --input_file "${scored_json}" \
        --output_file "${top_json}" \
        --top_n 10000

    echo "Finished mode: ${mode}"
    echo "Scored file: ${scored_json}"
    echo "Top file   : ${top_json}"
}

run_one_mode "phase"
run_one_mode "sa_sb"
run_one_mode "sb"
run_one_mode "sa"

echo "All ablation jobs finished."