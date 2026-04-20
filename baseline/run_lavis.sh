#!/usr/bin/env bash
set -euo pipefail

JSON_PATH=${JSON_PATH:-"/path/to/combined_data.json"}
IMAGE_DIR=${IMAGE_DIR:-"/path/to/images"}
OUTPUT_DIR=${OUTPUT_DIR:-"./lavis_outputs"}
DEVICE=${DEVICE:-"cuda"}
TOP_N=${TOP_N:-10000}

mkdir -p "${OUTPUT_DIR}"

python baseline/lavis_ranker.py \
  --model albef_feature_extractor \
  --json_path "${JSON_PATH}" \
  --image_dir "${IMAGE_DIR}" \
  --output_json "${OUTPUT_DIR}/albef_ranking.json" \
  --device "${DEVICE}" \
  --top_output_json "${OUTPUT_DIR}/albef_top.json" \
  --top_n "${TOP_N}"

python baseline/lavis_ranker.py \
  --model blip_feature_extractor \
  --json_path "${JSON_PATH}" \
  --image_dir "${IMAGE_DIR}" \
  --output_json "${OUTPUT_DIR}/blip_ranking.json" \
  --device "${DEVICE}" \
  --top_output_json "${OUTPUT_DIR}/blip_top.json" \
  --top_n "${TOP_N}"

python baseline/lavis_ranker.py \
  --model blip2_feature_extractor \
  --json_path "${JSON_PATH}" \
  --image_dir "${IMAGE_DIR}" \
  --output_json "${OUTPUT_DIR}/blip2_ranking.json" \
  --device "${DEVICE}" \
  --top_output_json "${OUTPUT_DIR}/blip2_top.json" \
  --top_n "${TOP_N}"

echo "Finished running all LAVIS baselines."