# EVIAN: Explainable Visual Instruction-Tuning Data Auditing

Official implementation of **EVIAN: Towards Explainable Visual Instruction-Tuning Data Auditing**.

EVIAN is a data auditing pipeline for visual instruction-tuning datasets. Given an image-instruction-response sample, EVIAN decomposes the response into visually grounded descriptions, subjective or logical inferences, and external factual claims. It then evaluates the sample from three complementary perspectives: **Image-Text Consistency**, **Logical Coherence**, and **Factual Accuracy**.

The resulting scores can be used to rank, filter, and select high-quality visual instruction-tuning data.

## Highlights

* Fine-grained auditing for image-instruction-response data.
* Explainable response decomposition into visual, inferential, and factual components.
* Three-stage decomposition pipeline: semantic tagging, visual distillation, and fluent visual synthesis.
* Multi-dimensional scoring with OpenAI-compatible LLM and VLM endpoints.
* Composite quality score for dataset ranking and top-k sample selection.
* Baselines including random sampling, CLIP ranking, LAVIS-based scoring, and VLM-judge scoring.
* Ablation scripts for analyzing the contribution of decomposition and individual score dimensions.

## Repository Structure

```text
.
|-- src/
|   |-- configs/            # Prompts and pipeline/API configuration
|   |-- mm_pipeline/        # Core EVIAN implementation
|   |-- scripts/            # CLI entry points for each pipeline stage
|   `-- run_pipeline.sh     # End-to-end local vLLM launcher
|-- baseline/               # Random, CLIP, LAVIS, and VLM-judge baselines
|-- ablation/               # Ablation runner and aggregation scripts
|-- requirements.txt        # Experiment environment snapshot
`-- README.md
```

## Model Services

Set the LLM and VLM service endpoints before running the pipeline:

```bash
export LLM_BASE_URL=http://localhost:8000/v1
export VLM_BASE_URL=http://localhost:8001/v1
export OPENAI_API_KEY=vllm
export LLM_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
export VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct-AWQ
```

## Pipeline Overview

EVIAN follows a decomposition-then-evaluation workflow:

1. **Semantic tagging**
   The response is tagged into visually grounded descriptions, subjective or logical inferences, and external factual claims.

2. **Visual distillation**
   Non-visual or weakly grounded content is removed or rewritten to obtain a cleaner visual response.

3. **Fluent visual synthesis**
   The distilled visual content is rewritten into a fluent visual-only summary.

4. **Multi-dimensional scoring**
   The sample is scored along three dimensions:

   * **Image-Text Consistency**
   * **Logical Coherence**
   * **Factual Accuracy**

5. **Dataset ranking and selection**
   The final composite score is used to rank the dataset and select high-quality samples.

## Step-by-Step Usage

Run the following commands from the `src` directory:

```bash
cd src
export PYTHONPATH=$PWD
```

### 1. Generate synthetic low-quality responses

```bash
python scripts/generate_low_quality.py \
  --input_file /path/to/original_dataset.json \
  --output_file /path/to/low_quality.json \
  --num_samples 250000 \
  --batch_size 128
```

### 2. Combine high-quality and generated low-quality data

```bash
python scripts/prepare_dataset.py \
  --original_data /path/to/high_quality.json \
  --low_quality_data /path/to/low_quality.json \
  --output_file /path/to/combined_data.json \
  --num_original_samples 50000 \
  --seed 42
```

### 3. Score the combined dataset

```bash
python scripts/score_dataset.py \
  --input_json /path/to/combined_data.json \
  --img_dir /path/to/images \
  --output_json /path/to/scored_data.json \
  --batch_size 128
```

### 4. Select top-ranked samples

```bash
python scripts/aggregate_results.py \
  --input_file /path/to/scored_data.json \
  --output_file /path/to/top_dataset.json \
  --top_n 10000
```

## Output Fields

Each scored sample contains the original data fields together with EVIAN audit metadata:

| Field                                      | Description                                                                           |
| ------------------------------------------ | ------------------------------------------------------------------------------------- |
| `step1_marked_response`                    | Response with `<INFER>` and `<KNOW>` spans.                                           |
| `step2_cleaned_response`                   | Response after removing or rewriting non-visual content.                              |
| `final_visual_summary`                     | Visual-only summary used for image-text consistency scoring.                          |
| `visual_consistency_score_str`             | VLM judgment for Image-Text Consistency.                                              |
| `inference_correctness_score_str`          | VLM judgment for Logical Coherence.                                                   |
| `external_knowledge_correctness_score_str` | VLM judgment for Factual Accuracy.                                                    |
| `composite_score`                          | Average score across Image-Text Consistency, Logical Coherence, and Factual Accuracy. |

## Baselines

Run baseline commands from the repository root.

### Random Sampling

```bash
python baseline/random_sampling.py \
  --json_path /path/to/data.json \
  --output_json /path/to/random_subset.json \
  --sample_size 10000 \
  --seed 42
```

### CLIP Ranking

```bash
python baseline/clip_ranker.py \
  --json_path /path/to/data.json \
  --image_dir /path/to/images \
  --output_json /path/to/clip_scores.json \
  --top_output_json /path/to/clip_top.json \
  --top_n 10000
```

### VLM-Judge Baseline

```bash
INPUT_JSON=/path/to/data.json \
IMAGE_DIR=/path/to/images \
OUTPUT_DIR=/path/to/baseline_outputs \
bash baseline/launch.sh
```

### LAVIS Baselines

```bash
JSON_PATH=/path/to/data.json \
IMAGE_DIR=/path/to/images \
OUTPUT_DIR=/path/to/lavis_outputs \
bash baseline/run_lavis.sh
```

## Ablations

Run a single ablation mode from the `ablation` directory:

```bash
cd ablation
export PYTHONPATH=$PWD

python run_ablation.py \
  --mode sa \
  --input_json /path/to/combined_data.json \
  --img_dir /path/to/images \
  --output_json /path/to/scored_ablation_sa.json \
  --batch_size 128 \
  --num_workers 128
```

Available modes:

| Mode    | Description                                        |
| ------- | -------------------------------------------------- |
| `phase` | Score the original response without decomposition. |
| `sa_sb` | Use only Image-Text Consistency.                   |
| `sb`    | Remove Factual Accuracy.                           |
| `sa`    | Remove Logical Coherence.                          |
| `full`  | Use all EVIAN score dimensions.                    |

Aggregate ablation results:

```bash
python aggregate_results.py \
  --input_file /path/to/scored_ablation_sa.json \
  --output_file /path/to/top_ablation_sa.json \
  --top_n 10000
```

To run all ablation modes with local vLLM services, return to the repository root and run:

```bash
cd /path/to/Evian_MultimodalDataAuditingPipeline
bash ablation/launch_job_ablation.sh
```

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{jia-etal-2026-evian,
  title = "{Evian}: Towards Explainable Visual Instruction-tuning Data Auditing",
  author = "Jia, Zimu and Xu, Mingjie and Estornell, Andrew and Wei, Jiaheng",
  editor = "Liakata, Maria and Moreira, Viviane P. and Zhang, Jiajun and Jurgens, David",
  booktitle = "Findings of the Association for Computational Linguistics: ACL 2026",
  month = jul,
  year = "2026",
  address = "San Diego, California, United States",
  publisher = "Association for Computational Linguistics",
  pages = "6272--6291",
  url = "https://aclanthology.org/2026.findings-acl.311/",
  isbn = "979-8-89176-395-1"
}
```
