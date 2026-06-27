# EVIAN: Explainable Visual Instruction-tuning Data Auditing

Official implementation of **EVIAN: Towards Explainable Visual Instruction-tuning Data Auditing**.

EVIAN audits visual instruction-tuning data with a decomposition-then-evaluation pipeline. It separates each response into visual description, subjective inference, and factual claim components, then evaluates them along three dimensions: Image-Text Consistency, Logical Coherence, and Factual Accuracy.

## Highlights

- Fine-grained auditing for image-instruction-response data.
- Three-stage response decomposition: semantic tagging, visual distillation, and fluent visual synthesis.
- Multi-dimensional scoring with OpenAI-compatible LLM and VLM endpoints.
- Dataset ranking and top-k selection through a composite quality score.
- Baseline scripts for random sampling, CLIP, LAVIS, and VLM-judge scoring.
- Ablation scripts for evaluating the contribution of decomposition and score dimensions.

## Repository Structure

```text
.
|-- src/
|   |-- configs/            # Prompts and pipeline/API configuration
|   |-- mm_pipeline/        # Core EVIAN implementation
|   |-- scripts/            # CLI entry points for each pipeline stage
|   `-- run_pipeline.sh     # End-to-end pipeline launcher
|-- baseline/               # Random, CLIP, LAVIS, and VLM-judge baselines
|-- ablation/               # Ablation runner and aggregation scripts
|-- requirements.txt        # Full experiment environment snapshot
`-- README.md
```

## Installation

Create a Python environment and install dependencies:

```bash
git clone <repo-url>
cd Evian_MultimodalDataAuditingPipeline
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

The provided `requirements.txt` is a full GPU experiment snapshot and includes CUDA, vLLM, and model-serving packages. For a lighter environment, install only the packages needed by the scripts you run, such as `openai`, `tqdm`, `pillow`, `torch`, `transformers`, and `vllm`. LAVIS baselines require the LAVIS package in addition to the core dependencies.

## Data Format

Input files use a LLaVA-style JSON list:

```json
[
  {
    "id": "sample_000001",
    "image": "relative/path/to/image.jpg",
    "conversations": [
      {"from": "human", "value": "Describe the image."},
      {"from": "gpt", "value": "A person is standing beside a red car."}
    ]
  }
]
```

`image` is resolved relative to the image directory passed to the scoring scripts.

## Model Services

EVIAN expects OpenAI-compatible chat-completion endpoints. The experiments use an LLM for text decomposition and a VLM for image-conditioned scoring.

```bash
export LLM_BASE_URL=http://localhost:8000/v1
export VLM_BASE_URL=http://localhost:8001/v1
export OPENAI_API_KEY=vllm
export LLM_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
export VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct-AWQ
```

You can serve local models with vLLM or point these variables to compatible hosted endpoints.

## End-to-End Usage

Set paths and launch the full pipeline:

```bash
IMG_DIR=/path/to/images \
GOOD_DATA_JSON=/path/to/high_quality.json \
ORIGINAL_DATA_JSON=/path/to/original_dataset.json \
FINAL_OUTPUT_DIR=/path/to/outputs \
bash src/run_pipeline.sh
```

The script generates low-quality samples, mixes them with high-quality samples, scores the combined dataset, and exports the top-ranked subset.

## Step-by-Step Usage

Run from the `src` directory so that `mm_pipeline` and `configs` are importable:

```bash
cd src
export PYTHONPATH=$PWD
```

Generate synthetic low-quality responses:

```bash
python scripts/generate_low_quality.py \
  --input_file /path/to/original_dataset.json \
  --output_file /path/to/low_quality.json \
  --num_samples 250000 \
  --batch_size 128
```

Combine high-quality and generated low-quality data:

```bash
python scripts/prepare_dataset.py \
  --original_data /path/to/high_quality.json \
  --low_quality_data /path/to/low_quality.json \
  --output_file /path/to/combined_data.json \
  --num_original_samples 50000 \
  --seed 42
```

Score the combined dataset:

```bash
python scripts/score_dataset.py \
  --input_json /path/to/combined_data.json \
  --img_dir /path/to/images \
  --output_json /path/to/scored_data.json \
  --batch_size 128
```

Select the top-ranked samples:

```bash
python scripts/aggregate_results.py \
  --input_file /path/to/scored_data.json \
  --output_file /path/to/top_dataset.json \
  --top_n 10000
```

## Output Fields

Scored samples include the original data fields plus audit metadata:

- `step1_marked_response`: response with `<INFER>` and `<KNOW>` spans.
- `step2_cleaned_response`: response after removing or rewriting non-visual content.
- `final_visual_summary`: visual-only summary used for image-text consistency scoring.
- `visual_consistency_score_str`: VLM judgment for Image-Text Consistency.
- `inference_correctness_score_str`: VLM judgment for Logical Coherence.
- `external_knowledge_correctness_score_str`: VLM/LLM judgment for Factual Accuracy.
- `composite_score`: average of the enabled score dimensions.

## Baselines

Random sampling:

```bash
python baseline/random_sampling.py \
  --json_path /path/to/data.json \
  --output_json /path/to/random_subset.json \
  --sample_size 10000 \
  --seed 42
```

CLIP ranking:

```bash
python baseline/clip_ranker.py \
  --json_path /path/to/data.json \
  --image_dir /path/to/images \
  --output_json /path/to/clip_scores.json \
  --top_output_json /path/to/clip_top.json \
  --top_n 10000
```

VLM-judge baseline:

```bash
INPUT_JSON=/path/to/data.json \
IMAGE_DIR=/path/to/images \
OUTPUT_DIR=/path/to/baseline_outputs \
bash baseline/launch.sh
```

LAVIS baselines:

```bash
JSON_PATH=/path/to/data.json \
IMAGE_DIR=/path/to/images \
OUTPUT_DIR=/path/to/lavis_outputs \
bash baseline/run_lavis.sh
```

## Ablations

Run a single ablation mode:

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

- `phase`: score the original response without decomposition.
- `sa_sb`: use only Image-Text Consistency.
- `sb`: remove Factual Accuracy.
- `sa`: remove Logical Coherence.
- `full`: use all EVIAN score dimensions.

Aggregate ablation results:

```bash
python aggregate_results.py \
  --input_file /path/to/scored_ablation_sa.json \
  --output_file /path/to/top_ablation_sa.json \
  --top_n 10000
```

## Reproducibility Notes

- Set `OPENAI_API_KEY`, `LLM_BASE_URL`, `VLM_BASE_URL`, `LLM_MODEL`, and `VLM_MODEL` before running API-based stages.
- Use absolute paths for datasets and image directories in shell scripts.
- `batch_size` controls both batch size and request concurrency in the main pipeline.
- Intermediate JSON and log files are ignored by Git through `.gitignore`.
- Large model weights, generated datasets, and experiment outputs should be released separately from the source code.

## Citation

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
