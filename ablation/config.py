from __future__ import annotations

import os

# =========================
# 基础路径配置
# =========================
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_JSON = "/root/to/EVIAN_MutimodalDataAuditing/combined_data_350k_cleaned.json"
IMG_DIR = "/root/to/MMDS-DataPool/images"
OUTPUT_DIR = os.path.join(CODE_DIR, "final_results")

# =========================
# 模型服务配置
# =========================
LLM_MODEL = os.getenv("LLM_MODEL", "/root/to/Qwen2.5-32B-Instruct-AWQ")
VLM_MODEL = os.getenv("VLM_MODEL", "/root/to/Qwen2.5-VL-7B-Instruct-AWQ")

LLM_PORT = int(os.getenv("LLM_PORT", "8000"))
VLM_PORT = int(os.getenv("VLM_PORT", "8001"))

LLM_GPU = os.getenv("LLM_GPU", "0")
VLM_GPU = os.getenv("VLM_GPU", "1")

LLM_MAX_MODEL_LEN = 4096
VLM_MAX_MODEL_LEN = 12288
LLM_GPU_MEMORY_UTILIZATION = 0.90
VLM_GPU_MEMORY_UTILIZATION = 0.85

# =========================
# 运行配置
# =========================
DEFAULT_BATCH_SIZE = 128
DEFAULT_TOP_N = 10000
DEFAULT_NUM_WORKERS = 16

# =========================
# 消融模式
# phase  : 不做LLM三步分解，直接拿原response打分
# sa_sb  : 只保留visual score
# sb     : 去掉knowledge score
# sa     : 去掉inference score
# full   : 完整版（以后你想开源完整版时可直接打开）
# =========================
ABLATION_MODES = {
    "phase": {
        "use_decomposition": False,
        "use_inference_score": True,
        "use_knowledge_score": True,
        "output_name": "scored_data_ablation_phase.json",
        "top_name": "top_10000_ablation_phase.json",
    },
    "sa_sb": {
        "use_decomposition": True,
        "use_inference_score": False,
        "use_knowledge_score": False,
        "output_name": "scored_data_ablation_sa_sb.json",
        "top_name": "top_10000_ablation_sa_sb.json",
    },
    "sb": {
        "use_decomposition": True,
        "use_inference_score": True,
        "use_knowledge_score": False,
        "output_name": "scored_data_ablation_sb.json",
        "top_name": "top_10000_ablation_sb.json",
    },
    "sa": {
        "use_decomposition": True,
        "use_inference_score": False,
        "use_knowledge_score": True,
        "output_name": "scored_data_ablation_sa.json",
        "top_name": "top_10000_ablation_sa.json",
    },
    "full": {
        "use_decomposition": True,
        "use_inference_score": True,
        "use_knowledge_score": True,
        "output_name": "scored_data_full.json",
        "top_name": "top_10000_full.json",
    },
}