from dataclasses import dataclass
from typing import Optional


# =========================
# 通用数据配置
# =========================
@dataclass
class DataConfig:
    json_path: str = ""
    image_dir: str = ""
    output_json: str = ""
    top_output_json: Optional[str] = None
    top_n: Optional[int] = None


# =========================
# CLIP 配置
# =========================
@dataclass
class ClipConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "cuda"


# =========================
# LAVIS 配置
# =========================
@dataclass
class LavisConfig:
    model_name: str = "blip_feature_extractor"
    device: str = "cuda"


# =========================
# 随机采样配置
# =========================
@dataclass
class SamplingConfig:
    sample_size: int = 50000
    seed: Optional[int] = None
    output_image_dir: Optional[str] = None


# =========================
# VLM 打分配置
# =========================
@dataclass
class ScoreConfig:
    top_k: int = 10000
    batch_size: int = 128
    save_every_n_batches: int = 5
    verbose: bool = True
    output_dir: str = "./outputs"
    output_full_scored_name: str = "all_scored_samples.json"
    output_top_clean_name: str = "top_10k_clean.json"
    output_top_eval_name: str = "top_10k_eval.json"


# =========================
# 服务配置
# =========================
@dataclass
class ServiceConfig:
    vlm_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    llm_model_name: str = "Qwen/Qwen2.5-32B-Instruct"
    vlm_base_url: str = "http://127.0.0.1:8001/v1"
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "vllm"
    timeout: int = 500