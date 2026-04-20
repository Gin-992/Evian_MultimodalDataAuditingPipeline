from dataclasses import dataclass
import os


@dataclass(frozen=True)
class APIConfig:
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    vlm_base_url: str = os.getenv("VLM_BASE_URL", "http://localhost:8001/v1")
    api_key: str = os.getenv("OPENAI_API_KEY", "vllm")

    llm_model_name: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ")
    vlm_model_name: str = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")

    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    vlm_max_tokens: int = int(os.getenv("VLM_MAX_TOKENS", "512"))

    temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0"))
    top_p: float = float(os.getenv("MODEL_TOP_P", "1.0"))

    max_retries: int = int(os.getenv("MODEL_MAX_RETRIES", "3"))
    retry_delay: int = int(os.getenv("MODEL_RETRY_DELAY", "5"))