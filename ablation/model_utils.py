from __future__ import annotations

import base64
import mimetypes
import os
import time

import openai

LLM_MODEL_NAME = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct")
VLM_MODEL_NAME = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

LLM_PORT = os.getenv("LLM_PORT", "8000")
VLM_PORT = os.getenv("VLM_PORT", "8001")

llm_client = openai.OpenAI(
    base_url=f"http://localhost:{LLM_PORT}/v1",
    api_key="vllm",
)

vlm_client = openai.OpenAI(
    base_url=f"http://localhost:{VLM_PORT}/v1",
    api_key="vllm",
)


def _encode_image_to_base64(image_path: str) -> str:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def query_llm(
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> str:
    for attempt in range(max_retries):
        try:
            response = llm_client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLM] request failed ({attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return "LLM_GENERATION_FAILED"


def query_vlm(
    prompt: str,
    image_path: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> str:
    if not image_path or not os.path.isfile(image_path):
        return "Score: 0\nExplanation: Image path is invalid or file not found."

    try:
        image_url = _encode_image_to_base64(image_path)
    except Exception as e:
        return f"Score: 0\nExplanation: Failed to read/encode image file: {e}"

    for attempt in range(max_retries):
        try:
            response = vlm_client.chat.completions.create(
                model=VLM_MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[VLM] request failed for {image_path} ({attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return "Score: 0\nExplanation: Runtime error during VLM API request after retries."