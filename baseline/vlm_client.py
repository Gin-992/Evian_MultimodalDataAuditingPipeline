import os
import time
import base64
import mimetypes
import asyncio
from typing import List

from openai import OpenAI, AsyncOpenAI


VLM_MODEL_NAME = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct")

VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://127.0.0.1:8001/v1")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.getenv("VLM_API_KEY", "vllm")
TIMEOUT = int(os.getenv("VLM_TIMEOUT", "500"))

llm_client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=API_KEY,
    timeout=TIMEOUT,
)

vlm_client = OpenAI(
    base_url=VLM_BASE_URL,
    api_key=API_KEY,
    timeout=TIMEOUT,
)

async_vlm_client = AsyncOpenAI(
    base_url=VLM_BASE_URL,
    api_key=API_KEY,
    timeout=TIMEOUT,
)


def encode_image_to_base64(image_path: str) -> str:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def query_vlm(prompt: str, image_path: str, max_retries: int = 3, retry_delay: int = 5) -> str:
    if not image_path or not os.path.isfile(image_path):
        return "Score: 0\nExplanation: Image path is invalid or file not found."

    try:
        image_url = encode_image_to_base64(image_path)
    except Exception as e:
        return f"Score: 0\nExplanation: Failed to encode image: {e}"

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
                temperature=0,
                top_p=1,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[WARN] VLM request failed ({attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    return "Score: 0\nExplanation: Runtime error during VLM API request after retries."


async def _query_vlm_once(prompt: str, image_path: str) -> str:
    try:
        image_url = encode_image_to_base64(image_path)
        response = await async_vlm_client.chat.completions.create(
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
            temperature=0,
            top_p=1,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[WARN] Async VLM request failed for {image_path}: {e}")
        return "Score: 0\nExplanation: Runtime error during async VLM API request."


async def query_vlm_batch(prompts: List[str], image_paths: List[str]) -> List[str]:
    if len(prompts) != len(image_paths):
        raise ValueError("prompts and image_paths must have the same length")

    tasks = []
    for prompt, image_path in zip(prompts, image_paths):
        if not image_path or not os.path.isfile(image_path):
            tasks.append(asyncio.sleep(0, result="Score: 0\nExplanation: Invalid image path."))
        else:
            tasks.append(_query_vlm_once(prompt, image_path))

    return await asyncio.gather(*tasks)


def query_llm(prompt: str, max_retries: int = 3, retry_delay: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            response = llm_client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                top_p=1,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[WARN] LLM request failed ({attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    return "LLM_GENERATION_FAILED"