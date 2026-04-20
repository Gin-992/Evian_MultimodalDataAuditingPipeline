from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import openai
from tqdm import tqdm

from configs.api_config import APIConfig
from mm_pipeline.utils import encode_image_to_data_uri, extract_json_from_text, remove_optional_prefix


class ModelClients:
    def __init__(self, config: Optional[APIConfig] = None) -> None:
        self.config = config or APIConfig()

        self.llm_client = openai.OpenAI(
            base_url=self.config.llm_base_url,
            api_key=self.config.api_key,
        )
        self.vlm_client = openai.OpenAI(
            base_url=self.config.vlm_base_url,
            api_key=self.config.api_key,
        )

    def query_llm(self, prompt: str) -> str:
        for attempt in range(self.config.max_retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.config.llm_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.llm_max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception:
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    return "LLM_GENERATION_FAILED"
        return "LLM_GENERATION_FAILED"

    def query_vlm(self, prompt: str, image_path: str) -> str:
        try:
            image_url = encode_image_to_data_uri(image_path)
        except Exception as exc:
            return f"Score: 0\nExplanation: Failed to read image file: {exc}"

        for attempt in range(self.config.max_retries):
            try:
                response = self.vlm_client.chat.completions.create(
                    model=self.config.vlm_model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        }
                    ],
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.vlm_max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception:
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    return "Score: 0\nExplanation: Runtime error during VLM API request after retries."
        return "Score: 0\nExplanation: Runtime error during VLM API request after retries."

    def batch_query_llm(
        self,
        prompts: List[str],
        num_workers: int,
        output_prefix_to_remove: Optional[str] = None,
        expect_json: bool = False,
        desc: str = "Querying LLM",
    ) -> List[Any]:
        responses: List[Any] = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.query_llm, prompt): idx
                for idx, prompt in enumerate(prompts)
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                idx = futures[future]
                try:
                    text = future.result()
                    if expect_json:
                        responses[idx] = extract_json_from_text(text) or {"error": "invalid_json"}
                    else:
                        responses[idx] = remove_optional_prefix(text, output_prefix_to_remove)
                except Exception as exc:
                    responses[idx] = {"error": str(exc)} if expect_json else "LLM_GENERATION_FAILED"

        return responses

    def batch_query_vlm(
        self,
        prompts: List[str],
        image_paths: List[Optional[str]],
        num_workers: int,
        desc: str = "Querying VLM",
    ) -> List[str]:
        responses: List[Optional[str]] = [None] * len(prompts)
        valid_jobs = []

        for idx, (prompt, image_path) in enumerate(zip(prompts, image_paths)):
            if not image_path:
                responses[idx] = "Score: 0\nExplanation: Image path is missing."
                continue
            valid_jobs.append((idx, prompt, image_path))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.query_vlm, prompt, image_path): idx
                for idx, prompt, image_path in valid_jobs
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                idx = futures[future]
                try:
                    responses[idx] = future.result().strip()
                except Exception:
                    responses[idx] = "Score: 0\nExplanation: VLM_GENERATION_FAILED"

        for idx, item in enumerate(responses):
            if item is None:
                responses[idx] = "Score: 0\nExplanation: Skipped due to invalid input."

        return responses  # type: ignore[return-value]