from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from configs.pipeline_config import LOW_QUALITY_CONFIG
from configs.prompts import (
    ERROR_CATALOG,
    PROMPT_ANALYZE_TEXT,
    PROMPT_CHOOSE_ERROR_SUBTYPE,
    PROMPT_GENERATE_LOW_QUALITY,
)
from mm_pipeline.clients import ModelClients
from mm_pipeline.utils import batch_iterator, load_json, save_json


def _catalog_by_type(error_type: str) -> List[Dict[str, str]]:
    return [item for item in ERROR_CATALOG if item["type"] == error_type]


class LowQualityGenerator:
    def __init__(self, clients: Optional[ModelClients] = None) -> None:
        self.clients = clients or ModelClients()

    def _choose_category(self, analysis: Dict[str, Any]) -> str:
        if analysis.get("contains_knowledge") and random.random() < LOW_QUALITY_CONFIG.analysis_probability_knowledge:
            return "knowledge"
        if analysis.get("contains_reasoning") and random.random() < LOW_QUALITY_CONFIG.analysis_probability_reasoning:
            return "reasoning"
        return "consistency"

    def _build_error_options_text(self, category_options: List[Dict[str, str]]) -> str:
        return "\n".join(
            f"- {item['error_code']}: {item['prompt_instruction']}"
            for item in category_options
        )

    def generate(
        self,
        input_file: str,
        output_file: str,
        num_samples: Optional[int] = None,
        batch_size: int = LOW_QUALITY_CONFIG.default_batch_size,
    ) -> None:
        existing_results: List[Dict[str, Any]] = []
        processed_ids = set()

        try:
            existing_results = load_json(output_file)
            processed_ids = {item["id"] for item in existing_results if "id" in item}
        except Exception:
            existing_results = []
            processed_ids = set()

        original_data: List[Dict[str, Any]] = load_json(input_file)
        candidates = [item for item in original_data if item.get("id") not in processed_ids]

        if num_samples is not None:
            remaining = max(num_samples - len(existing_results), 0)
            candidates = candidates[:remaining]

        if not candidates:
            save_json(existing_results, output_file)
            return

        new_items: List[Dict[str, Any]] = []
        total_batches = (len(candidates) + batch_size - 1) // batch_size

        for batch in tqdm(batch_iterator(candidates, batch_size), total=total_batches, desc="Generating low-quality data"):
            batch_responses = [item["conversations"][1]["value"] for item in batch]

            analysis_prompts = [
                PROMPT_ANALYZE_TEXT.format(text=text) for text in batch_responses
            ]
            analysis_results = self.clients.batch_query_llm(
                analysis_prompts,
                num_workers=batch_size,
                expect_json=True,
                desc="Analyze text",
            )

            chosen_categories: List[str] = []
            chosen_codes: List[Optional[str]] = [None] * len(batch)

            subtype_query_indices = []
            subtype_prompts = []

            for idx, analysis in enumerate(analysis_results):
                if not isinstance(analysis, dict) or analysis.get("error"):
                    category = "consistency"
                else:
                    category = self._choose_category(analysis)

                chosen_categories.append(category)
                options = _catalog_by_type(category)

                if category in {"knowledge", "reasoning"}:
                    subtype_query_indices.append(idx)
                    subtype_prompts.append(
                        PROMPT_CHOOSE_ERROR_SUBTYPE.format(
                            options=self._build_error_options_text(options),
                            text=batch_responses[idx],
                        )
                    )
                else:
                    chosen_codes[idx] = random.choice(options)["error_code"]

            if subtype_prompts:
                subtype_results = self.clients.batch_query_llm(
                    subtype_prompts,
                    num_workers=batch_size,
                    expect_json=True,
                    desc="Choose error subtype",
                )
                for batch_idx, result in zip(subtype_query_indices, subtype_results):
                    category = chosen_categories[batch_idx]
                    options = _catalog_by_type(category)
                    valid_codes = {item["error_code"] for item in options}
                    if isinstance(result, dict) and result.get("best_choice") in valid_codes:
                        chosen_codes[batch_idx] = result["best_choice"]
                    else:
                        chosen_codes[batch_idx] = random.choice(options)["error_code"]

            final_prompts = []
            generation_indices = []

            for idx, code in enumerate(chosen_codes):
                if not code:
                    continue
                error_def = next(item for item in ERROR_CATALOG if item["error_code"] == code)
                final_prompts.append(
                    PROMPT_GENERATE_LOW_QUALITY.format(
                        instruction=error_def["prompt_instruction"],
                        text=batch_responses[idx],
                    )
                )
                generation_indices.append(idx)

            generated_subset = []
            if final_prompts:
                generated_subset = self.clients.batch_query_llm(
                    final_prompts,
                    num_workers=batch_size,
                    desc="Generate corrupted text",
                )

            generated_responses: List[Optional[str]] = [None] * len(batch)
            for idx, text in zip(generation_indices, generated_subset):
                generated_responses[idx] = text

            for idx, item in enumerate(batch):
                low_quality_text = generated_responses[idx]
                if not low_quality_text or low_quality_text == "LLM_GENERATION_FAILED" or len(low_quality_text) < 5:
                    continue

                error_def = next(item_ for item_ in ERROR_CATALOG if item_["error_code"] == chosen_codes[idx])

                new_item = item.copy()
                new_item["conversations"] = [
                    item["conversations"][0],
                    {"from": "gpt", "value": low_quality_text},
                ]
                new_item["error_category"] = error_def["type"]
                new_item["error_subtype"] = error_def["error_code"]
                new_items.append(new_item)

        save_json(existing_results + new_items, output_file)