from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from configs.pipeline_config import SCORING_CONFIG
from configs.prompts import (
    PROMPT_EXTERNAL_KNOWLEDGE_CORRECTNESS,
    PROMPT_INFERENCE_CORRECTNESS,
    PROMPT_STEP1_MARK_NON_VISUAL,
    PROMPT_STEP2_REMOVE_NON_VISUAL,
    PROMPT_STEP3_REPHRASE_VISUAL,
    PROMPT_VISUAL_CONSISTENCY,
)
from mm_pipeline.clients import ModelClients
from mm_pipeline.utils import (
    batch_iterator,
    clean_text,
    get_image_path,
    load_json,
    parse_score,
    save_json,
)


def _contains_tag(text: str, tag: str) -> bool:
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    return any(match.strip() for match in matches)


class DatasetScorer:
    def __init__(self, clients: Optional[ModelClients] = None) -> None:
        self.clients = clients or ModelClients()

    def _conditional_vlm_scores(
        self,
        prompt_template: str,
        tag: str,
        marked_responses: List[str],
        image_paths: List[Optional[str]],
        batch_size: int,
    ) -> List[str]:
        prompts: List[Optional[str]] = []
        for text in marked_responses:
            if _contains_tag(text, tag):
                prompts.append(prompt_template.format(text_to_evaluate=text))
            else:
                prompts.append(None)

        final_results = [SCORING_CONFIG.neutral_no_content_score_text] * len(prompts)

        subset_indices = [idx for idx, prompt in enumerate(prompts) if prompt is not None]
        if not subset_indices:
            return final_results

        subset_prompts = [prompts[idx] for idx in subset_indices]  # type: ignore[list-item]
        subset_image_paths = [image_paths[idx] for idx in subset_indices]

        subset_results = self.clients.batch_query_vlm(
            subset_prompts,
            subset_image_paths,
            num_workers=batch_size,
            desc=f"Evaluate <{tag}>",
        )

        for idx, result in zip(subset_indices, subset_results):
            final_results[idx] = result

        return final_results

    def score(
        self,
        input_json: str,
        img_dir: str,
        output_json: str,
        batch_size: int = SCORING_CONFIG.default_batch_size,
    ) -> None:
        full_dataset: List[Dict[str, Any]] = load_json(input_json)

        existing_results: List[Dict[str, Any]] = []
        scored_ids = set()

        try:
            existing_results = load_json(output_json)
            scored_ids = {
                item["id"] for item in existing_results
                if "id" in item and "error" not in item
            }
        except Exception:
            existing_results = []
            scored_ids = set()

        dataset_to_score = [item for item in full_dataset if item.get("id") not in scored_ids]
        if not dataset_to_score:
            save_json(existing_results, output_json)
            return

        new_results: List[Dict[str, Any]] = []
        total_batches = (len(dataset_to_score) + batch_size - 1) // batch_size

        for batch in tqdm(batch_iterator(dataset_to_score, batch_size), total=total_batches, desc="Scoring dataset"):
            try:
                instructions = [item["conversations"][0]["value"] for item in batch]
                responses = [item["conversations"][1]["value"] for item in batch]
                image_paths = [get_image_path(img_dir, item) for item in batch]

                step1_prompts = [
                    PROMPT_STEP1_MARK_NON_VISUAL.format(response=response)
                    for response in responses
                ]
                marked_responses = self.clients.batch_query_llm(
                    step1_prompts,
                    num_workers=batch_size,
                    output_prefix_to_remove="Marked Response:",
                    desc="Step1 mark non-visual",
                )

                step2_prompts = [
                    PROMPT_STEP2_REMOVE_NON_VISUAL.format(
                        instruction=instruction,
                        marked_response=marked_response,
                    )
                    for instruction, marked_response in zip(instructions, marked_responses)
                ]
                cleaned_responses = self.clients.batch_query_llm(
                    step2_prompts,
                    num_workers=batch_size,
                    output_prefix_to_remove="Cleaned Response:",
                    desc="Step2 clean response",
                )

                step3_prompts = [
                    PROMPT_STEP3_REPHRASE_VISUAL.format(
                        instruction=instruction,
                        cleaned_response=cleaned_response,
                    )
                    for instruction, cleaned_response in zip(instructions, cleaned_responses)
                ]
                visual_summaries = self.clients.batch_query_llm(
                    step3_prompts,
                    num_workers=batch_size,
                    output_prefix_to_remove="Visual Summary:",
                    desc="Step3 rephrase visual",
                )

                visual_prompts = [
                    PROMPT_VISUAL_CONSISTENCY.format(text_input=text)
                    for text in visual_summaries
                ]
                visual_score_texts = self.clients.batch_query_vlm(
                    visual_prompts,
                    image_paths,
                    num_workers=batch_size,
                    desc="Visual consistency",
                )

                inference_score_texts = self._conditional_vlm_scores(
                    PROMPT_INFERENCE_CORRECTNESS,
                    "INFER",
                    marked_responses,
                    image_paths,
                    batch_size,
                )

                knowledge_score_texts = self._conditional_vlm_scores(
                    PROMPT_EXTERNAL_KNOWLEDGE_CORRECTNESS,
                    "KNOW",
                    marked_responses,
                    image_paths,
                    batch_size,
                )

            except Exception as exc:
                for item in batch:
                    failed_item = item.copy()
                    failed_item["error"] = f"Batch processing failed: {exc}"
                    new_results.append(failed_item)
                continue

            for idx, item in enumerate(batch):
                try:
                    if (
                        marked_responses[idx] == "LLM_GENERATION_FAILED"
                        or cleaned_responses[idx] == "LLM_GENERATION_FAILED"
                        or visual_summaries[idx] == "LLM_GENERATION_FAILED"
                    ):
                        continue

                    vis_score_str = clean_text(visual_score_texts[idx])
                    inf_score_str = clean_text(inference_score_texts[idx])
                    ext_score_str = clean_text(knowledge_score_texts[idx])

                    vis_score = parse_score(vis_score_str)
                    inf_score = parse_score(inf_score_str)
                    ext_score = parse_score(ext_score_str)

                    if vis_score == 0 or inf_score == 0 or ext_score == 0:
                        continue

                    output_item = item.copy()
                    output_item["step1_marked_response"] = marked_responses[idx]
                    output_item["step2_cleaned_response"] = cleaned_responses[idx]
                    output_item["final_visual_summary"] = visual_summaries[idx]
                    output_item["visual_consistency_score_str"] = vis_score_str
                    output_item["inference_correctness_score_str"] = inf_score_str
                    output_item["external_knowledge_correctness_score_str"] = ext_score_str
                    output_item["composite_score"] = (vis_score + inf_score + ext_score) / 3.0

                    new_results.append(output_item)
                except Exception:
                    continue

        final_results_dict = {item["id"]: item for item in existing_results if "id" in item}
        for item in new_results:
            if "id" in item:
                final_results_dict[item["id"]] = item

        final_results = list(final_results_dict.values())

        for item in final_results:
            if "error" in item:
                item["composite_score"] = 0

        save_json(final_results, output_json)