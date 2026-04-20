from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from config import (
    ABLATION_MODES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    IMG_DIR,
    INPUT_JSON,
    OUTPUT_DIR,
)
from model_utils import query_llm, query_vlm


# ============================================================
# Prompt Templates
# ============================================================

PROMPT_STEP1_MARK_NON_VISUAL = """
Response: {response}

Your task is to precisely insert <INFER> for subjective judgments and <KNOW> for external knowledge.

**Critical Guidelines for Annotation:**
1.  **Tag the Complete Thought:** Precisely wrap the shortest, complete phrase that conveys the entire logical idea (like a cause-and-effect statement) or the full piece of external information.
2.  **Tag Interpretations of Effect/Cause:** Always tag phrases that describe the *effect*, *purpose*, or *reason* for a visual element.
3.  **Strictly Visual is NOT Tagged:** DO NOT tag objective, verifiable descriptions of visual facts.
4.  **Do Not Change Words:** Do not add, delete, or rephrase any original words, like Visible Text or Numbers.

Output Format:
    Marked Response: <your_text>
---
Examples:

Input: The lighting in the room is soft, creating a cozy atmosphere. The design suggests it is from the Victorian era.
Output: Marked Response: The lighting in the room is soft, <INFER>creating a cozy atmosphere</INFER>. <INFER>The design suggests it is from the Victorian era</INFER>.

Input: This is a 1976 postage stamp from Hungary, a country in Central Europe.
Output: Marked Response: This is a 1976 postage stamp from Hungary, <KNOW>a country in Central Europe</KNOW>.

Input: The image shows a can of Coca-Cola.
Output: Marked Response: The image shows a can of Coca-Cola.
"""

PROMPT_STEP2_REMOVE_NON_VISUAL = """
Instruction: {instruction}
Annotated Response: {marked_response}

Task: Process the "Annotated Response" by modifying ONLY the segments wrapped in <INFER>…</INFER> or <KNOW>…</KNOW> tags.
- Rewrite or entirely remove tagged segments to leave only what is directly and objectively visible in the image.
- **Crucially, all content NOT wrapped in tags MUST be preserved exactly as is, without any modification.**

Guidelines:
1.  **Rewrite When Possible:** If a tagged idea can be rephrased as a neutral, objective, image-based description, rewrite it and remove the tags. For example, change "<INFER>creating a cozy atmosphere</INFER>" to "which illuminates the scene."
2.  **Delete When Necessary:** For clearly irrelevant or purely speculative content that cannot be visually confirmed, delete the entire tagged segment (including the tags).
3.  **No New Information:** DO NOT introduce any new guesses, opinions, or visual details that were not already present in the untagged parts of the original response.
4.  **Output Format:** Your response must start with the prefix "Cleaned Response:".

Example:
Input Annotated Response:
"A person wearing sunglasses stands under a tree. <INFER>She must be shielding her eyes from harsh sunlight.</INFER> Leaves are scattered on the ground. <KNOW>This park is famous for its autumn foliage tours.</KNOW>"

Output:
Cleaned Response: A person wearing sunglasses stands under a tree. Leaves are scattered on the ground.
"""

PROMPT_STEP3_REPHRASE_VISUAL = """
Instruction: {instruction}
Cleaned Response: {cleaned_response}

Task: Rephrase the "Cleaned Response" into a single, cohesive, and purely visual description.

Guidelines:
1.  **Strictly Adhere to Input:** Your output MUST be a faithful reorganization of ONLY the information present in the "Cleaned Response."
2.  **Preserve All Details:** Do not omit any visual information. Every object, attribute, and spatial relation from the input must be represented in your summary.
3.  **No New Content or Inference:** Crucially, DO NOT add any new visual details, reasoning, assumptions, or subjective/interpretive language (e.g., "beautiful," "seems like," "creates a sense of"). Your job is to describe, not to analyze.
4.  **Improve Flow:** Focus on improving sentence structure and grammatical correctness to create a natural-sounding paragraph.
5.  **Output Format:** Your response must start with the prefix "Visual Summary:".

Example:
Input Cleaned Response: "A white cat is on a windowsill. The background shows buildings. Light is coming through the window."
Output:
Visual Summary: A white cat sits on a windowsill where bright light is streaming in. Buildings are visible in the background.
"""

PROMPT_VISUAL_CONSISTENCY = """
Input Text: {text_input}

Task: You are a visual consistency scoring assistant. Your task is to evaluate whether the extracted text description’s assertions can be verified by the given image. Only assess consistency, not completeness: do NOT penalize the description for omitting image details, but DO penalize any assertions that contradict or cannot be supported by the image.

CORE SCORING GUIDELINE: Be decisive in your scoring. If the description is fully and accurately supported by the image without any errors, the score must be 5. Do not default to 4 if a 5 is warranted.

Output Format:
Score: integer 1–5
Explanation: Brief justification, indicating which assertions are verifiable and which are inconsistent or unclear.

Scoring Rubric:
Score 1: Severely inconsistent or completely unrelated. Most or all assertions contradict the image.
Score 2: Largely inconsistent. Only one or two minor assertions can be matched to the image.
Score 3: Partially consistent. Some key assertions align with the image, but others are vague, potentially incorrect, or unsupported.
Score 4: Mostly consistent. The bulk of assertions are supported by the image, but there is at least one minor imprecision or slight unsupported detail that does not mislead. Use this score for responses that are good but not perfect.
Score 5: Fully consistent and accurate. Every single assertion in the text is clearly and precisely verifiable in the image. There are no unsupported or contradictory claims. If all claims are verified, you MUST assign this score.
"""

PROMPT_INFERENCE_CORRECTNESS = """
Input Text for Evaluation: {text_to_evaluate}

Task: You are an AI assistant designed to evaluate the correctness of logical reasoning. Your primary focus is to rigorously scrutinize the logical soundness and validity of the reasoning contained ONLY within the <INFER>…</INFER> tags, based on the visual evidence in the image.

Evaluation and Scoring Rules:
1.  Isolate and Evaluate: Focus exclusively on the statements inside the <INFER> tags.
2.  Assess Plausibility against Image: Judge if the inference is a logical and plausible conclusion derived from the visual information in the image.
3.  Output Format:
    Score: integer 1–5
    Explanation: A brief evaluation of the logical rigor, noting key flaws or strengths.

Scoring Rubric:
Score 1: Grossly Illogical or Baseless. The inference is pure speculation with no connection to the image (e.g., predicting the future from a photo of a cat), or it's self-contradictory. 
Score 2: Significant Logical Gaps. The inference is a major leap in logic. While loosely related to the image, it is highly unlikely or requires many unsupported assumptions. (e.g., "A person is running, <INFER>so this must be a professional athlete training for the Olympics</INFER>.")
Score 3: Plausible but Unprovable. The inference is reasonable and could be true, but it is not strongly supported by visual evidence and remains a subjective interpretation. (e.g., "The room is dim, <INFER>creating a sad atmosphere</INFER>.")
Score 4: Logically Sound. The inference is very likely correct and follows directly from strong visual evidence, with only very minor room for doubt. (e.g., "The man holds an umbrella, <INFER>suggesting it is raining or about to rain</INFER>.")
Score 5: Logically Airtight. The inference is an undeniable conclusion based on the visual facts and common-sense logic; it is virtually irrefutable. (e.g., "The wreck shows a crushed car, <INFER>indicating a high-impact collision occurred</INFER>.")
"""

PROMPT_EXTERNAL_KNOWLEDGE_CORRECTNESS = """
Input Text for Evaluation: {text_to_evaluate}

Task: You are an expert fact-checking assistant. Your task is to evaluate the factual correctness of the information contained ONLY within the <KNOW>…</KNOW> tags. Base your assessment on your internal, general knowledge.

Output Format:
Score: integer 1–5
Explanation: A brief justification for your score, specifying which facts are correct or incorrect.

Scoring Rubric:
Score 1: **Entirely Incorrect or Fabricated.** The information is factually wrong, nonsensical, or a complete fabrication (e.g., contains imaginary objects like the 'Luminara Scepter'). 
Score 2: **Largely Incorrect.** Contains a core factual error, even if minor details are correct. (e.g., "<KNOW>Paris, the capital of England...</KNOW>"). The presence of a single major error means the score cannot be higher than 2.
Score 3: **Partially Correct but Misleading.** Contains a mix of correct and incorrect information, or the information is technically correct but presented in a highly misleading context.
Score 4: **Mostly Correct.** The core assertion is factually sound but contains a minor, non-critical inaccuracy (e.g., a slightly wrong year, a minor detail about a standard feature).
Score 5: **Fully Correct and Accurate.** Every single claim within the tags is factually sound, precise, and widely accepted.
"""

PROMPT_INFERENCE_CORRECTNESS_NO_DECOMP = """
Input Text for Evaluation: {text_to_evaluate}

Task: You are an AI assistant designed to evaluate the correctness of logical reasoning. Your primary focus is to rigorously scrutinize the logical soundness and validity of any reasoning or inference contained within the entire input text, based on the visual evidence in the image.

Evaluation and Scoring Rules:
1.  Evaluate the entire text for logical inferences.
2.  Assess Plausibility against Image: Judge if the inferences are logical and plausible conclusions derived from the visual information in the image.
3.  Output Format:
    Score: integer 1–5
    Explanation: A brief evaluation of the logical rigor, noting key flaws or strengths.

Scoring Rubric:
Score 1: Grossly Illogical or Baseless. Any inference is pure speculation with no connection to the image.
Score 2: Significant Logical Gaps. Any inference is a major leap in logic and highly unlikely.
Score 3: Plausible but Unprovable. Inferences are reasonable but not strongly supported by visual evidence.
Score 4: Logically Sound. Inferences are very likely correct and follow directly from strong visual evidence.
Score 5: Logically Airtight. Inferences are undeniable conclusions based on the visual facts.
"""

PROMPT_EXTERNAL_KNOWLEDGE_CORRECTNESS_NO_DECOMP = """
Input Text for Evaluation: {text_to_evaluate}

Task: You are an expert fact-checking assistant. Your task is to evaluate the factual correctness of any external, real-world knowledge contained within the entire input text. Base your assessment on your internal, general knowledge.

Output Format:
    Score: integer 1–5
    Explanation: A brief justification for your score, specifying which facts are correct or incorrect.

Scoring Rubric:
Score 1: **Entirely Incorrect or Fabricated.** The information is factually wrong.
Score 2: **Largely Incorrect.** Contains a core factual error.
Score 3: **Partially Correct but Misleading.** Contains a mix of correct and incorrect information.
Score 4: **Mostly Correct.** The core assertion is factually sound but contains a minor inaccuracy.
Score 5: **Fully Correct and Accurate.** Every single claim is factually sound and precise.
"""


# ============================================================
# Helpers
# ============================================================

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return (
        text.replace("**", "")
        .replace("*", "")
        .replace("_", "")
        .replace("`", "")
        .replace("#", "")
        .strip()
    )


def parse_score(text_with_score: str) -> int:
    text = str(text_with_score)
    match = re.search(r"(?:Score|Final Score)\s*[:：]?\s*([1-5])\b", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    matches = re.findall(r"\b([1-5])\b", text)
    if matches:
        return int(matches[-1])

    return 0


def batch_iterator(iterable: Iterable, n: int) -> Iterable[List]:
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk


def generate_batched_llm_output(
    prompts: List[str],
    output_prefix_to_remove: Optional[str] = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> List[str]:
    responses = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_index = {executor.submit(query_llm, p): i for i, p in enumerate(prompts)}

        for future in tqdm(as_completed(future_to_index), total=len(prompts), desc="Querying LLM"):
            idx = future_to_index[future]
            try:
                res = future.result()
                if output_prefix_to_remove and isinstance(res, str) and res.strip().startswith(output_prefix_to_remove):
                    responses[idx] = res.strip()[len(output_prefix_to_remove):].strip()
                else:
                    responses[idx] = res
            except Exception as e:
                print(f"[LLM] prompt idx={idx} failed: {e}")
                responses[idx] = "LLM_GENERATION_FAILED"

    return responses


def generate_batched_vlm_output(
    image_paths: List[str],
    prompts: List[str],
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> List[str]:
    responses = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_index = {
            executor.submit(query_vlm, prompt, img_path): i
            for i, (prompt, img_path) in enumerate(zip(prompts, image_paths))
        }

        for future in tqdm(as_completed(future_to_index), total=len(prompts), desc="Querying VLM"):
            idx = future_to_index[future]
            try:
                responses[idx] = future.result()
            except Exception as e:
                print(f"[VLM] prompt idx={idx} failed: {e}")
                responses[idx] = "Score: 0\nExplanation: VLM request generated an exception."

    return responses


def get_conditional_scores(
    prompt_template: str,
    tag: str,
    marked_responses: List[str],
    image_paths: List[str],
    num_workers: int,
) -> List[str]:
    prompts: List[Optional[str]] = []

    for res in marked_responses:
        matches = re.findall(rf"<{tag}>(.*?)</{tag}>", res or "", re.DOTALL)
        if any(m.strip() for m in matches):
            prompts.append(prompt_template.format(text_to_evaluate=res))
        else:
            prompts.append(None)

    indexed_prompts = [(i, p) for i, p in enumerate(prompts) if p is not None]
    if not indexed_prompts:
        return ["Score: 2\nExplanation: No content detected."] * len(prompts)

    indices, actual_prompts = zip(*indexed_prompts)
    actual_paths = [image_paths[i] for i in indices]
    subset_results = generate_batched_vlm_output(actual_paths, list(actual_prompts), num_workers=num_workers)

    final_results = ["Score: 2\nExplanation: No content detected."] * len(prompts)
    for i, res in zip(indices, subset_results):
        final_results[i] = res
    return final_results


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_existing_results(output_json: str) -> Tuple[List[Dict], set]:
    if not os.path.exists(output_json):
        return [], set()

    try:
        data = load_json(output_json)
        scored_ids = {x["id"] for x in data if "id" in x and "error" not in x}
        return data, scored_ids
    except Exception as e:
        print(f"[WARN] failed to load existing output: {e}")
        return [], set()


# ============================================================
# Core
# ============================================================

def build_output_paths(mode: str, output_json: Optional[str]) -> str:
    ensure_dir(OUTPUT_DIR)
    if output_json:
        return output_json
    return os.path.join(OUTPUT_DIR, ABLATION_MODES[mode]["output_name"])


def process_batch(
    batch: List[Dict],
    img_dir: str,
    use_decomposition: bool,
    use_inference_score: bool,
    use_knowledge_score: bool,
    num_workers: int,
) -> List[Dict]:
    batch_instructions = [sample["conversations"][0]["value"] for sample in batch]
    batch_responses = [sample["conversations"][1]["value"] for sample in batch]
    batch_image_paths = []

    for sample in batch:
        image_filename = sample.get("image")
        if image_filename and isinstance(image_filename, str):
            batch_image_paths.append(os.path.join(img_dir, image_filename))
        else:
            batch_image_paths.append("")

    if use_decomposition:
        step1_prompts = [
            PROMPT_STEP1_MARK_NON_VISUAL.format(response=res)
            for res in batch_responses
        ]
        batch_marked_responses = generate_batched_llm_output(
            step1_prompts,
            output_prefix_to_remove="Marked Response:",
            num_workers=num_workers,
        )

        step2_prompts = [
            PROMPT_STEP2_REMOVE_NON_VISUAL.format(instruction=instr, marked_response=marked)
            for instr, marked in zip(batch_instructions, batch_marked_responses)
        ]
        batch_cleaned_responses = generate_batched_llm_output(
            step2_prompts,
            output_prefix_to_remove="Cleaned Response:",
            num_workers=num_workers,
        )

        step3_prompts = [
            PROMPT_STEP3_REPHRASE_VISUAL.format(instruction=instr, cleaned_response=cleaned)
            for instr, cleaned in zip(batch_instructions, batch_cleaned_responses)
        ]
        batch_final_visual_summaries = generate_batched_llm_output(
            step3_prompts,
            output_prefix_to_remove="Visual Summary:",
            num_workers=num_workers,
        )
    else:
        batch_marked_responses = ["ABLATED"] * len(batch)
        batch_cleaned_responses = ["ABLATED"] * len(batch)
        batch_final_visual_summaries = batch_responses

    vis_prompts = [
        PROMPT_VISUAL_CONSISTENCY.format(text_input=summary)
        for summary in batch_final_visual_summaries
    ]
    batch_vis_scores_str = generate_batched_vlm_output(
        batch_image_paths,
        vis_prompts,
        num_workers=num_workers,
    )

    if use_inference_score:
        if use_decomposition:
            batch_inf_scores_str = get_conditional_scores(
                PROMPT_INFERENCE_CORRECTNESS,
                "INFER",
                batch_marked_responses,
                batch_image_paths,
                num_workers=num_workers,
            )
        else:
            inf_prompts = [
                PROMPT_INFERENCE_CORRECTNESS_NO_DECOMP.format(text_to_evaluate=res)
                for res in batch_responses
            ]
            batch_inf_scores_str = generate_batched_vlm_output(
                batch_image_paths,
                inf_prompts,
                num_workers=num_workers,
            )
    else:
        batch_inf_scores_str = [""] * len(batch)

    if use_knowledge_score:
        if use_decomposition:
            batch_know_scores_str = get_conditional_scores(
                PROMPT_EXTERNAL_KNOWLEDGE_CORRECTNESS,
                "KNOW",
                batch_marked_responses,
                batch_image_paths,
                num_workers=num_workers,
            )
        else:
            know_prompts = [
                PROMPT_EXTERNAL_KNOWLEDGE_CORRECTNESS_NO_DECOMP.format(text_to_evaluate=res)
                for res in batch_responses
            ]
            batch_know_scores_str = generate_batched_vlm_output(
                batch_image_paths,
                know_prompts,
                num_workers=num_workers,
            )
    else:
        batch_know_scores_str = [""] * len(batch)

    results = []
    for i, sample in enumerate(batch):
        item = sample.copy()

        if (batch_marked_responses[i] == "LLM_GENERATION_FAILED" or
            batch_cleaned_responses[i] == "LLM_GENERATION_FAILED" or
            batch_final_visual_summaries[i] == "LLM_GENERATION_FAILED"):
            item["error"] = "Item processing failed: LLM_GENERATION_FAILED"
            item["composite_score"] = 0
            results.append(item)
            continue

        item["step1_marked_response"] = batch_marked_responses[i]
        item["step2_cleaned_response"] = batch_cleaned_responses[i]
        item["final_visual_summary"] = batch_final_visual_summaries[i]
        item["visual_consistency_score_str"] = clean_text(batch_vis_scores_str[i])
        item["inference_correctness_score_str"] = clean_text(batch_inf_scores_str[i])
        item["external_knowledge_correctness_score_str"] = clean_text(batch_know_scores_str[i])

        vis_score = parse_score(item["visual_consistency_score_str"])
        enabled_scores = [vis_score]

        if use_inference_score:
            enabled_scores.append(parse_score(item["inference_correctness_score_str"]))
        if use_knowledge_score:
            enabled_scores.append(parse_score(item["external_knowledge_correctness_score_str"]))

        item["composite_score"] = sum(enabled_scores) / max(len(enabled_scores), 1)
        results.append(item)

    return results


def run(
    input_json: str,
    img_dir: str,
    output_json: str,
    batch_size: int,
    mode: str,
    num_workers: int,
) -> None:
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unsupported mode: {mode}")

    mode_cfg = ABLATION_MODES[mode]

    full_dataset = load_json(input_json)
    print(f"Loaded {len(full_dataset)} total records.")

    existing_results, scored_ids = load_existing_results(output_json)
    dataset_to_score = [x for x in full_dataset if x.get("id") not in scored_ids]

    if not dataset_to_score:
        print("Nothing to do. All records are already scored.")
        return

    print(f"Mode = {mode}")
    print(f"Need to process {len(dataset_to_score)} records.")
    print(f"Output = {output_json}")

    new_results = []
    total_batches = (len(dataset_to_score) + batch_size - 1) // batch_size

    for batch in tqdm(batch_iterator(dataset_to_score, batch_size), total=total_batches, desc=f"Scoring[{mode}]"):
        try:
            batch_results = process_batch(
                batch=batch,
                img_dir=img_dir,
                use_decomposition=mode_cfg["use_decomposition"],
                use_inference_score=mode_cfg["use_inference_score"],
                use_knowledge_score=mode_cfg["use_knowledge_score"],
                num_workers=num_workers,
            )
            new_results.extend(batch_results)
        except Exception as e:
            print(f"[BATCH ERROR] {e}")
            traceback.print_exc()
            for sample in batch:
                failed = sample.copy()
                failed["error"] = f"Batch processing failed: {e}"
                failed["composite_score"] = 0
                new_results.append(failed)

    final_dict = {x["id"]: x for x in existing_results if "id" in x}
    for x in new_results:
        final_dict[x["id"]] = x

    final_results = list(final_dict.values())
    save_json(final_results, output_json)
    print(f"Done. Saved to {output_json}")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified ablation runner.")
    parser.add_argument("--mode", type=str, required=True, choices=list(ABLATION_MODES.keys()))
    parser.add_argument("--input_json", type=str, default=INPUT_JSON)
    parser.add_argument("--img_dir", type=str, default=IMG_DIR)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_json = build_output_paths(args.mode, args.output_json)
    run(
        input_json=args.input_json,
        img_dir=args.img_dir,
        output_json=output_json,
        batch_size=args.batch_size,
        mode=args.mode,
        num_workers=args.num_workers,
    )