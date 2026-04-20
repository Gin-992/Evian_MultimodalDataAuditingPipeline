import os
import gc
import time
import json
import asyncio
import argparse
from typing import Dict, Any, List, Tuple

from tqdm import tqdm

from prompts import build_eval_prompt_from_conversations
from utils import (
    load_json,
    save_json,
    atomic_save_json,
    resolve_image_path,
    parse_first_int,
    load_processed_ids,
    ensure_dir,
)
from vlm_client import query_vlm_batch


def build_output_paths(output_dir: str, top_k: int) -> Dict[str, str]:
    ensure_dir(output_dir)
    return {
        "full_scored": os.path.join(output_dir, "all_scored_samples.json"),
        "top_clean": os.path.join(output_dir, f"top_{top_k}_clean.json"),
        "top_eval": os.path.join(output_dir, f"top_{top_k}_eval.json"),
    }


def prepare_candidates(
    all_samples: List[Dict[str, Any]],
    image_dir: str,
    processed_ids: set,
    verbose: bool = True,
) -> List[Tuple[Dict[str, Any], str, str]]:
    candidates = []

    for sample in all_samples:
        sid = sample.get("id")
        if sid in processed_ids:
            continue

        img_path = resolve_image_path(image_dir, sample.get("image"))
        if img_path is None:
            if verbose:
                print(f"[WARN] Missing image for sample id={sid}")
            continue

        conversations = sample.get("conversations", [])
        eval_prompt = build_eval_prompt_from_conversations(conversations)
        if eval_prompt is None:
            if verbose:
                print(f"[WARN] Failed to build evaluation prompt for sample id={sid}")
            continue

        candidates.append((sample, img_path, eval_prompt))

    return candidates


def save_progress(processed_data: List[Dict[str, Any]], output_path: str):
    atomic_save_json(processed_data, output_path)


def process_batches(
    candidates: List[Tuple[Dict[str, Any], str, str]],
    processed_data: List[Dict[str, Any]],
    processed_ids: set,
    output_full_scored_path: str,
    batch_size: int,
    save_every_n_batches: int,
    verbose: bool = True,
):
    start_time = time.time()
    processed_batches_since_last_save = 0

    for i in tqdm(range(0, len(candidates), batch_size), desc="Processing batches"):
        batch = candidates[i: i + batch_size]
        if not batch:
            continue

        batch_samples = [x[0] for x in batch]
        batch_image_paths = [x[1] for x in batch]
        batch_prompts = [x[2] for x in batch]

        try:
            batch_responses = asyncio.run(
                query_vlm_batch(
                    prompts=batch_prompts,
                    image_paths=batch_image_paths,
                )
            )

            for sample, response_text in zip(batch_samples, batch_responses):
                score = parse_first_int(response_text, default=0)
                final_sample = sample.copy()
                final_sample["score"] = score

                processed_data.append(final_sample)
                if final_sample.get("id") is not None:
                    processed_ids.add(final_sample["id"])

            processed_batches_since_last_save += 1

            if processed_batches_since_last_save >= save_every_n_batches:
                save_progress(processed_data, output_full_scored_path)
                processed_batches_since_last_save = 0
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"[INFO] Saved progress. processed={len(processed_data)}, elapsed={elapsed:.1f}s")

        except Exception as e:
            print(f"[ERROR] Failed on batch {i // batch_size + 1}: {e}")
        finally:
            gc.collect()

    if processed_batches_since_last_save > 0:
        save_progress(processed_data, output_full_scored_path)


def export_top_k(
    full_scored_path: str,
    output_top_clean_path: str,
    output_top_eval_path: str,
    top_k: int,
):
    all_scored_data = load_json(full_scored_path)
    sorted_data = sorted(all_scored_data, key=lambda x: x.get("score", 0), reverse=True)
    top_k_samples = sorted_data[: min(top_k, len(sorted_data))]

    top_clean_dataset = []
    for sample in top_k_samples:
        top_clean_dataset.append(
            {
                "id": sample.get("id"),
                "image": sample.get("image"),
                "conversations": sample.get("conversations"),
            }
        )

    save_json(top_clean_dataset, output_top_clean_path)
    save_json(top_k_samples, output_top_eval_path)

    print(f"Saved clean top-k to: {output_top_clean_path}")
    print(f"Saved eval  top-k to: {output_top_eval_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Score dataset with a VLM judge and export top-k samples")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_every_n_batches", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = build_output_paths(args.output_dir, args.top_k)

    processed_ids, processed_data = load_processed_ids(paths["full_scored"])
    print(f"Loaded processed samples: {len(processed_ids)}")

    all_samples = load_json(args.input_json)
    print(f"Loaded total samples: {len(all_samples)}")

    candidates = prepare_candidates(
        all_samples=all_samples,
        image_dir=args.image_dir,
        processed_ids=processed_ids,
        verbose=args.verbose,
    )
    print(f"Prepared candidates: {len(candidates)}")

    process_batches(
        candidates=candidates,
        processed_data=processed_data,
        processed_ids=processed_ids,
        output_full_scored_path=paths["full_scored"],
        batch_size=args.batch_size,
        save_every_n_batches=args.save_every_n_batches,
        verbose=args.verbose,
    )

    export_top_k(
        full_scored_path=paths["full_scored"],
        output_top_clean_path=paths["top_clean"],
        output_top_eval_path=paths["top_eval"],
        top_k=args.top_k,
    )

    print("All done.")


if __name__ == "__main__":
    main()