import os
import argparse
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from utils import load_json, save_json, resolve_image_path, get_last_conversation_text, select_top_n_by_score


def compute_clip_similarity(image: Image.Image, text: str, processor, model, device: str) -> float:
    inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=processor.tokenizer.model_max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        image_features = F.normalize(outputs.image_embeds, p=2, dim=-1)
        text_features = F.normalize(outputs.text_embeds, p=2, dim=-1)
        similarity = image_features @ text_features.T

    return similarity.item()


def process_dataset(
    json_path: str,
    image_dir: str,
    output_json: str,
    model_name: str,
    device: str,
    top_output_json: str = None,
    top_n: int = None,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, fallback to CPU.")
        device = "cpu"

    print(f"Loading CLIP model: {model_name}")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)

    data: List[Dict[str, Any]] = load_json(json_path)

    for entry in tqdm(data, desc="Computing CLIP similarities"):
        img_path = resolve_image_path(image_dir, entry.get("image"))
        if img_path is None:
            entry["similarity_score"] = None
            continue

        text = get_last_conversation_text(entry)

        try:
            with Image.open(img_path) as im:
                image = im.convert("RGB")
            score = compute_clip_similarity(image, text, processor, model, device)
            entry["similarity_score"] = score
        except Exception as e:
            print(f"[WARN] Failed to process {img_path}: {e}")
            entry["similarity_score"] = None

    save_json(data, output_json)
    print(f"Saved full results to: {output_json}")

    if top_output_json and top_n is not None:
        top_entries = select_top_n_by_score(data, "similarity_score", top_n)
        save_json(top_entries, top_output_json)
        print(f"Saved top-{len(top_entries)} results to: {top_output_json}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute CLIP image-text similarity ranking")
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--model_name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_output_json", default=None)
    parser.add_argument("--top_n", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_dataset(
        json_path=args.json_path,
        image_dir=args.image_dir,
        output_json=args.output_json,
        model_name=args.model_name,
        device=args.device,
        top_output_json=args.top_output_json,
        top_n=args.top_n,
    )