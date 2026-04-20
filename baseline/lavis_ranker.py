import argparse
from typing import Dict, Any, List

import torch
from PIL import Image
from tqdm import tqdm
from lavis.models import load_model_and_preprocess

from utils import load_json, save_json, resolve_image_path, get_last_conversation_text, select_top_n_by_score


def compute_lavis_similarity(model, vis_processors, txt_processors, image, caption: str, device: str) -> float:
    img_tensor = vis_processors["eval"](image).unsqueeze(0).to(device)
    text_input = txt_processors["eval"](caption)

    feats_img = model.extract_features(
        {"image": img_tensor, "text_input": [text_input]},
        mode="image",
    ).image_embeds_proj

    feats_txt = model.extract_features(
        {"image": img_tensor, "text_input": [text_input]},
        mode="text",
    ).text_embeds_proj

    img_feat = feats_img[:, 0, :]
    txt_feat = feats_txt[:, 0, :]

    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    return (img_feat @ txt_feat.t()).item()


def infer_model_type(model_name: str) -> str:
    return "pretrain" if "blip2" in model_name else "base"


def process_dataset(
    model_name: str,
    json_path: str,
    image_dir: str,
    output_json: str,
    device: str,
    top_output_json: str = None,
    top_n: int = None,
):
    device = str(device)
    model_type = infer_model_type(model_name)

    print(f"Loading LAVIS model: name={model_name}, type={model_type}")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=model_name,
        model_type=model_type,
        is_eval=True,
        device=torch.device(device),
    )

    data: List[Dict[str, Any]] = load_json(json_path)

    for entry in tqdm(data, desc=f"Computing {model_name} similarities"):
        img_path = resolve_image_path(image_dir, entry.get("image"))
        if img_path is None:
            entry["similarity_score"] = None
            continue

        text = get_last_conversation_text(entry)

        try:
            with Image.open(img_path) as im:
                image = im.convert("RGB")
            entry["similarity_score"] = compute_lavis_similarity(
                model, vis_processors, txt_processors, image, text, device
            )
        except Exception as e:
            print(f"[WARN] Error processing {img_path}: {e}")
            entry["similarity_score"] = None

    save_json(data, output_json)
    print(f"Saved full results to: {output_json}")

    if top_output_json and top_n is not None:
        top_entries = select_top_n_by_score(data, "similarity_score", top_n)
        save_json(top_entries, top_output_json)
        print(f"Saved top-{len(top_entries)} results to: {top_output_json}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute image-text similarity using LAVIS")
    parser.add_argument("--model", required=True, help="e.g. albef_feature_extractor / blip_feature_extractor / blip2_feature_extractor")
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_output_json", default=None)
    parser.add_argument("--top_n", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_dataset(
        model_name=args.model,
        json_path=args.json_path,
        image_dir=args.image_dir,
        output_json=args.output_json,
        device=args.device,
        top_output_json=args.top_output_json,
        top_n=args.top_n,
    )