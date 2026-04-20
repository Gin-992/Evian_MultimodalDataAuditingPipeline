import os
import re
import json
import random
import shutil
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def atomic_save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_image_path(image_dir: str, rel_path: Optional[str]) -> Optional[str]:
    if not rel_path or not isinstance(rel_path, str):
        return None
    rel_path = rel_path.strip()
    if not rel_path:
        return None
    full_path = os.path.join(image_dir, rel_path)
    if not os.path.isfile(full_path):
        return None
    return full_path


def get_last_conversation_text(entry: Dict[str, Any]) -> str:
    conversations = entry.get("conversations", [])
    if not conversations:
        return ""
    last_item = conversations[-1]
    if isinstance(last_item, dict):
        return last_item.get("value", "")
    return ""


def parse_first_int(text: Any, default: int = 0) -> int:
    if not isinstance(text, str):
        return default
    numbers = re.findall(r"\d+", text)
    if not numbers:
        return default
    try:
        return int(numbers[0])
    except Exception:
        return default


def select_top_n_by_score(data: List[Dict[str, Any]], score_key: str, top_n: int) -> List[Dict[str, Any]]:
    valid = [x for x in data if isinstance(x.get(score_key), (int, float))]
    valid.sort(key=lambda x: x[score_key], reverse=True)
    return valid[: min(top_n, len(valid))]


def sample_without_replacement(data: List[Any], sample_size: int, seed: Optional[int] = None) -> List[Any]:
    if seed is not None:
        random.seed(seed)
    if sample_size > len(data):
        raise ValueError(f"sample_size={sample_size} exceeds dataset size={len(data)}")
    return random.sample(data, sample_size)


def copy_images(sampled: List[Dict[str, Any]], image_dir: str, output_image_dir: str) -> None:
    ensure_dir(output_image_dir)
    for entry in sampled:
        rel_path = entry.get("image")
        src = resolve_image_path(image_dir, rel_path)
        if src is None:
            continue
        dst = os.path.join(output_image_dir, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"[WARN] Failed to copy {src} -> {dst}: {e}")


def load_processed_ids(existing_scored_path: str) -> Tuple[set, List[Dict[str, Any]]]:
    if not os.path.exists(existing_scored_path):
        return set(), []

    try:
        data = load_json(existing_scored_path)
        processed_ids = set()
        for item in data:
            if isinstance(item, dict) and item.get("id") is not None:
                processed_ids.add(item["id"])
        return processed_ids, data
    except Exception as e:
        print(f"[WARN] Failed to read existing scored file: {e}")
        return set(), []