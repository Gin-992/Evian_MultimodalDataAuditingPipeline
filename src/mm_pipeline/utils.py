from __future__ import annotations

import base64
import json
import mimetypes
import os
import random
import re
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def batch_iterator(iterable: Iterable[Any], batch_size: int) -> Iterator[List[Any]]:
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, batch_size))
        if not chunk:
            break
        yield chunk


def clean_text(text: Any) -> str:
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


def parse_score(text_with_score: Any) -> int:
    if not isinstance(text_with_score, str):
        return 0

    match = re.search(r"Score\s*[:：]\s*(\d+)", text_with_score, re.IGNORECASE)
    if not match:
        return 0

    score = int(match.group(1))
    return score if 1 <= score <= 5 else 0


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def remove_optional_prefix(text: str, prefix: Optional[str]) -> str:
    if not prefix:
        return text.strip()
    stripped = text.strip()
    if stripped.startswith(prefix):
        return stripped[len(prefix):].strip()
    return stripped


def sample_or_all(items: List[Any], n: int, seed: int = 42) -> List[Any]:
    if len(items) <= n:
        return list(items)
    rng = random.Random(seed)
    return rng.sample(items, n)


def encode_image_to_data_uri(image_path: str | Path) -> str:
    image_path = str(image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def get_image_path(img_dir: str | Path, sample: Dict[str, Any]) -> Optional[str]:
    image_name = sample.get("image")
    if not image_name or not isinstance(image_name, str):
        return None
    return str(Path(img_dir) / image_name)