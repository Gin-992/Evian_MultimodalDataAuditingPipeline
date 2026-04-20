from __future__ import annotations

from typing import Any, Dict, List

from mm_pipeline.utils import load_json, save_json


def finalize_dataset(
    input_file_path: str,
    output_file_path: str,
    top_n: int,
) -> None:
    all_data: List[Dict[str, Any]] = load_json(input_file_path)
    if not all_data:
        save_json([], output_file_path)
        return

    all_data.sort(key=lambda item: item.get("composite_score", 0), reverse=True)
    top_data = all_data[:top_n]

    cleaned_data: List[Dict[str, Any]] = []
    for item in top_data:
        cleaned_item: Dict[str, Any] = {}

        for field in ["id", "image", "conversations"]:
            if field in item:
                cleaned_item[field] = item[field]

        cleaned_item["composite_score"] = item.get("composite_score")

        if "error_category" in item:
            cleaned_item["error_category"] = item["error_category"]
        if "error_subtype" in item:
            cleaned_item["error_subtype"] = item["error_subtype"]

        cleaned_data.append(cleaned_item)

    save_json(cleaned_data, output_file_path)