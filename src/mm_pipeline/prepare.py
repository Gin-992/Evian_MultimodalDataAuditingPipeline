from __future__ import annotations

import random
from typing import Any, Dict, List

from mm_pipeline.utils import load_json, save_json


def combine_datasets(
    original_data_path: str,
    low_quality_data_path: str,
    output_file_path: str,
    num_original_samples: int = 100000,
    seed: int = 42,
) -> None:
    original_data: List[Dict[str, Any]] = load_json(original_data_path)
    low_quality_data: List[Dict[str, Any]] = load_json(low_quality_data_path)

    if len(original_data) <= num_original_samples:
        sampled_original_data = original_data
    else:
        rng = random.Random(seed)
        sampled_original_data = rng.sample(original_data, num_original_samples)

    combined_data = list(low_quality_data) + list(sampled_original_data)
    random.Random(seed).shuffle(combined_data)

    save_json(combined_data, output_file_path)