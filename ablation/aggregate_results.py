from __future__ import annotations

import argparse
import json
from typing import List, Dict


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def aggregate(input_file: str, output_file: str, top_n: int) -> None:
    data: List[Dict] = load_json(input_file)

    valid = [x for x in data if "error" not in x]
    valid.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

    top_data = valid[:top_n]
    save_json(top_data, output_file)

    print(f"Input records : {len(data)}")
    print(f"Valid records : {len(valid)}")
    print(f"Top N         : {len(top_data)}")
    print(f"Saved to      : {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate ablation results by composite_score.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    aggregate(args.input_file, args.output_file, args.top_n)