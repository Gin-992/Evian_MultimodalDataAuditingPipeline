import argparse

from mm_pipeline.scoring import DatasetScorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Score dataset with LLM+VLM pipeline.")
    parser.add_argument("--input_json", required=True, help="Path to the input JSON file.")
    parser.add_argument("--img_dir", required=True, help="Directory containing images.")
    parser.add_argument("--output_json", required=True, help="Path to the output scored JSON file.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size / concurrency.")
    args = parser.parse_args()

    scorer = DatasetScorer()
    scorer.score(
        input_json=args.input_json,
        img_dir=args.img_dir,
        output_json=args.output_json,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()