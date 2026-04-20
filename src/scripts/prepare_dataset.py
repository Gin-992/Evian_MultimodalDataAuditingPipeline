import argparse

from mm_pipeline.prepare import combine_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine high-quality and low-quality datasets.")
    parser.add_argument("--original_data", required=True, help="Path to the original JSON file.")
    parser.add_argument("--low_quality_data", required=True, help="Path to the low-quality JSON file.")
    parser.add_argument("--output_file", required=True, help="Path to the combined output JSON file.")
    parser.add_argument("--num_original_samples", type=int, default=100000, help="Number of high-quality samples to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    combine_datasets(
        original_data_path=args.original_data,
        low_quality_data_path=args.low_quality_data,
        output_file_path=args.output_file,
        num_original_samples=args.num_original_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()