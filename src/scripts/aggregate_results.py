import argparse

from mm_pipeline.aggregate import finalize_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Sort scored results and keep top-N samples.")
    parser.add_argument("--input_file", required=True, help="Path to the scored JSON file.")
    parser.add_argument("--output_file", required=True, help="Path to the cleaned output JSON file.")
    parser.add_argument("--top_n", type=int, default=10000, help="Number of records to keep.")
    args = parser.parse_args()

    finalize_dataset(
        input_file_path=args.input_file,
        output_file_path=args.output_file,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()