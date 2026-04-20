import argparse

from mm_pipeline.low_quality import LowQualityGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate low-quality image-text responses.")
    parser.add_argument("--input_file", required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", required=True, help="Path to the output JSON file.")
    parser.add_argument("--num_samples", type=int, default=None, help="Target total number of samples.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size / concurrency.")
    args = parser.parse_args()

    generator = LowQualityGenerator()
    generator.generate(
        input_file=args.input_file,
        output_file=args.output_file,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()