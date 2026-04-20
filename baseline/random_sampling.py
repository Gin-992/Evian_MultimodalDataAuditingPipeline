import argparse

from utils import load_json, save_json, sample_without_replacement, copy_images


def parse_args():
    parser = argparse.ArgumentParser(description="Randomly sample entries from a JSON dataset")
    parser.add_argument("--json_path", "-j", required=True)
    parser.add_argument("--output_json", "-o", required=True)
    parser.add_argument("--sample_size", "-n", type=int, default=10000)
    parser.add_argument("--seed", "-s", type=int, default=None)
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--output_image_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_image_dir and not args.image_dir:
        raise ValueError("--image_dir is required when --output_image_dir is set")

    data = load_json(args.json_path)
    sampled = sample_without_replacement(data, args.sample_size, args.seed)
    save_json(sampled, args.output_json)
    print(f"Saved {len(sampled)} sampled entries to: {args.output_json}")

    if args.output_image_dir:
        copy_images(sampled, args.image_dir, args.output_image_dir)
        print(f"Copied sampled images to: {args.output_image_dir}")


if __name__ == "__main__":
    main()
