import argparse
import yaml

from src.eval.error_analysis import save_misclassifications


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    save_misclassifications(config, args.checkpoint, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
