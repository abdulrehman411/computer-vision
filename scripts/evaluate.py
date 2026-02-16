import argparse

from src.eval.evaluate import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    evaluate_checkpoint(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
