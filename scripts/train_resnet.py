import argparse

from src.training.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train_from_config(args.config, model_override="resnet18")


if __name__ == "__main__":
    main()
