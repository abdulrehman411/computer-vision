import argparse
import json

from src.inference.predict import predict_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    result = predict_image(args.image, args.checkpoint)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
