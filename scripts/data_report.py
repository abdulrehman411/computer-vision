import argparse

from src.data.analysis import write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    write_report(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
