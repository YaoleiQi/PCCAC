"""
Generate list files for the training, validation, and test sets.
"""

import argparse
from pathlib import Path
from typing import Tuple


def parse_range(range_str: str) -> Tuple[int, int]:
    """
    Parse an inclusive range string.

    Example:
        "1,700" -> (1, 700)

    Args:
        range_str: Range string in the format "start,end".

    Returns:
        Inclusive integer range.

    Raises:
        argparse.ArgumentTypeError: If the range format is invalid.
    """
    try:
        start, end = map(int, range_str.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Range must be in the format 'start,end'."
        ) from exc

    if start > end:
        raise argparse.ArgumentTypeError("Range start must be <= range end.")

    return start, end


def write_split_list(
    output_path: Path,
    index_range: Tuple[int, int],
    num_variants: int = 8,
) -> None:
    """
    Write a dataset split list file.

    Each line follows the format:

        sample_id/variant_id

    Example:
        1/0
        1/1
        ...
        2/0

    Args:
        output_path: Path to the output list file.
        index_range: Inclusive sample index range.
        num_variants: Number of partial variants per sample.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_index, end_index = index_range

    with open(output_path, "w", encoding="utf-8") as file:
        for sample_id in range(start_index, end_index + 1):
            for variant_id in range(num_variants):
                print(f"{sample_id}/{variant_id}", file=file)


def print_category_ids(index_range: Tuple[int, int]) -> None:
    """
    Print sample IDs in a Python-list-style format.

    This keeps the behavior of the original script, for example:

        "801",
        "802",
        ...

    Args:
        index_range: Inclusive sample index range.
    """
    start_index, end_index = index_range

    for sample_id in range(start_index, end_index + 1):
        print(f'\t\t\t"{sample_id}",')


def generate_all_lists(
    output_root: Path,
    train_range: Tuple[int, int],
    val_range: Tuple[int, int],
    test_range: Tuple[int, int],
    num_variants: int = 8,
) -> None:
    """
    Generate train.list, val.list, and test.list.

    Args:
        output_root: Dataset root directory.
        train_range: Inclusive index range for the training set.
        val_range: Inclusive index range for the validation set.
        test_range: Inclusive index range for the test set.
        num_variants: Number of partial variants per sample.
    """
    write_split_list(
        output_path=output_root / "train.list",
        index_range=train_range,
        num_variants=num_variants,
    )
    write_split_list(
        output_path=output_root / "val.list",
        index_range=val_range,
        num_variants=num_variants,
    )
    write_split_list(
        output_path=output_root / "test.list",
        index_range=test_range,
        num_variants=num_variants,
    )

    print(f"Generated list files under: {output_root}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate dataset list files for train, validation, and test splits."
    )

    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("CAS"),
        help="Dataset root directory where list files will be saved.",
    )
    parser.add_argument(
        "--train_range",
        type=parse_range,
        default=(1, 700),
        help="Inclusive training index range, format: start,end.",
    )
    parser.add_argument(
        "--val_range",
        type=parse_range,
        default=(701, 800),
        help="Inclusive validation index range, format: start,end.",
    )
    parser.add_argument(
        "--test_range",
        type=parse_range,
        default=(801, 1000),
        help="Inclusive test index range, format: start,end.",
    )
    parser.add_argument(
        "--num_variants",
        type=int,
        default=8,
        help="Number of partial variants per sample.",
    )
    parser.add_argument(
        "--print_ids",
        action="store_true",
        help="Print sample IDs instead of writing list files.",
    )
    parser.add_argument(
        "--print_split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split IDs to print when --print_ids is enabled.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point.
    """
    args = parse_args()

    split_ranges = {
        "train": args.train_range,
        "val": args.val_range,
        "test": args.test_range,
    }

    if args.print_ids:
        print_category_ids(split_ranges[args.print_split])
    else:
        generate_all_lists(
            output_root=args.output_root,
            train_range=args.train_range,
            val_range=args.val_range,
            test_range=args.test_range,
            num_variants=args.num_variants,
        )


if __name__ == "__main__":
    main()