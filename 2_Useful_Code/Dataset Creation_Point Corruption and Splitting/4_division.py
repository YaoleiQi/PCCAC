"""
Split the dataset into training, validation, and test sets.
"""

import argparse
import shutil
from pathlib import Path
from typing import Tuple


def make_dir(path: Path) -> None:
    """
    Create a directory if it does not exist.

    Args:
        path: Directory path.
    """
    path.mkdir(parents=True, exist_ok=True)


def prepare_output_dirs(output_root: Path) -> None:
    """
    Create output directories for train/valid/test splits.

    The directory structure is:

        output_root/
            train/
                complete/
                partial/
            valid/
                complete/
                partial/
            test/
                complete/
                partial/

    Args:
        output_root: Root directory for the split dataset.
    """
    dirs = [
        output_root / "train" / "complete",
        output_root / "train" / "partial",
        output_root / "valid" / "complete",
        output_root / "valid" / "partial",
        output_root / "test" / "complete",
        output_root / "test" / "partial",
    ]

    for directory in dirs:
        make_dir(directory)


def copy_one_sample(
    index: int,
    complete_path: Path,
    partial_path: Path,
    output_root: Path,
    split: str,
    overwrite: bool = False,
) -> None:
    """
    Copy one complete point cloud and its corresponding partial variants.

    Args:
        index: Sample index.
        complete_path: Directory containing complete point clouds.
        partial_path: Directory containing partial point cloud folders.
        output_root: Root directory for the split dataset.
        split: Dataset split name, e.g., train, valid, or test.
        overwrite: Whether to overwrite existing partial folders.

    Raises:
        FileNotFoundError: If the complete file or partial folder does not exist.
        FileExistsError: If the destination partial folder exists and overwrite=False.
    """
    filename = f"{index}.ply"

    src_complete_file = complete_path / filename
    src_partial_dir = partial_path / str(index)

    dst_complete_file = output_root / split / "complete" / filename
    dst_partial_dir = output_root / split / "partial" / str(index)

    if not src_complete_file.exists():
        raise FileNotFoundError(f"Complete point cloud not found: {src_complete_file}")

    if not src_partial_dir.exists():
        raise FileNotFoundError(f"Partial point cloud folder not found: {src_partial_dir}")

    shutil.copyfile(src_complete_file, dst_complete_file)

    if dst_partial_dir.exists():
        if overwrite:
            shutil.rmtree(dst_partial_dir)
        else:
            raise FileExistsError(
                f"Destination partial folder already exists: {dst_partial_dir}. "
                f"Use --overwrite to replace it."
            )

    shutil.copytree(src_partial_dir, dst_partial_dir)


def split_dataset(
    complete_path: Path,
    partial_path: Path,
    output_root: Path,
    train_range: Tuple[int, int] = (1, 700),
    valid_range: Tuple[int, int] = (701, 800),
    test_range: Tuple[int, int] = (801, 1000),
    overwrite: bool = False,
) -> None:
    """
    Split the dataset into training, validation, and test sets.

    Args:
        complete_path: Directory containing complete point clouds.
        partial_path: Directory containing partial point cloud folders.
        output_root: Root directory for the split dataset.
        train_range: Inclusive index range for the training set.
        valid_range: Inclusive index range for the validation set.
        test_range: Inclusive index range for the test set.
        overwrite: Whether to overwrite existing partial folders.
    """
    prepare_output_dirs(output_root)

    split_configs = [
        ("train", train_range),
        ("valid", valid_range),
        ("test", test_range),
    ]

    for split, index_range in split_configs:
        start_index, end_index = index_range

        for index in range(start_index, end_index + 1):
            copy_one_sample(
                index=index,
                complete_path=complete_path,
                partial_path=partial_path,
                output_root=output_root,
                split=split,
                overwrite=overwrite,
            )

            print(f"Copied sample {index} to {split} set.")


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
        argparse.ArgumentTypeError: If the format is invalid.
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


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Split the point cloud dataset into train, validation, and test sets."
    )

    parser.add_argument(
        "--complete_path",
        type=Path,
        default=Path("new_break/center_azg"),
        help="Directory containing complete point clouds.",
    )
    parser.add_argument(
        "--partial_path",
        type=Path,
        default=Path("new_break/partial_azg"),
        help="Directory containing partial point cloud folders.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("CAS"),
        help="Output root directory for the split dataset.",
    )
    parser.add_argument(
        "--train_range",
        type=parse_range,
        default=(1, 700),
        help="Inclusive index range for training set, format: start,end.",
    )
    parser.add_argument(
        "--valid_range",
        type=parse_range,
        default=(701, 800),
        help="Inclusive index range for validation set, format: start,end.",
    )
    parser.add_argument(
        "--test_range",
        type=parse_range,
        default=(801, 1000),
        help="Inclusive index range for test set, format: start,end.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing partial folders if they already exist.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point.
    """
    args = parse_args()

    split_dataset(
        complete_path=args.complete_path,
        partial_path=args.partial_path,
        output_root=args.output_root,
        train_range=args.train_range,
        valid_range=args.valid_range,
        test_range=args.test_range,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()