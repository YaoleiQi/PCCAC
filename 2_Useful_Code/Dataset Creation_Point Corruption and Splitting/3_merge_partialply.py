"""
Merge fractured point clouds with complete point clouds.

The commented section at the bottom directly merges point clouds with the same
filename from two folders without considering rendering variants.

The currently used section considers rendering variants. Please modify the
numeric range and number of variants according to the actual dataset.
"""

import argparse
from pathlib import Path

import open3d as o3d


def read_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    """
    Read a point cloud from file.

    Args:
        path: Path to the point cloud file.

    Returns:
        Open3D point cloud object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the loaded point cloud is empty.
    """
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {path}")

    point_cloud = o3d.io.read_point_cloud(str(path))

    if len(point_cloud.points) == 0:
        raise ValueError(f"Empty point cloud: {path}")

    return point_cloud


def write_point_cloud(path: Path, point_cloud: o3d.geometry.PointCloud) -> None:
    """
    Write a point cloud to file.

    Args:
        path: Output file path.
        point_cloud: Open3D point cloud object.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), point_cloud, write_ascii=True)


def merge_rendering_variants(
    aorta_dir: Path,
    partial_dir: Path,
    output_dir: Path,
    start_index: int = 1,
    end_index: int = 1000,
    num_variants: int = 8,
) -> None:
    """
    Merge each complete aorta point cloud with multiple fractured variants.

    Expected input format:
        aorta_dir/
            1.ply
            2.ply
            ...

        partial_dir/
            1/
                0.ply
                1.ply
                ...
            2/
                0.ply
                1.ply
                ...

    Output format:
        output_dir/
            1/
                0.ply
                1.ply
                ...
            2/
                0.ply
                1.ply
                ...

    Args:
        aorta_dir: Directory containing complete aorta point clouds.
        partial_dir: Directory containing fractured point cloud variants.
        output_dir: Directory to save merged point clouds.
        start_index: Start index of point cloud filenames.
        end_index: End index of point cloud filenames, inclusive.
        num_variants: Number of fractured variants for each point cloud.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(start_index, end_index + 1):
        aorta_path = aorta_dir / f"{index}.ply"

        if not aorta_path.exists():
            print(f"Skipped missing aorta file: {aorta_path}")
            continue

        aorta_pcd = read_point_cloud(aorta_path)

        for variant_index in range(num_variants):
            partial_path = partial_dir / str(index) / f"{variant_index}.ply"

            if not partial_path.exists():
                print(f"Skipped missing partial file: {partial_path}")
                continue

            partial_pcd = read_point_cloud(partial_path)

            merged_pcd = aorta_pcd + partial_pcd

            save_path = output_dir / str(index) / f"{variant_index}.ply"
            write_point_cloud(save_path, merged_pcd)

            print(f"Merged and saved: {save_path}")


def merge_same_name_point_clouds(
    folder1: Path,
    folder2: Path,
    output_folder: Path,
) -> None:
    """
    Merge point clouds with the same filename from two folders.

    This function does not consider rendering variants. It directly merges files
    with the same name from two flat directories.

    Args:
        folder1: First point cloud folder.
        folder2: Second point cloud folder.
        output_folder: Output folder for merged point clouds.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    for file_path2 in sorted(folder2.iterdir()):
        if file_path2.suffix.lower() != ".ply":
            continue

        file_path1 = folder1 / file_path2.name

        if not file_path1.exists():
            print(f"Skipped missing file: {file_path1}")
            continue

        pcd1 = read_point_cloud(file_path1)
        pcd2 = read_point_cloud(file_path2)

        merged_pcd = pcd1 + pcd2

        output_file = output_folder / file_path2.name
        write_point_cloud(output_file, merged_pcd)

        print(f"Merged and saved: {output_file}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Merge aorta point clouds with fractured point clouds."
    )

    parser.add_argument(
        "--aorta_dir",
        type=Path,
        default=Path("new_break/center_aorta"),
        help="Directory containing complete aorta point clouds.",
    )
    parser.add_argument(
        "--partial_dir",
        type=Path,
        default=Path("new_break/partial_zg"),
        help="Directory containing fractured point cloud variants.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("new_break/partial_azg"),
        help="Directory to save merged point clouds.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="Start index of point cloud filenames.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=1000,
        help="End index of point cloud filenames, inclusive.",
    )
    parser.add_argument(
        "--num_variants",
        type=int,
        default=8,
        help="Number of rendering/fracture variants for each point cloud.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="variants",
        choices=["variants", "same_name"],
        help=(
            "Merge mode. 'variants' merges rendering variants; "
            "'same_name' merges same-name files from two folders."
        ),
    )
    parser.add_argument(
        "--folder1",
        type=Path,
        default=Path("center_aorta"),
        help="First folder used when mode is 'same_name'.",
    )
    parser.add_argument(
        "--folder2",
        type=Path,
        default=Path("center_zg"),
        help="Second folder used when mode is 'same_name'.",
    )
    parser.add_argument(
        "--same_name_output",
        type=Path,
        default=Path("center_merge"),
        help="Output folder used when mode is 'same_name'.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point.
    """
    args = parse_args()

    if args.mode == "variants":
        merge_rendering_variants(
            aorta_dir=args.aorta_dir,
            partial_dir=args.partial_dir,
            output_dir=args.output_dir,
            start_index=args.start_index,
            end_index=args.end_index,
            num_variants=args.num_variants,
        )
    else:
        merge_same_name_point_clouds(
            folder1=args.folder1,
            folder2=args.folder2,
            output_folder=args.same_name_output,
        )


if __name__ == "__main__":
    main()