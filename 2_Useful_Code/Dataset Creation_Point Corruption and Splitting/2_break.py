"""
Point Cloud Corruption / Fracture Generation Script

Description:
    Convert a complete point cloud dataset into a corrupted point cloud dataset
    with random fractures or missing regions.

    For each input point cloud, multiple fractured variants with different
    degrees of corruption will be generated.

Main Parameters:
    - rendering:
        Controls the number of fractured variants generated for each point cloud.
        The default value is 8.

    - ranbrokennum:
        Controls the number of fractured regions in each variant.
        By default, it is computed by the getbrokennum function.

        When using getbrokennum, the number of fractured regions for the
        8 variants is:
            1, 2, 2, 3, 3, 4, 4, 5

    - arr.shape[0] * 0.1:
        Controls the total proportion of points to be removed from each point
        cloud. The default removal ratio is 10%.

Usage:
    1. Place the complete point clouds in the 'center_zg' directory.
    2. The script will generate corresponding fractured versions in the
       'partial_zg' directory.
    3. For each input point cloud, multiple subfolders will be generated.
       Each subfolder contains different fractured variants.

Output Format:
    partial_zg/
        ├── point_cloud_1/
        │   ├── 0.ply  # 1st fractured variant with 1 fractured region
        │   ├── 1.ply  # 2nd fractured variant with 2 fractured regions
        │   ├── 2.ply  # 3rd fractured variant with 2 fractured regions
        │   └── ...
        ├── point_cloud_2/
        └── ...
"""

"""
Point Cloud Corruption / Fracture Generation Script

Description:
    Convert a complete point cloud dataset into a corrupted point cloud dataset
    with random fractures or missing regions.

    For each input point cloud, multiple fractured variants with different
    degrees of corruption will be generated.

Main Parameters:
    - num_variants:
        Controls the number of fractured variants generated for each point cloud.
        The default value is 8.

    - broken_region_num:
        Controls the number of fractured regions in each variant.
        By default, it is computed by the get_broken_region_num function.

        When using get_broken_region_num, the number of fractured regions for
        the 8 variants is:
            1, 2, 2, 3, 3, 4, 4, 5

    - remove_ratio:
        Controls the total proportion of points to be removed from each point
        cloud. The default removal ratio is 10%.

Usage:
    1. Place the complete point clouds in the 'center_zg' directory.
    2. The script will generate corresponding fractured versions in the
       'partial_zg' directory.
    3. For each input point cloud, a subfolder will be generated.
       Each subfolder contains different fractured variants.

Output Format:
    partial_zg/
        ├── point_cloud_1/
        │   ├── 0.ply  # 1st fractured variant with 1 fractured region
        │   ├── 1.ply  # 2nd fractured variant with 2 fractured regions
        │   ├── 2.ply  # 3rd fractured variant with 2 fractured regions
        │   └── ...
        ├── point_cloud_2/
        └── ...
"""

import argparse
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d as o3d


def get_broken_region_num(index: int) -> int:
    """
    Compute the number of fractured regions for a given variant index.

    For indices 0 to 7, this function returns:
        1, 2, 2, 3, 3, 4, 4, 5

    Args:
        index: Variant index.

    Returns:
        Number of fractured regions.
    """
    return (index // 2) + 1 if index % 2 == 0 else (index // 2) + 2


def split_randomly(total_num: int, num_parts: int) -> List[int]:
    """
    Randomly split an integer into several positive parts.

    Args:
        total_num: Total number to be split.
        num_parts: Number of parts.

    Returns:
        A list of positive integers whose sum equals total_num.

    Raises:
        ValueError: If total_num is smaller than num_parts.
    """
    if num_parts <= 0:
        raise ValueError("num_parts must be positive.")

    if total_num < num_parts:
        raise ValueError(
            f"total_num must be greater than or equal to num_parts, "
            f"but got total_num={total_num}, num_parts={num_parts}."
        )

    result = []
    remaining = total_num

    for i in range(num_parts - 1):
        split = random.randint(1, remaining - num_parts + i + 1)
        result.append(split)
        remaining -= split

    result.append(remaining)

    return result


def find_nearest_neighbor_indices(
    point_cloud: np.ndarray,
    point: np.ndarray,
    num_neighbors: int,
) -> np.ndarray:
    """
    Find nearest neighbor indices of a query point in a point cloud.

    Args:
        point_cloud: Point cloud array of shape (N, 3).
        point: Query point of shape (3,).
        num_neighbors: Number of nearest neighbors to find.

    Returns:
        Indices of nearest neighbor points.
    """
    num_neighbors = min(num_neighbors, point_cloud.shape[0])

    distances = np.linalg.norm(point_cloud - point, axis=1)
    nearest_indices = np.argsort(distances)[:num_neighbors]

    return nearest_indices


def read_point_cloud(path: Path) -> np.ndarray:
    """
    Read a point cloud file as a NumPy array.

    Args:
        path: Path to the input point cloud file.

    Returns:
        Point cloud array of shape (N, 3).

    Raises:
        ValueError: If the loaded point cloud is empty.
    """
    point_cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(point_cloud.points)

    if points.shape[0] == 0:
        raise ValueError(f"Empty point cloud: {path}")

    return points


def write_point_cloud(path: Path, points: np.ndarray) -> None:
    """
    Write a NumPy point cloud array to a point cloud file.

    Args:
        path: Output point cloud file path.
        points: Point cloud array of shape (N, 3).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    o3d.io.write_point_cloud(str(path), point_cloud, write_ascii=True)


def corrupt_point_cloud(
    points: np.ndarray,
    variant_index: int,
    remove_ratio: float = 0.1,
) -> np.ndarray:
    """
    Generate a corrupted point cloud by removing several local point regions.

    Args:
        points: Input complete point cloud of shape (N, 3).
        variant_index: Index of the corrupted variant.
        remove_ratio: Total ratio of points to remove.

    Returns:
        Corrupted point cloud array.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input points must have shape (N, 3).")

    if not 0.0 < remove_ratio < 1.0:
        raise ValueError("remove_ratio must be in the range (0, 1).")

    corrupted_points = points.copy()

    broken_region_num = get_broken_region_num(variant_index)
    total_remove_num = int(corrupted_points.shape[0] * remove_ratio)

    if total_remove_num <= 0:
        return corrupted_points

    # Ensure that the number of removed points can be split into positive parts.
    total_remove_num = max(total_remove_num, broken_region_num)

    broken_sizes = split_randomly(total_remove_num, broken_region_num)

    for broken_size in broken_sizes:
        if corrupted_points.shape[0] == 0:
            break

        # Randomly select a center point and remove its nearest neighbors.
        center_index = random.randint(0, corrupted_points.shape[0] - 1)
        center_point = corrupted_points[center_index]

        remove_indices = find_nearest_neighbor_indices(
            point_cloud=corrupted_points,
            point=center_point,
            num_neighbors=broken_size,
        )

        keep_mask = np.ones(corrupted_points.shape[0], dtype=bool)
        keep_mask[remove_indices] = False
        corrupted_points = corrupted_points[keep_mask]

    return corrupted_points


def generate_corrupted_dataset(
    input_dir: Path,
    output_dir: Path,
    num_variants: int = 8,
    remove_ratio: float = 0.1,
    seed: Optional[int] = None,
) -> None:
    """
    Generate corrupted point cloud variants for all point clouds in a directory.

    Args:
        input_dir: Directory containing complete point clouds.
        output_dir: Directory to save corrupted point clouds.
        num_variants: Number of corrupted variants per input point cloud.
        remove_ratio: Ratio of points to remove from each point cloud.
        seed: Random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    point_cloud_files = sorted(
        file for file in input_dir.iterdir() if file.is_file()
    )

    if len(point_cloud_files) == 0:
        raise ValueError(f"No point cloud files found in: {input_dir}")

    for point_cloud_path in point_cloud_files:
        points = read_point_cloud(point_cloud_path)

        save_dir = output_dir / point_cloud_path.stem
        save_dir.mkdir(parents=True, exist_ok=True)

        for variant_index in range(num_variants):
            corrupted_points = corrupt_point_cloud(
                points=points,
                variant_index=variant_index,
                remove_ratio=remove_ratio,
            )

            save_path = save_dir / f"{variant_index}.ply"
            write_point_cloud(save_path, corrupted_points)

            print(f"{save_path} {corrupted_points.shape[0]}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate randomly fractured point clouds."
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("center_zg"),
        help="Directory containing complete point clouds.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("partial_zg"),
        help="Directory to save corrupted point clouds.",
    )
    parser.add_argument(
        "--num_variants",
        type=int,
        default=8,
        help="Number of corrupted variants generated for each point cloud.",
    )
    parser.add_argument(
        "--remove_ratio",
        type=float,
        default=0.1,
        help="Total ratio of points to remove from each point cloud.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point.
    """
    args = parse_args()

    generate_corrupted_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_variants=args.num_variants,
        remove_ratio=args.remove_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()