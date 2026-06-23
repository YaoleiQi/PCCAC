import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import open3d as o3d


def read_point_cloud(path: Path) -> np.ndarray:
    """
    Read a point cloud file and return its points as a NumPy array.

    Args:
        path: Path to the point cloud file.

    Returns:
        Point cloud array of shape (N, 3).

    Raises:
        FileNotFoundError: If the point cloud file does not exist.
        ValueError: If the loaded point cloud is empty.
    """
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {path}")

    point_cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(point_cloud.points, dtype=np.float64)

    if points.shape[0] == 0:
        raise ValueError(f"Empty point cloud: {path}")

    return points


def write_point_cloud(path: Path, points: np.ndarray) -> None:
    """
    Write a NumPy point cloud array to a PLY/PCD file.

    Args:
        path: Output file path.
        points: Point cloud array of shape (N, 3).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    o3d.io.write_point_cloud(str(path), point_cloud, write_ascii=True)


def compute_normalization_params(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute normalization parameters from a reference point cloud.

    The centroid is computed as the mean coordinate of all points.
    The scale is computed as the maximum Euclidean distance from points to the
    centroid after centering.

    Args:
        points: Reference point cloud of shape (N, 3).

    Returns:
        A tuple containing:
            - centroid of shape (3,)
            - scale factor

    Raises:
        ValueError: If the computed scale is zero.
    """
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    scale = float(np.max(np.linalg.norm(centered_points, axis=1)))

    if scale <= 0.0:
        raise ValueError("Invalid normalization scale. The point cloud may be degenerate.")

    return centroid, scale


def normalize_points(
    points: np.ndarray,
    centroid: np.ndarray,
    scale: float,
    shrink_factor: float = 2.0,
) -> np.ndarray:
    """
    Normalize a point cloud using a given centroid and scale.

    The normalization is:

        normalized = (points - centroid) / scale / shrink_factor

    Args:
        points: Input point cloud of shape (N, 3).
        centroid: Centroid used for centering, shape (3,).
        scale: Scale factor used for normalization.
        shrink_factor: Additional shrink factor. The original script uses 2.0.

    Returns:
        Normalized point cloud of shape (N, 3).
    """
    return (points - centroid) / scale / shrink_factor


def normalize_dataset(
    combined_dir: Path,
    aorta_dir: Path,
    main_branch_dir: Path,
    save_combined_dir: Path,
    save_aorta_dir: Path,
    save_main_branch_dir: Path,
    centroid_map_path: Path,
    scale_map_path: Path,
    shrink_factor: float = 2.0,
) -> None:
    """
    Normalize three related point cloud folders with shared parameters.

    For each file, the normalization centroid and scale are computed from the
    combined point cloud. The same centroid and scale are then applied to:
        - combined point cloud
        - aorta point cloud
        - main-branch point cloud

    Args:
        combined_dir: Directory containing combined point clouds.
        aorta_dir: Directory containing aorta point clouds.
        main_branch_dir: Directory containing main-branch point clouds.
        save_combined_dir: Output directory for normalized combined point clouds.
        save_aorta_dir: Output directory for normalized aorta point clouds.
        save_main_branch_dir: Output directory for normalized main-branch point clouds.
        centroid_map_path: Output path for saving centroids.
        scale_map_path: Output path for saving scale factors.
        shrink_factor: Additional scaling factor after max-radius normalization.
    """
    point_cloud_files = sorted(
        file for file in combined_dir.iterdir() if file.is_file()
    )

    if len(point_cloud_files) == 0:
        raise ValueError(f"No point cloud files found in: {combined_dir}")

    centroids: List[np.ndarray] = []
    scales: List[float] = []

    for combined_path in point_cloud_files:
        filename = combined_path.name

        aorta_path = aorta_dir / filename
        main_branch_path = main_branch_dir / filename

        save_combined_path = save_combined_dir / filename
        save_aorta_path = save_aorta_dir / filename
        save_main_branch_path = save_main_branch_dir / filename

        combined_points = read_point_cloud(combined_path)
        aorta_points = read_point_cloud(aorta_path)
        main_branch_points = read_point_cloud(main_branch_path)

        centroid, scale = compute_normalization_params(combined_points)

        normalized_combined = normalize_points(
            combined_points,
            centroid,
            scale,
            shrink_factor,
        )
        normalized_aorta = normalize_points(
            aorta_points,
            centroid,
            scale,
            shrink_factor,
        )
        normalized_main_branch = normalize_points(
            main_branch_points,
            centroid,
            scale,
            shrink_factor,
        )

        write_point_cloud(save_combined_path, normalized_combined)
        write_point_cloud(save_aorta_path, normalized_aorta)
        write_point_cloud(save_main_branch_path, normalized_main_branch)

        centroids.append(centroid)
        scales.append(scale)

        print(f"Processed: {filename}")

    np.savetxt(
        str(centroid_map_path),
        np.asarray(centroids),
        fmt="%.10f",
    )
    np.savetxt(
        str(scale_map_path),
        np.asarray(scales),
        fmt="%.18f",
    )

    print(f"Saved centroid map to: {centroid_map_path}")
    print(f"Saved scale map to: {scale_map_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Normalize point cloud datasets using shared centroid and scale "
            "computed from the combined point cloud."
        )
    )

    parser.add_argument(
        "--combined_dir",
        type=Path,
        default=Path("complete_azg"),
        help="Directory containing combined point clouds.",
    )
    parser.add_argument(
        "--aorta_dir",
        type=Path,
        default=Path("aorta"),
        help="Directory containing aorta point clouds.",
    )
    parser.add_argument(
        "--main_branch_dir",
        type=Path,
        default=Path("complete_zg"),
        help="Directory containing main-branch point clouds.",
    )

    parser.add_argument(
        "--save_combined_dir",
        type=Path,
        default=Path("center_azg"),
        help="Output directory for normalized combined point clouds.",
    )
    parser.add_argument(
        "--save_aorta_dir",
        type=Path,
        default=Path("center_aorta"),
        help="Output directory for normalized aorta point clouds.",
    )
    parser.add_argument(
        "--save_main_branch_dir",
        type=Path,
        default=Path("center_zg"),
        help="Output directory for normalized main-branch point clouds.",
    )

    parser.add_argument(
        "--centroid_map_path",
        type=Path,
        default=Path("map_centroid.txt"),
        help="Output path for saved centroid values.",
    )
    parser.add_argument(
        "--scale_map_path",
        type=Path,
        default=Path("map_m.txt"),
        help="Output path for saved scale values.",
    )
    parser.add_argument(
        "--shrink_factor",
        type=float,
        default=2.0,
        help="Additional shrink factor after max-radius normalization.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point.
    """
    args = parse_args()

    normalize_dataset(
        combined_dir=args.combined_dir,
        aorta_dir=args.aorta_dir,
        main_branch_dir=args.main_branch_dir,
        save_combined_dir=args.save_combined_dir,
        save_aorta_dir=args.save_aorta_dir,
        save_main_branch_dir=args.save_main_branch_dir,
        centroid_map_path=args.centroid_map_path,
        scale_map_path=args.scale_map_path,
        shrink_factor=args.shrink_factor,
    )


if __name__ == "__main__":
    main()