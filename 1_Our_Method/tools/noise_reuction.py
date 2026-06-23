from typing import Tuple

import numpy as np
import open3d as o3d


def del_zdm(
    points: np.ndarray,
    nb_neighbors: int = 200,
    std_ratio: float = 0.01,
    return_outliers: bool = False,
) -> np.ndarray:
    """
    Remove statistical outliers from a point cloud.

    This function uses Open3D's statistical outlier removal method. By default,
    it returns the denoised point cloud, i.e., the inlier points.

    Args:
        points: Input point cloud array of shape (N, 3).
        nb_neighbors: Number of neighbors used to analyze each point.
        std_ratio: Standard deviation ratio threshold. A smaller value removes
            more points as outliers.
        return_outliers: If True, return the removed outlier points instead of
            the denoised point cloud.

    Returns:
        A NumPy array containing either:
            - denoised point cloud points, if return_outliers=False
            - outlier points, if return_outliers=True

    Raises:
        ValueError: If the input point cloud is not a valid Nx3 array.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input points must have shape (N, 3).")

    if points.shape[0] == 0:
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Open3D returns:
    # - filtered_pcd: point cloud containing inlier points
    # - inlier_indices: indices of inlier points
    _, inlier_indices = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )

    all_indices = np.arange(len(pcd.points))
    outlier_indices = np.setdiff1d(all_indices, inlier_indices)

    if return_outliers:
        selected_pcd = pcd.select_by_index(outlier_indices)
    else:
        selected_pcd = pcd.select_by_index(inlier_indices)

    return np.asarray(selected_pcd.points, dtype=points.dtype)