from typing import Optional

import numpy as np
import open3d as o3d

from tools.noise_reuction import del_zdm


def to_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Convert a NumPy point array to an Open3D point cloud.

    Args:
        points: Point cloud array of shape (N, 3).

    Returns:
        Open3D point cloud object.
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    return point_cloud


def remove_overlapping_points(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Remove points from the source point cloud if they overlap with the reference point cloud.

    For each point in `source_points`, this function searches neighboring points
    in `reference_points` within the given radius. If at least one neighbor is
    found, the source point is considered overlapping and will be removed.

    Args:
        source_points: Source point cloud of shape (N, 3).
        reference_points: Reference point cloud of shape (M, 3).
        radius: Search radius for determining overlapping points.

    Returns:
        Filtered source point cloud with overlapping points removed.

    Raises:
        ValueError: If radius is not positive.
    """
    if radius <= 0:
        raise ValueError("radius must be positive.")

    if source_points.shape[0] == 0:
        return source_points

    if reference_points.shape[0] == 0:
        return source_points

    source_cloud = to_point_cloud(source_points)
    reference_cloud = to_point_cloud(reference_points)

    kdtree_reference = o3d.geometry.KDTreeFlann(reference_cloud)

    indices_to_remove = []

    for i, point in enumerate(source_cloud.points):
        _, neighbor_indices, _ = kdtree_reference.search_radius_vector_3d(
            point,
            radius,
        )

        if len(neighbor_indices) > 0:
            indices_to_remove.append(i)

    if len(indices_to_remove) == 0:
        return source_points

    filtered_points = np.delete(
        np.asarray(source_cloud.points),
        indices_to_remove,
        axis=0,
    )

    return np.asarray(filtered_points, dtype=source_points.dtype)


def get_miss_g(
    points_c: np.ndarray,
    points_p: np.ndarray,
    radius: float = 0.00001,
) -> np.ndarray:
    """
    Extract missing points by removing points that already exist in the partial point cloud.

    This function compares the complete point cloud `points_c` with the partial
    point cloud `points_p`. Points in `points_c` that have neighboring points in
    `points_p` within the given radius are removed. The remaining points are
    treated as missing points.

    Args:
        points_c: Complete point cloud of shape (N, 3).
        points_p: Partial point cloud of shape (M, 3).
        radius: Search radius used to identify overlapping points.

    Returns:
        Missing point cloud array.
    """
    return remove_overlapping_points(
        source_points=points_c,
        reference_points=points_p,
        radius=radius,
    )


def get_miss_o(
    points_c: np.ndarray,
    points_p: np.ndarray,
    radius: float = 0.006,
) -> np.ndarray:
    """
    Extract missing points after noise or redundant structure removal.

    Both input point clouds are first processed by `del_zdm()`, and then
    overlapping points are removed from the processed complete point cloud.

    Args:
        points_c: Complete point cloud of shape (N, 3).
        points_p: Partial point cloud of shape (M, 3).
        radius: Search radius used to identify overlapping points.

    Returns:
        Missing point cloud array after preprocessing.
    """
    complete_points = del_zdm(points_c)
    partial_points = del_zdm(points_p)

    return remove_overlapping_points(
        source_points=complete_points,
        reference_points=partial_points,
        radius=radius,
    )