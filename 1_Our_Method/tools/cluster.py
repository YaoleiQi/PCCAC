from typing import Iterable, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial.distance import pdist


def compute_average_distance(pcd: o3d.geometry.PointCloud) -> float:
    """
    Compute the average pairwise Euclidean distance of a point cloud.

    Self-to-self distances are excluded from the computation.

    Args:
        pcd: Open3D point cloud object.

    Returns:
        Average pairwise Euclidean distance.

    Raises:
        ValueError: If the point cloud contains fewer than two points.
    """
    points = np.asarray(pcd.points)

    if points.shape[0] < 2:
        raise ValueError("At least two points are required to compute average distance.")

    # pdist computes pairwise distances without including self-to-self distances.
    distances = pdist(points, metric="euclidean")

    return float(np.mean(distances))


def seg_zg(
    pc: np.ndarray,
    seed1: float,
    seed2: int,
) -> Tuple[np.ndarray, int]:
    """
    Segment the main vessel branch from a point cloud using DBSCAN clustering.

    The DBSCAN radius is determined by the average pairwise distance divided by
    `seed1`. Clusters with more than `seed2` points are treated as the main
    branch, while smaller clusters are treated as other structures.

    Args:
        pc: Input point cloud array of shape (N, 3).
        seed1: Scale factor used to determine the DBSCAN eps value.
        seed2: Minimum cluster size threshold for identifying the main branch.

    Returns:
        A tuple containing:
            - segmented main-branch point cloud of shape (M, 3)
            - number of points in the segmented main branch

    Raises:
        ValueError: If the input point cloud is empty or seed1 is not positive.
    """
    if pc.shape[0] == 0:
        raise ValueError("Input point cloud is empty.")

    if seed1 <= 0:
        raise ValueError("seed1 must be positive.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    average_dist = compute_average_distance(pcd)

    # Cluster the point cloud with DBSCAN.
    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=average_dist / seed1,
            min_points=1,
        )
    )

    new_labels = np.zeros_like(labels)

    unique_labels, counts = np.unique(labels, return_counts=True)

    for label, count in zip(unique_labels, counts):
        mask = labels == label

        if count > seed2:
            # Main branch.
            new_labels[mask] = 1
        else:
            # Other small branches or noisy structures.
            new_labels[mask] = 2

    main_branch_mask = new_labels == 1
    main_branch = pc[main_branch_mask]
    main_branch_count = int(np.sum(main_branch_mask))

    return main_branch, main_branch_count


def get_zg(
    pc: np.ndarray,
    seed_candidates: Iterable[float] = (38,),
    min_cluster_size: int = 40,
    target_points: int = 1024,
) -> np.ndarray:
    """
    Select the best main-branch segmentation result.

    Different DBSCAN scale factors can be tested, and the result whose point
    count is closest to `target_points` will be selected.

    Args:
        pc: Input point cloud array of shape (N, 3).
        seed_candidates: Candidate scale factors for DBSCAN eps computation.
        min_cluster_size: Minimum cluster size threshold for the main branch.
        target_points: Target number of points for the segmented main branch.

    Returns:
        Segmented main-branch point cloud.
    """
    min_diff = float("inf")
    best_result = pc

    for seed in seed_candidates:
        main_branch, num_points = seg_zg(pc, seed, min_cluster_size)
        diff = abs(num_points - target_points)

        if diff < min_diff:
            min_diff = diff
            best_result = main_branch

    return best_result