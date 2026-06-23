import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d


def to_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Convert a NumPy point array to an Open3D point cloud.

    Args:
        points: Point cloud array of shape (N, 3).

    Returns:
        Open3D point cloud object.
    """
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def read_point_cloud(path: Path) -> np.ndarray:
    """
    Read a point cloud file and convert it to a NumPy array.

    Args:
        path: Path to the point cloud file.

    Returns:
        Point cloud coordinates with shape (N, 3), dtype float32.
    """
    point_cloud = o3d.io.read_point_cloud(str(path))
    return np.asarray(point_cloud.points, dtype=np.float32)


def f_score(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.01) -> float:
    """
    Compute the F-score between predicted and ground-truth point clouds.

    A point is considered correctly matched if its nearest-neighbor distance is
    smaller than the given threshold.

    Args:
        pred: Predicted point cloud of shape (N, 3).
        gt: Ground-truth point cloud of shape (M, 3).
        threshold: Distance threshold used to compute precision and recall.

    Returns:
        F-score value.
    """
    pred_pc = to_point_cloud(pred)
    gt_pc = to_point_cloud(gt)

    # Distance from each predicted point to the ground-truth point cloud.
    pred_to_gt = pred_pc.compute_point_cloud_distance(gt_pc)

    # Distance from each ground-truth point to the predicted point cloud.
    gt_to_pred = gt_pc.compute_point_cloud_distance(pred_pc)

    if len(pred_to_gt) == 0 or len(gt_to_pred) == 0:
        return 0.0

    precision = float(sum(d < threshold for d in pred_to_gt)) / float(len(pred_to_gt))
    recall = float(sum(d < threshold for d in gt_to_pred)) / float(len(gt_to_pred))

    if precision + recall == 0.0:
        return 0.0

    return 2.0 * recall * precision / (recall + precision)


def fidelity(input_pc: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute the fidelity between input and predicted point clouds.

    Fidelity is defined as the average nearest-neighbor distance from each input
    point to the predicted point cloud.

    Args:
        input_pc: Input partial point cloud of shape (N, 3).
        pred: Predicted point cloud of shape (M, 3).

    Returns:
        Average nearest-neighbor distance from input_pc to pred.
    """
    input_cloud = to_point_cloud(input_pc)
    pred_cloud = to_point_cloud(pred)

    distances = input_cloud.compute_point_cloud_distance(pred_cloud)
    if len(distances) == 0:
        return 0.0

    return float(np.mean(np.asarray(distances, dtype=np.float32)))


def bidirectional_mse(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """
    Compute bidirectional mean squared nearest-neighbor distances.

    Args:
        pred: Predicted point cloud of shape (N, 3).
        gt: Ground-truth point cloud of shape (M, 3).

    Returns:
        A tuple containing:
            - Mean squared distance from pred to gt.
            - Mean squared distance from gt to pred.
    """
    pred_pc = to_point_cloud(pred)
    gt_pc = to_point_cloud(gt)

    pred_to_gt = np.asarray(pred_pc.compute_point_cloud_distance(gt_pc), dtype=np.float32)
    gt_to_pred = np.asarray(gt_pc.compute_point_cloud_distance(pred_pc), dtype=np.float32)

    if len(pred_to_gt) == 0 or len(gt_to_pred) == 0:
        return 0.0, 0.0

    pred_to_gt_mse = float(np.mean(np.square(pred_to_gt)))
    gt_to_pred_mse = float(np.mean(np.square(gt_to_pred)))

    return pred_to_gt_mse, gt_to_pred_mse


def evaluate(
    gt_dir: Path,
    output_dir: Path,
    input_dir: Path,
    threshold: float = 0.01,
) -> None:
    """
    Evaluate predicted point clouds using F-score, fidelity, and bidirectional MSE.

    The file names in gt_dir, output_dir, and input_dir are expected to match.

    Args:
        gt_dir: Directory containing ground-truth point clouds.
        output_dir: Directory containing predicted point clouds.
        input_dir: Directory containing input partial point clouds.
        threshold: Distance threshold used for F-score.
    """
    gt_files = sorted(gt_dir.iterdir())

    if len(gt_files) == 0:
        raise ValueError(f"No point cloud files found in ground-truth directory: {gt_dir}")

    total_f_score = 0.0
    total_fidelity = 0.0
    total_pred_to_gt = 0.0
    total_gt_to_pred = 0.0
    valid_count = 0

    for gt_path in gt_files:
        if not gt_path.is_file():
            continue

        output_path = output_dir / gt_path.name
        input_path = input_dir / gt_path.name

        if not output_path.exists():
            print(f"[Warning] Missing prediction file: {output_path}")
            continue

        if not input_path.exists():
            print(f"[Warning] Missing input file: {input_path}")
            continue

        gt = read_point_cloud(gt_path)
        pred = read_point_cloud(output_path)
        input_pc = read_point_cloud(input_path)

        total_f_score += f_score(pred, gt, threshold)
        total_fidelity += fidelity(input_pc, pred)

        pred_to_gt, gt_to_pred = bidirectional_mse(pred, gt)
        total_pred_to_gt += pred_to_gt
        total_gt_to_pred += gt_to_pred

        valid_count += 1

    if valid_count == 0:
        raise ValueError("No valid point cloud pairs were found for evaluation.")

    print(f"ave_f_score: {total_f_score / valid_count:.4f}")
    print(f"ave_fidelity: {total_fidelity / valid_count:.6f}")
    print(
        "pred-gt/gt-pred: "
        f"{total_pred_to_gt / valid_count:.8f}/"
        f"{total_gt_to_pred / valid_count:.8f}"
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate point cloud reconstruction results."
    )

    parser.add_argument(
        "--gt_dir",
        type=Path,
        required=True,
        help="Directory containing ground-truth point clouds.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory containing predicted point clouds.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing input partial point clouds.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Distance threshold used to compute F-score.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for point cloud evaluation.
    """
    args = parse_args()

    evaluate(
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()