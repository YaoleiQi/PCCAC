import argparse
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import open3d as o3d
import torch
import torch.utils.data as Data

from dataset import Coronary
from dcd.dcd import calc_dcd
from metrics.metric import emd as metric_emd
from metrics.metric import f_score, l1_cd, l2_cd
from models import TSRNet
from visualization import plot_pcd_one_view


CATEGORIES = ["coronary"]
BEST_CHECKPOINT_NAME = "best_all_l1_cd.pth"


PathLike = Union[str, Path]


def make_dir(dir_path: PathLike) -> None:
    """
    Create a directory if it does not exist.

    Args:
        dir_path: Directory path to create.
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def export_ply(filename: PathLike, points: np.ndarray) -> None:
    """
    Export a point cloud to a PLY file.

    Args:
        filename: Output PLY file path.
        points: Point cloud array of shape (N, 3).
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(str(filename), pc, write_ascii=True)


def to_scalar(value: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
    """
    Convert a metric output to a scalar tensor.

    Some metric functions may return a tensor directly, while others may return
    a tuple whose first element is the target metric.

    Args:
        value: Metric output.

    Returns:
        Scalar tensor.
    """
    if isinstance(value, tuple):
        value = value[0]

    if value.ndim > 0:
        value = value.mean()

    return value


def test_single_category(
    category: str,
    model: torch.nn.Module,
    params: argparse.Namespace,
    save: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Evaluate TSRNet on a single category.

    Args:
        category: Category name.
        model: Loaded TSRNet model.
        params: Command-line arguments.
        save: Whether to save visualizations and PLY files.

    Returns:
        A tuple containing:
            - average L1 Chamfer Distance
            - average L2 Chamfer Distance
            - average F-score
            - average DCD
    """
    if save:
        cat_dir = Path("test_res") / params.result_dir / category
        image_dir = cat_dir / "image"
        output_dir_gt = cat_dir / "output_gt"
        output_dir_res = cat_dir / "output_res"
        output_dir_fps = cat_dir / "output_fps"
        output_dir_coarse = cat_dir / "output_coarse"
        output_dir_input = cat_dir / "output_input"

        for directory in [
            cat_dir,
            image_dir,
            output_dir_gt,
            output_dir_res,
            output_dir_fps,
            output_dir_coarse,
            output_dir_input,
        ]:
            make_dir(directory)

    test_dataset = Coronary(
        dataroot=params.dataroot,
        split="test",
        category=category,
    )

    test_dataloader = Data.DataLoader(
        test_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=params.device.startswith("cuda"),
    )

    total_l1_cd = 0.0
    total_l2_cd = 0.0
    total_f_score = 0.0
    total_dcd = 0.0
    total_samples = 0

    index = 1

    with torch.no_grad():
        for partial, complete in test_dataloader:
            partial = partial.to(params.device, non_blocking=True)
            complete = complete.to(params.device, non_blocking=True)

            coarse, fine, output = model(partial)

            batch_size = partial.size(0)

            batch_l1_cd = to_scalar(l1_cd(output, complete))
            batch_l2_cd = to_scalar(l2_cd(output, complete))
            batch_dcd = to_scalar(calc_dcd(output, complete))

            total_l1_cd += batch_l1_cd.item() * batch_size
            total_l2_cd += batch_l2_cd.item() * batch_size
            total_dcd += batch_dcd.item() * batch_size
            total_samples += batch_size

            for i in range(batch_size):
                input_pc = partial[i].detach().cpu().numpy()
                output_pc = output[i].detach().cpu().numpy()
                gt_pc = complete[i].detach().cpu().numpy()

                total_f_score += f_score(output_pc, gt_pc)

                if save:
                    plot_pcd_one_view(
                        filename=image_dir / f"{index:03d}.png",
                        pcds=[input_pc, output_pc, gt_pc],
                        titles=["Input", "Output", "GT"],
                        xlim=(-0.35, 0.35),
                        ylim=(-0.35, 0.35),
                        zlim=(-0.35, 0.35),
                    )

                    export_ply(output_dir_res / f"{index:03d}.ply", output_pc)
                    export_ply(output_dir_gt / f"{index:03d}.ply", gt_pc)
                    export_ply(output_dir_input / f"{index:03d}.ply", input_pc)
                    export_ply(
                        output_dir_fps / f"{index:03d}.ply",
                        coarse[i].detach().cpu().numpy(),
                    )
                    export_ply(
                        output_dir_coarse / f"{index:03d}.ply",
                        fine[i].detach().cpu().numpy(),
                    )

                index += 1

    if total_samples == 0:
        raise RuntimeError(f"No test samples found for category: {category}")

    avg_l1_cd = total_l1_cd / total_samples
    avg_l2_cd = total_l2_cd / total_samples
    avg_f_score = total_f_score / total_samples
    avg_dcd = total_dcd / total_samples

    return avg_l1_cd, avg_l2_cd, avg_f_score, avg_dcd


def load_model(params: argparse.Namespace) -> torch.nn.Module:
    """
    Load a pretrained TSRNet model.

    Args:
        params: Command-line arguments.

    Returns:
        Loaded TSRNet model in evaluation mode.
    """
    model = TSRNet().to(params.device)

    checkpoint = torch.load(params.ckpt_path, map_location=params.device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    model.eval()

    return model


def test(params: argparse.Namespace, save: bool = False) -> None:
    """
    Evaluate TSRNet with CD, F-score, and DCD.

    Args:
        params: Command-line arguments.
        save: Whether to save point clouds and visualization images.
    """
    if save:
        make_dir(Path("test_res") / params.result_dir)

    print(f"Experiment: {params.exp_name}")
    print(f"Checkpoint: {params.ckpt_path}")

    model = load_model(params)

    header = "{:20s}{:20s}{:20s}{:20s}{:20s}".format(
        "Category",
        "L1_CD(1e-3)",
        "L2_CD(1e-4)",
        "F-score(%)",
        "DCD",
    )
    separator = "{:20s}{:20s}{:20s}{:20s}{:20s}".format(
        "--------",
        "-----------",
        "-----------",
        "----------",
        "---",
    )

    print(f"\033[33m{header}\033[0m")
    print(f"\033[33m{separator}\033[0m")

    if params.category == "all":
        categories = CATEGORIES

        l1_cds = []
        l2_cds = []
        fscores = []
        dcds = []

        for category in categories:
            avg_l1_cd, avg_l2_cd, avg_f_score, avg_dcd = test_single_category(
                category=category,
                model=model,
                params=params,
                save=save,
            )

            print(
                "{:20s}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}".format(
                    category.title(),
                    1e3 * avg_l1_cd,
                    1e4 * avg_l2_cd,
                    1e2 * avg_f_score,
                    avg_dcd,
                )
            )

            l1_cds.append(avg_l1_cd)
            l2_cds.append(avg_l2_cd)
            fscores.append(avg_f_score)
            dcds.append(avg_dcd)

        print(f"\033[33m{separator}\033[0m")
        print(
            "\033[32m{:20s}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}\033[0m".format(
                "Average",
                1e3 * float(np.mean(l1_cds)),
                1e4 * float(np.mean(l2_cds)),
                1e2 * float(np.mean(fscores)),
                float(np.mean(dcds)),
            )
        )

    else:
        avg_l1_cd, avg_l2_cd, avg_f_score, avg_dcd = test_single_category(
            category=params.category,
            model=model,
            params=params,
            save=save,
        )

        print(
            "{:20s}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}".format(
                params.category.title(),
                1e3 * avg_l1_cd,
                1e4 * avg_l2_cd,
                1e2 * avg_f_score,
                avg_dcd,
            )
        )


def test_single_category_emd(
    category: str,
    model: torch.nn.Module,
    params: argparse.Namespace,
) -> float:
    """
    Evaluate EMD on a single category.

    Args:
        category: Category name.
        model: Loaded TSRNet model.
        params: Command-line arguments.

    Returns:
        Average EMD.
    """
    test_dataset = Coronary(
        dataroot=params.dataroot,
        split="test",
        category=category,
    )

    test_dataloader = Data.DataLoader(
        test_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=params.device.startswith("cuda"),
    )

    total_emd = 0.0
    total_samples = 0

    with torch.no_grad():
        for partial, complete in test_dataloader:
            partial = partial.to(params.device, non_blocking=True)
            complete = complete.to(params.device, non_blocking=True)

            _, _, output = model(partial)

            batch_size = partial.size(0)
            batch_emd = to_scalar(metric_emd(output, complete))

            total_emd += batch_emd.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(f"No test samples found for category: {category}")

    return total_emd / total_samples


def test_emd(params: argparse.Namespace) -> None:
    """
    Evaluate TSRNet with EMD.

    Args:
        params: Command-line arguments.
    """
    print(f"Experiment: {params.exp_name}")
    print(f"Checkpoint: {params.ckpt_path}")

    model = load_model(params)

    print("\033[33m{:20s}{:20s}\033[0m".format("Category", "EMD(1e-3)"))
    print("\033[33m{:20s}{:20s}\033[0m".format("--------", "---------"))

    if params.category == "all":
        emds = []

        for category in CATEGORIES:
            avg_emd = test_single_category_emd(category, model, params)
            print("{:20s}{:<20.4f}".format(category.title(), 1e3 * avg_emd))
            emds.append(avg_emd)

        print("\033[33m{:20s}{:20s}\033[0m".format("--------", "---------"))
        print(
            "\033[32m{:20s}{:<20.4f}\033[0m".format(
                "Average",
                1e3 * float(np.mean(emds)),
            )
        )

    else:
        avg_emd = test_single_category_emd(params.category, model, params)
        print("{:20s}{:<20.4f}".format(params.category.title(), 1e3 * avg_emd))


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser("TSRNet Point Cloud Completion Testing")

    parser.add_argument(
        "--exp_name",
        type=str,
        default="TSRNet",
        help="Experiment name.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="TSRNet",
        help="Directory name under test_res for saving results.",
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        required=True,
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=f"log/TSRNet/all/checkpoints/{BEST_CHECKPOINT_NAME}",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        help="Category of point clouds. Use 'all' to evaluate all categories.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for the data loader.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for the data loader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device used for testing.",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Disable saving test results.",
    )
    parser.add_argument(
        "--emd",
        action="store_true",
        help="Evaluate EMD instead of CD/F-score/DCD.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point.
    """
    params = parse_args()

    if params.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please use --device cpu or enable CUDA.")

    save = not params.no_save

    if params.emd:
        test_emd(params)
    else:
        test(params, save=save)


if __name__ == "__main__":
    main()