import argparse
import datetime
import random
from pathlib import Path
from typing import TextIO, Tuple

import numpy as np
import torch
import torch.optim as Optim
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from dataset import Coronary
from metrics.loss import cd_loss_L1
from metrics.metric import l1_cd
from models import TSRNet
from tools.noise_reuction import del_zdm
from visualization import plot_pcd_one_view


BEST_CHECKPOINT_NAME = "best_all_l1_cd.pth"


def make_dir(dir_path: Path) -> None:
    """
    Create a directory if it does not exist.

    Args:
        dir_path: Directory path to create.
    """
    dir_path.mkdir(parents=True, exist_ok=True)


def np2tensor(points: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert a NumPy point cloud to a batched torch tensor.

    Args:
        points: Point cloud array of shape (N, 3).
        device: Target device.

    Returns:
        Point cloud tensor of shape (1, N, 3).
    """
    return torch.from_numpy(points).to(device).unsqueeze(0).float()


def log(log_fd: TextIO, message: str, with_time: bool = True) -> None:
    """
    Write a message to both log file and console.

    Args:
        log_fd: Open log file descriptor.
        message: Message to log.
        with_time: Whether to prepend the current timestamp.
    """
    if with_time:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = " ==> ".join([timestamp, message])

    log_fd.write(message + "\n")
    log_fd.flush()
    print(message)


def prepare_logger(
    params: argparse.Namespace,
) -> Tuple[Path, Path, TextIO, SummaryWriter, SummaryWriter]:
    """
    Prepare logging directories, log file, and TensorBoard writers.

    Args:
        params: Training arguments.

    Returns:
        A tuple containing:
            - checkpoint directory
            - epoch visualization directory
            - log file descriptor
            - training TensorBoard writer
            - validation TensorBoard writer
    """
    log_root = Path(params.log_dir)
    logger_path = log_root / params.exp_name / params.category
    ckpt_dir = logger_path / "checkpoints"
    epochs_dir = logger_path / "epochs"

    make_dir(logger_path)
    make_dir(ckpt_dir)
    make_dir(epochs_dir)

    logger_file = logger_path / "logger.log"
    log_fd = open(logger_file, "a", encoding="utf-8")

    log(log_fd, f"Experiment: {params.exp_name}", with_time=False)
    log(log_fd, f"Logger directory: {logger_path}", with_time=False)
    log(log_fd, str(params), with_time=False)

    train_writer = SummaryWriter(str(logger_path / "train"))
    val_writer = SummaryWriter(str(logger_path / "val"))

    return ckpt_dir, epochs_dir, log_fd, train_writer, val_writer


def load_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
    log_fd: TextIO,
) -> None:
    """
    Load a pretrained checkpoint into the model.

    Args:
        model: Model instance.
        ckpt_path: Path to checkpoint.
        device: Target device.
        log_fd: Log file descriptor.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    log(log_fd, f"Loaded checkpoint from: {ckpt_path}")


def filter_zg_points(points: np.ndarray) -> np.ndarray:
    """
    Apply vessel/noise filtering to a point cloud.

    If filtering returns an empty point cloud, the original point cloud is used
    as a fallback to avoid invalid Chamfer Distance computation.

    Args:
        points: Input point cloud of shape (N, 3).

    Returns:
        Filtered point cloud of shape (M, 3).
    """
    filtered = del_zdm(points)

    if filtered.shape[0] == 0:
        return points

    return filtered


def compute_zg_loss(
    coarse_pred: torch.Tensor,
    dense_pred: torch.Tensor,
    complete: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the filtered vessel-region Chamfer loss.

    Note:
        This loss is computed after converting predictions to NumPy arrays and
        applying Open3D/NumPy-based filtering. Therefore, this term is not fully
        differentiable with respect to the model outputs.

    Args:
        coarse_pred: Coarse prediction tensor of shape (B, N, 3).
        dense_pred: Dense prediction tensor of shape (B, N, 3).
        complete: Ground-truth point cloud tensor of shape (B, M, 3).
        device: Target device.

    Returns:
        Averaged filtered-region L1 Chamfer loss.
    """
    batch_size = complete.size(0)
    loss = torch.zeros((), device=device)

    for idx in range(batch_size):
        pred_zg_coarse = filter_zg_points(
            coarse_pred[idx].detach().cpu().numpy()
        )
        pred_zg_dense = filter_zg_points(
            dense_pred[idx].detach().cpu().numpy()
        )
        gt_zg = filter_zg_points(
            complete[idx].detach().cpu().numpy()
        )

        loss_coarse = cd_loss_L1(
            np2tensor(pred_zg_coarse, device),
            np2tensor(gt_zg, device),
        )
        loss_dense = cd_loss_L1(
            np2tensor(pred_zg_dense, device),
            np2tensor(gt_zg, device),
        )

        loss = loss + 0.5 * (loss_coarse + loss_dense)

    return loss / batch_size


def train(params: argparse.Namespace) -> None:
    """
    Train TSRNet.

    Args:
        params: Training arguments.
    """
    device = torch.device(params.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please use --device cpu or enable CUDA.")

    torch.backends.cudnn.benchmark = True

    ckpt_dir, epochs_dir, log_fd, train_writer, val_writer = prepare_logger(params)

    try:
        log(log_fd, "Loading data...")

        train_dataset = Coronary(
            dataroot=params.dataroot,
            split="train",
            category=params.category,
        )
        val_dataset = Coronary(
            dataroot=params.dataroot,
            split="val",
            category=params.category,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=params.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=params.batch_size,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

        log(log_fd, f"Training samples: {len(train_dataset)}")
        log(log_fd, f"Validation samples: {len(val_dataset)}")
        log(log_fd, "Dataset loaded!")

        # Build TSRNet model.
        model = TSRNet().to(device)

        optimizer = Optim.Adam(
            model.parameters(),
            lr=params.lr,
            betas=(0.9, 0.999),
        )
        lr_scheduler = Optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.lr_step_size,
            gamma=params.lr_gamma,
        )

        if params.ckpt_path is not None:
            load_checkpoint(model, params.ckpt_path, device, log_fd)

        best_all_l1_cd = float("inf")
        best_zg_l1_cd = float("inf")
        best_loss_all = float("inf")
        best_loss_zg = float("inf")

        best_epoch_all_l1 = -1
        best_epoch_zg_l1 = -1
        best_loss_all_epoch = -1
        best_loss_zg_epoch = -1

        train_step = 0
        val_step = 0

        for epoch in range(1, params.epochs + 1):
            model.train()

            for iteration, (partial, complete) in enumerate(train_dataloader):
                partial = partial.to(device, non_blocking=True)
                complete = complete.to(device, non_blocking=True)

                optimizer.zero_grad()

                # Forward pass.
                fps_points, coarse_pred, dense_pred = model(partial)

                # Full-shape reconstruction loss.
                loss_all_coarse = cd_loss_L1(coarse_pred, complete)
                loss_all_dense = cd_loss_L1(dense_pred, complete)
                loss_all = 0.5 * (loss_all_coarse + loss_all_dense)

                # Filtered vessel-region loss.
                loss_zg = compute_zg_loss(
                    coarse_pred=coarse_pred,
                    dense_pred=dense_pred,
                    complete=complete,
                    device=device,
                )

                # Use only full-shape loss in early epochs, then combine both losses.
                if epoch <= params.warmup_epochs:
                    loss = loss_all
                else:
                    loss = 0.5 * (loss_all + loss_zg)

                loss_avg = 0.5 * (loss_all + loss_zg)

                loss_avg_value = loss_avg.item()
                loss_zg_value = loss_zg.item()

                if loss_avg_value < best_loss_all:
                    best_loss_all = loss_avg_value
                    best_loss_all_epoch = epoch
                    torch.save(model.state_dict(), ckpt_dir / "loss_all.pth")

                if loss_zg_value < best_loss_zg:
                    best_loss_zg = loss_zg_value
                    best_loss_zg_epoch = epoch
                    torch.save(model.state_dict(), ckpt_dir / "loss_zg.pth")

                # Backward pass.
                loss.backward()
                optimizer.step()

                if (iteration + 1) % params.log_frequency == 0:
                    log(
                        log_fd,
                        (
                            "Training Epoch [{:03d}/{:03d}] - "
                            "Iteration [{:03d}/{:03d}]: "
                            "all loss = {:.6f}, zg loss = {:.6f}, total loss = {:.6f}"
                        ).format(
                            epoch,
                            params.epochs,
                            iteration + 1,
                            len(train_dataloader),
                            loss_all.item() * 1e3,
                            loss_zg.item() * 1e3,
                            loss_avg.item() * 1e3,
                        ),
                    )

                train_writer.add_scalar("all", loss_all.item(), train_step)
                train_writer.add_scalar("zg", loss_zg.item(), train_step)
                train_writer.add_scalar("total", loss_avg.item(), train_step)
                train_step += 1

            lr_scheduler.step()

            # Save periodic checkpoints.
            if epoch % params.save_frequency == 0:
                torch.save(model.state_dict(), ckpt_dir / f"{epoch}.pth")

            # Validation.
            model.eval()

            total_all_l1_cd = 0.0
            total_zg_l1_cd = 0.0
            total_samples = 0

            rand_iter = random.randint(0, max(len(val_dataloader) - 1, 0))

            with torch.no_grad():
                for iteration, (partial, complete) in enumerate(val_dataloader):
                    partial = partial.to(device, non_blocking=True)
                    complete = complete.to(device, non_blocking=True)

                    fps_points, coarse_pred, dense_pred = model(partial)

                    batch_size = partial.size(0)

                    batch_all_l1_cd = l1_cd(dense_pred, complete).item()
                    total_all_l1_cd += batch_all_l1_cd * batch_size

                    batch_zg_l1_cd = torch.zeros((), device=device)

                    for idx in range(batch_size):
                        pred_zg = filter_zg_points(
                            dense_pred[idx].detach().cpu().numpy()
                        )
                        gt_zg = filter_zg_points(
                            complete[idx].detach().cpu().numpy()
                        )

                        batch_zg_l1_cd = batch_zg_l1_cd + cd_loss_L1(
                            np2tensor(pred_zg, device),
                            np2tensor(gt_zg, device),
                        )

                    batch_zg_l1_cd = batch_zg_l1_cd / batch_size
                    total_zg_l1_cd += batch_zg_l1_cd.item() * batch_size
                    total_samples += batch_size

                    # Save one visualization per epoch.
                    if iteration == rand_iter:
                        vis_index = random.randint(0, batch_size - 1)

                        plot_pcd_one_view(
                            filename=epochs_dir / f"epoch_{epoch:03d}.png",
                            pcds=[
                                partial[vis_index].detach().cpu().numpy(),
                                fps_points[vis_index].detach().cpu().numpy(),
                                coarse_pred[vis_index].detach().cpu().numpy(),
                                dense_pred[vis_index].detach().cpu().numpy(),
                                complete[vis_index].detach().cpu().numpy(),
                            ],
                            titles=[
                                "Input",
                                "FPS",
                                "Coarse",
                                "Dense",
                                "Ground Truth",
                            ],
                            xlim=(-0.35, 0.35),
                            ylim=(-0.35, 0.35),
                            zlim=(-0.35, 0.35),
                        )

            if total_samples == 0:
                raise RuntimeError("No validation samples found.")

            avg_all_l1_cd = total_all_l1_cd / total_samples
            avg_zg_l1_cd = total_zg_l1_cd / total_samples

            val_writer.add_scalar("all_l1_cd", avg_all_l1_cd, val_step)
            val_writer.add_scalar("zg_l1_cd", avg_zg_l1_cd, val_step)
            val_step += 1

            log(
                log_fd,
                (
                    "Validate Epoch [{:03d}/{:03d}]: "
                    "all L1 Chamfer Distance = {:.6f}, "
                    "zg L1 Chamfer Distance = {:.6f}"
                ).format(
                    epoch,
                    params.epochs,
                    avg_all_l1_cd * 1e3,
                    avg_zg_l1_cd * 1e3,
                ),
            )

            if avg_all_l1_cd < best_all_l1_cd:
                best_all_l1_cd = avg_all_l1_cd
                best_epoch_all_l1 = epoch
                torch.save(model.state_dict(), ckpt_dir / BEST_CHECKPOINT_NAME)

            if avg_zg_l1_cd < best_zg_l1_cd:
                best_zg_l1_cd = avg_zg_l1_cd
                best_epoch_zg_l1 = epoch
                torch.save(model.state_dict(), ckpt_dir / "best_zg_l1_cd.pth")

        log(
            log_fd,
            (
                "Best all l1 cd model in epoch {}, "
                "the minimum all l1 cd is {:.6f}"
            ).format(best_epoch_all_l1, best_all_l1_cd * 1e3),
        )
        log(
            log_fd,
            (
                "Best zg l1 cd model in epoch {}, "
                "the minimum zg l1 cd is {:.6f}"
            ).format(best_epoch_zg_l1, best_zg_l1_cd * 1e3),
        )
        log(
            log_fd,
            (
                "Best train loss_all model in epoch {}, "
                "the loss_all is {:.6f}"
            ).format(best_loss_all_epoch, best_loss_all * 1e3),
        )
        log(
            log_fd,
            (
                "Best train loss_zg model in epoch {}, "
                "the loss_zg is {:.6f}"
            ).format(best_loss_zg_epoch, best_loss_zg * 1e3),
        )

    finally:
        train_writer.close()
        val_writer.close()
        log_fd.close()


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed training arguments.
    """
    parser = argparse.ArgumentParser("TSRNet Point Cloud Completion Training")

    parser.add_argument(
        "--exp_name",
        type=str,
        default="TSRNet",
        help="Experiment name.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="log",
        help="Logger directory.",
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
        default=None,
        help="Path to pretrained checkpoint.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--lr_step_size",
        type=int,
        default=50,
        help="Step size for learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.7,
        help="Gamma for learning rate scheduler.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        help="Category of point clouds.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=200,
        help="Number of epochs using only full-shape reconstruction loss.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for data loader.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for data loader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for training.",
    )
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=20,
        help="Logging frequency in each epoch.",
    )
    parser.add_argument(
        "--save_frequency",
        type=int,
        default=100,
        help="Checkpoint saving frequency.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point.
    """
    params = parse_args()
    train(params)


if __name__ == "__main__":
    main()