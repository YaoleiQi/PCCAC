from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


PathLike = Union[str, Path]
AxisLimit = Tuple[float, float]


def o3d_visualize_pc(pc: np.ndarray) -> None:
    """
    Visualize a point cloud using Open3D.

    Args:
        pc: Point cloud array of shape (N, 3).
    """
    if pc.ndim != 2 or pc.shape[1] != 3:
        raise ValueError("Input point cloud must have shape (N, 3).")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)

    o3d.visualization.draw_geometries([point_cloud])


def plot_pcd_one_view(
    filename: PathLike,
    pcds: Sequence[np.ndarray],
    titles: Sequence[str],
    suptitle: str = "",
    sizes: Optional[Sequence[float]] = None,
    cmap: str = "Reds",
    zdir: str = "y",
    xlim: AxisLimit = (-0.5, 0.5),
    ylim: AxisLimit = (-0.5, 0.5),
    zlim: AxisLimit = (-0.5, 0.5),
    elev: float = 90.0,
    azim: float = -20.0,
    vmin: float = -1.0,
    vmax: float = 0.5,
) -> None:
    """
    Plot multiple point clouds from a fixed 3D view and save the figure.

    Args:
        filename: Output image file path.
        pcds: Sequence of point cloud arrays, each with shape (N, 3).
        titles: Titles for each subplot. The length should match `pcds`.
        suptitle: Super title of the whole figure.
        sizes: Marker sizes for each point cloud. If None, all sizes are set to 0.5.
        cmap: Matplotlib colormap used for point coloring.
        zdir: Axis direction used by Matplotlib's 3D scatter.
        xlim: X-axis limits.
        ylim: Y-axis limits.
        zlim: Z-axis limits.
        elev: Elevation angle of the 3D view.
        azim: Azimuth angle of the 3D view.
        vmin: Minimum value for color normalization.
        vmax: Maximum value for color normalization.

    Raises:
        ValueError: If `pcds` and `titles` have different lengths, or if any point
            cloud does not have shape (N, 3).
    """
    if len(pcds) != len(titles):
        raise ValueError("`pcds` and `titles` must have the same length.")

    if len(pcds) == 0:
        raise ValueError("At least one point cloud is required for plotting.")

    if sizes is None:
        sizes = [0.5 for _ in range(len(pcds))]

    if len(sizes) != len(pcds):
        raise ValueError("`sizes` and `pcds` must have the same length.")

    for pcd in pcds:
        if pcd.ndim != 2 or pcd.shape[1] != 3:
            raise ValueError("Each point cloud must have shape (N, 3).")

    fig = plt.figure(figsize=(len(pcds) * 3 * 1.4, 3 * 1.6))

    for idx, (pcd, size) in enumerate(zip(pcds, sizes)):
        # Use x-coordinate as the point color for better depth perception.
        color = pcd[:, 0]

        ax = fig.add_subplot(1, len(pcds), idx + 1, projection="3d")
        ax.view_init(elev=elev, azim=azim)

        ax.scatter(
            pcd[:, 0],
            pcd[:, 1],
            pcd[:, 2],
            zdir=zdir,
            c=color,
            s=size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(titles[idx])
        ax.set_axis_off()

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.05,
        top=0.9,
        wspace=0.1,
        hspace=0.1,
    )

    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)