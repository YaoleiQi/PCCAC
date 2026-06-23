import torch

from extensions.chamfer_distance.chamfer_distance import ChamferDistance
from extensions.earth_movers_distance.emd import EarthMoverDistance


# Initialize distance modules once and reuse them to avoid repeated construction.
CD = ChamferDistance()
EMD = EarthMoverDistance()


def cd_loss_L1(
    pcs1: torch.Tensor,
    pcs2: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute the L1 Chamfer Distance between two point clouds.

    The Chamfer Distance is computed bidirectionally. The underlying Chamfer
    distance implementation returns squared nearest-neighbor distances. This
    function applies square root to obtain an L1-style distance.

    Args:
        pcs1: First point cloud tensor of shape (B, N, 3).
        pcs2: Second point cloud tensor of shape (B, M, 3).
        eps: Minimum value used to clamp distances before applying square root.

    Returns:
        Scalar tensor representing the mean L1 Chamfer Distance.
    """
    dist1, dist2 = CD(pcs1, pcs2)

    # Clamp distances for numerical stability before square root.
    dist1 = torch.sqrt(torch.clamp(dist1, min=eps))
    dist2 = torch.sqrt(torch.clamp(dist2, min=eps))

    return (torch.mean(dist1) + torch.mean(dist2)) / 2.0


def cd_loss_L2(
    pcs1: torch.Tensor,
    pcs2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the L2 Chamfer Distance between two point clouds.

    The Chamfer Distance is computed bidirectionally using squared
    nearest-neighbor distances.

    Args:
        pcs1: First point cloud tensor of shape (B, N, 3).
        pcs2: Second point cloud tensor of shape (B, M, 3).

    Returns:
        Scalar tensor representing the mean L2 Chamfer Distance.
    """
    dist1, dist2 = CD(pcs1, pcs2)

    return torch.mean(dist1) + torch.mean(dist2)


def emd_loss(
    pcs1: torch.Tensor,
    pcs2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Earth Mover's Distance loss between two point clouds.

    Note:
        Most EMD implementations require the two point clouds to have the same
        number of points, i.e., pcs1.shape[1] == pcs2.shape[1].

    Args:
        pcs1: First point cloud tensor of shape (B, N, 3).
        pcs2: Second point cloud tensor of shape (B, N, 3).

    Returns:
        Scalar tensor representing the mean EMD loss.
    """
    dists = EMD(pcs1, pcs2)

    return torch.mean(dists)