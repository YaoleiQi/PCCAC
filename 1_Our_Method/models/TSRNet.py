from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from torch.nn import BatchNorm1d, Linear as Lin, ReLU, Sequential as Seq

from models.utils import PointNet_SA_Module_KNN, vTransformer


BEST_CHECKPOINT_NAME = "best_all_l1_cd.pth"


def fps(pc: torch.Tensor, num: int) -> torch.Tensor:
    """
    Perform furthest point sampling on a point cloud.

    Args:
        pc: Input point cloud of shape (B, N, 3).
        num: Number of points to sample.

    Returns:
        Sampled point cloud of shape (B, num, 3).
    """
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num)
    sub_pc = pointnet2_utils.gather_operation(
        pc.transpose(1, 2).contiguous(),
        fps_idx,
    ).transpose(1, 2).contiguous()

    return sub_pc


class FeatureExtractor(nn.Module):
    """
    Point cloud feature extractor based on PointNet++ set abstraction and
    vector attention transformer blocks.

    Args:
        out_dim: Output feature dimension.
        n_knn: Number of nearest neighbors used in transformer blocks.
    """

    def __init__(self, out_dim: int = 1024, n_knn: int = 20) -> None:
        super().__init__()

        self.sa_module_1 = PointNet_SA_Module_KNN(
            1024,
            16,
            3,
            [128, 256],
            group_all=False,
            if_bn=False,
            if_idx=True,
        )
        self.transformer_1 = vTransformer(256, dim=64, n_knn=n_knn)

        self.sa_module_2 = PointNet_SA_Module_KNN(
            256,
            16,
            256,
            [256, 512],
            group_all=False,
            if_bn=False,
            if_idx=True,
        )
        self.transformer_2 = vTransformer(512, dim=64, n_knn=n_knn)

        self.sa_module_3 = PointNet_SA_Module_KNN(
            None,
            None,
            512,
            [512, out_dim],
            group_all=True,
            if_bn=False,
        )

    def forward(self, partial_cloud: torch.Tensor) -> torch.Tensor:
        """
        Extract a global feature from a partial point cloud.

        Args:
            partial_cloud: Partial point cloud of shape (B, 3, N).

        Returns:
            Global feature tensor of shape (B, out_dim, 1).
        """
        l0_xyz = partial_cloud
        l0_points = partial_cloud

        # First set abstraction stage.
        l1_xyz, l1_points, _ = self.sa_module_1(l0_xyz, l0_points)
        l1_points = self.transformer_1(l1_points, l1_xyz)

        # Second set abstraction stage.
        l2_xyz, l2_points, _ = self.sa_module_2(l1_xyz, l1_points)
        l2_points = self.transformer_2(l2_points, l2_xyz)

        # Global feature aggregation.
        _, l3_points = self.sa_module_3(l2_xyz, l2_points)

        return l3_points


class SA(nn.Module):
    """
    Self-attention block for point-wise feature refinement.

    Args:
        d_model: Input feature dimension.
        d_model_out: Output feature dimension used by multi-head attention.
        nhead: Number of attention heads.
        dim_feedforward: Hidden dimension of the feed-forward network.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_model_out: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

        self.multihead = nn.MultiheadAttention(
            embed_dim=d_model_out,
            num_heads=nhead,
            dropout=dropout,
        )

        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = nn.GELU()

    @staticmethod
    def with_pos_embed(
        tensor: torch.Tensor,
        pos: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Add positional embedding to a tensor if provided.

        Args:
            tensor: Input tensor.
            pos: Optional positional embedding.

        Returns:
            Tensor with positional embedding added.
        """
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src1: torch.Tensor,
        src2: torch.Tensor,
        if_act: bool = False,
    ) -> torch.Tensor:
        """
        Apply attention from src1 to src2.

        Args:
            src1: Query feature tensor of shape (B, C, N).
            src2: Key/value feature tensor of shape (B, C, N).
            if_act: Reserved argument for compatibility.

        Returns:
            Refined feature tensor of shape (B, d_model_out, N).
        """
        del if_act

        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        batch_size, channels, _ = src1.shape

        # Convert from (B, C, N) to (N, B, C), which is required by
        # nn.MultiheadAttention when batch_first=False.
        src1 = src1.reshape(batch_size, channels, -1).permute(2, 0, 1)
        src2 = src2.reshape(batch_size, channels, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        attn_out = self.multihead(query=src1, key=src2, value=src2)[0]

        src1 = src1 + self.dropout12(attn_out)
        src1 = self.norm12(src1)

        ff_out = self.linear12(
            self.dropout1(
                self.activation1(
                    self.linear11(src1),
                )
            )
        )
        src1 = src1 + self.dropout13(ff_out)

        # Convert back to (B, C, N).
        src1 = src1.permute(1, 2, 0)

        return src1


def MLP(
    channels: Sequence[int],
    bn: bool = True,
    last: bool = False,
) -> nn.Sequential:
    """
    Build a multi-layer perceptron.

    Args:
        channels: Sequence of channel dimensions.
        bn: Whether to use BatchNorm1d.
        last: If True, the last layer has no activation.

    Returns:
        Sequential MLP module.
    """
    if len(channels) < 2:
        raise ValueError("`channels` must contain at least two dimensions.")

    layers: List[nn.Module] = []

    for i in range(1, len(channels) - 1):
        if bn:
            layers.append(
                Seq(
                    Lin(channels[i - 1], channels[i], bias=False),
                    BatchNorm1d(channels[i]),
                    ReLU(),
                )
            )
        else:
            layers.append(
                Seq(
                    Lin(channels[i - 1], channels[i], bias=True),
                    ReLU(),
                )
            )

    if last:
        layers.append(
            Seq(
                Lin(channels[-2], channels[-1], bias=True),
            )
        )
    else:
        if bn:
            layers.append(
                Seq(
                    Lin(channels[-2], channels[-1], bias=False),
                    BatchNorm1d(channels[-1]),
                    ReLU(),
                )
            )
        else:
            layers.append(
                Seq(
                    Lin(channels[-2], channels[-1], bias=True),
                    ReLU(),
                )
            )

    return Seq(*layers)


class HypersphericalModule(nn.Module):
    """
    Project global features to a hyperspherical latent space.

    Args:
        channels: MLP channel dimensions.
        is_normalized: Whether to output an Lp-normalized feature.
        norm_order: Norm order used for feature normalization.
        is_BN: Whether to apply BatchNorm1d before the MLP.
        eps: Small value used to avoid division by zero.
    """

    def __init__(
        self,
        channels: Sequence[int],
        is_normalized: bool,
        norm_order: int,
        is_BN: bool,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        self.is_BN = is_BN
        self.is_normalized = is_normalized
        self.norm_order = norm_order
        self.eps = eps

        if self.is_BN:
            self.bn = nn.BatchNorm1d(channels[0])

        self.mlp = MLP(channels, bn=False, last=True)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input feature tensor of shape (B, F).

        Returns:
            A list containing:
                - projected feature
                - normalized projected feature, if is_normalized=True
        """
        if self.is_BN:
            x = self.mlp(F.relu(self.bn(x)))
        else:
            x = self.mlp(x)

        outputs = [x]

        if self.is_normalized:
            norm = x.norm(dim=-1, keepdim=True, p=self.norm_order).clamp_min(self.eps)
            outputs.append(x / norm)

        return outputs


class RefineDense(nn.Module):
    """
    Dense point refinement module.

    This module upsamples and refines a coarse point cloud using global features
    and attention-based local feature refinement.

    Args:
        channel: Base feature channel dimension.
        ratio: Upsampling ratio.
    """

    def __init__(self, channel: int = 128, ratio: int = 1) -> None:
        super().__init__()

        self.ratio = ratio
        self.channel = channel

        self.relu = nn.GELU()

        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)
        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)

        self.conv_111 = nn.Conv1d(1024, 512, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)

        self.sa1 = SA(channel * 2, 512)
        self.sa2 = SA(512, 512)
        self.sa3 = SA(512, channel * ratio)

        self.conv_ps = nn.Conv1d(channel * ratio, channel * ratio, kernel_size=1)
        self.conv_delta = nn.Conv1d(channel * 2, channel, kernel_size=1)

        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)

    def forward(
        self,
        x: Optional[torch.Tensor],
        coarse: torch.Tensor,
        feat_g: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Refine and upsample a point cloud.

        Args:
            x: Reserved input for compatibility with previous implementations.
            coarse: Coarse point cloud of shape (B, 3, N).
            feat_g: Global feature tensor of shape (B, 1024, 1).

        Returns:
            A tuple containing:
                - refined point cloud of shape (B, 3, N * ratio)
                - refined feature tensor of shape (B, channel * ratio, N * ratio)
        """
        del x

        batch_size, _, num_points = coarse.size()

        # Extract local features from coarse coordinates.
        y = self.conv_x1(self.relu(self.conv_x(coarse)))

        # Project global feature to the same channel dimension.
        feat_g = self.conv_111(feat_g)
        feat_g = self.conv_11(self.relu(feat_g))
        feat_g = self.conv_1(self.relu(feat_g))

        # Concatenate local and global features.
        y0 = torch.cat(
            [y, feat_g.repeat(1, 1, y.shape[-1])],
            dim=1,
        )

        y1 = self.sa1(y0, y0)
        y2 = self.sa2(y1, y1)
        y3 = self.sa3(y2, y2)

        # Pixel-shuffle-like feature expansion.
        y3 = self.conv_ps(y3).reshape(batch_size, -1, num_points * self.ratio)

        # Repeat local features to match the upsampled resolution.
        y_up = y.repeat(1, 1, self.ratio)

        y_cat = torch.cat([y3, y_up], dim=1)
        y4 = self.conv_delta(y_cat)

        # Predict coordinate offsets and add them to repeated coarse points.
        refined = self.conv_out(self.relu(self.conv_out1(y4)))
        refined = refined + coarse.repeat(1, 1, self.ratio)

        return refined, y3


class TSRNet(nn.Module):
    """
    TSRNet model for point cloud completion.

    The network first extracts a global feature from the partial input point
    cloud, then progressively upsamples a sampled coarse point cloud through
    dense refinement modules.

    Input:
        xyz: Partial point cloud of shape (B, N, 3).

    Output:
        coarse: Coarse point cloud of shape (B, 1024, 3).
        fine: First-stage refined point cloud of shape (B, 2048, 3).
        fine1: Second-stage refined point cloud of shape (B, 4096, 3).
    """

    def __init__(self) -> None:
        super().__init__()

        step1 = 2
        step2 = 2

        self.feat_extractor = FeatureExtractor()

        self.hyperspherical_module = HypersphericalModule(
            channels=[1024, 1024],
            is_normalized=True,
            norm_order=2,
            is_BN=False,
        )

        self.refine = RefineDense(ratio=step1)
        self.refine1 = RefineDense(ratio=step2)

    def forward(
        self,
        xyz: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of TSRNet.

        Args:
            xyz: Input partial point cloud of shape (B, N, 3).

        Returns:
            A tuple containing:
                - coarse point cloud of shape (B, 1024, 3)
                - first refined point cloud of shape (B, 2048, 3)
                - second refined point cloud of shape (B, 4096, 3)
        """
        batch_size = xyz.size(0)

        # Encoder.
        partial_cloud = xyz.permute(0, 2, 1).contiguous()
        feat_g = self.feat_extractor(partial_cloud)

        hyper = self.hyperspherical_module(feat_g.view(batch_size, 1024))
        feat_g = hyper[1].unsqueeze(2)

        # Decoder.
        coarse = fps(xyz, 1024)
        new_x = coarse.transpose(1, 2).contiguous()

        fine, feat_fine = self.refine(None, new_x, feat_g)
        fine1, _ = self.refine1(feat_fine, fine, feat_g)

        return (
            coarse.contiguous(),
            fine.transpose(1, 2).contiguous(),
            fine1.transpose(1, 2).contiguous(),
        )


# Backward-compatible aliases.
# These aliases allow old training or evaluation scripts to keep working.
CACNet = TSRNet
refine_dense = RefineDense