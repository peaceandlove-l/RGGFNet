import math
import os
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
IntrinsicsType = Optional[Union[Tensor, Sequence[float]]]
FieldValue = Union[Tensor, Tuple[int, int], int]
FieldLevel = Dict[str, FieldValue]


class LayerNorm2d(nn.Module):
    """对 BCHW 特征的通道维执行 LayerNorm。"""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"LayerNorm2d 需要 BCHW 输入，实际得到 {tuple(x.shape)}")
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()


def make_group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """生成通道数可整除的 GroupNorm，避免小 batch 下 BatchNorm 不稳定。"""
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvNormAct(nn.Sequential):
    """卷积、归一化和 GELU。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )


class FourierCoordinateEncoder(nn.Module):
    """
    将连续坐标 (x, y, z) 编码为 Fourier spatial code。

    输入：
        coordinates: [B, 3, H, W]
        valid_mask:  [B, 1, H, W]
    输出：
        coordinate embedding: [B, field_dim, H, W]
    """

    def __init__(self, field_dim: int, num_bands: int = 6) -> None:
        super().__init__()
        if num_bands < 1:
            raise ValueError("num_bands 必须大于等于 1")

        frequencies = math.pi * (2.0 ** torch.arange(num_bands, dtype=torch.float32))
        self.register_buffer("frequencies", frequencies, persistent=False)

        encoded_channels = 3 + 3 * 2 * num_bands + 1
        self.projection = nn.Sequential(
            ConvNormAct(encoded_channels, field_dim),
            nn.Conv2d(field_dim, field_dim, kernel_size=1, bias=False),
            LayerNorm2d(field_dim),
            nn.GELU(),
        )

    def forward(self, coordinates: Tensor, valid_mask: Tensor) -> Tensor:
        if coordinates.ndim != 4 or coordinates.shape[1] != 3:
            raise ValueError(
                "coordinates 应为 [B, 3, H, W]，"
                f"实际得到 {tuple(coordinates.shape)}"
            )
        if valid_mask.ndim != 4 or valid_mask.shape[1] != 1:
            raise ValueError(
                "valid_mask 应为 [B, 1, H, W]，"
                f"实际得到 {tuple(valid_mask.shape)}"
            )
        if coordinates.shape[0] != valid_mask.shape[0] or coordinates.shape[-2:] != valid_mask.shape[-2:]:
            raise ValueError("coordinates 与 valid_mask 的 batch/空间尺寸必须一致")

        frequencies = self.frequencies.to(
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        phase = coordinates.unsqueeze(2) * frequencies.view(1, 1, -1, 1, 1)
        sin_code = torch.sin(phase).flatten(1, 2)
        cos_code = torch.cos(phase).flatten(1, 2)

        encoded = torch.cat(
            [coordinates, sin_code, cos_code, valid_mask.to(coordinates.dtype)],
            dim=1,
        )
        return self.projection(encoded)


class NeuralGeometryFieldLevel(nn.Module):
    """
    单个 DINOv3 层级对应的 NGFC。

    核心场论建模：
      1. RGB 语义、Depth 几何与连续坐标共同构造 neural field；
      2. 几何场通过仿射调制影响语义场；
      3. 语义场反向门控几何场；
      4. 每个空间位置预测场描述子及位置、尺度、旋转、置信度等场属性。
    """

    def __init__(
        self,
        in_channels: int = 768,
        field_dim: int = 256,
        semantic_dim: int = 128,
        coordinate_bands: int = 6,
        initial_scale: float = 0.05,
        minimum_scale: float = 1e-4,
        position_delta_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if initial_scale <= minimum_scale:
            raise ValueError("initial_scale 必须大于 minimum_scale")
        if field_dim < 8 or semantic_dim < 4:
            raise ValueError("field_dim 和 semantic_dim 设置过小")

        self.field_dim = field_dim
        self.semantic_dim = semantic_dim
        self.minimum_scale = minimum_scale
        self.position_delta_scale = position_delta_scale

        self.rgb_projection = ConvNormAct(in_channels, field_dim)
        self.depth_projection = ConvNormAct(in_channels, field_dim)
        self.coordinate_encoder = FourierCoordinateEncoder(
            field_dim=field_dim,
            num_bands=coordinate_bands,
        )

        self.geometry_to_semantic = nn.Conv2d(
            field_dim, 2 * field_dim, kernel_size=1, bias=True
        )
        self.semantic_to_geometry = nn.Conv2d(
            field_dim, field_dim, kernel_size=1, bias=True
        )

        self.field_constructor = nn.Sequential(
            nn.Conv2d(3 * field_dim, 2 * field_dim, kernel_size=1, bias=False),
            LayerNorm2d(2 * field_dim),
            nn.GELU(),
            nn.Conv2d(2 * field_dim, field_dim, kernel_size=1, bias=False),
            LayerNorm2d(field_dim),
            nn.GELU(),
        )

        self.local_field_refinement = nn.Sequential(
            nn.Conv2d(
                field_dim,
                field_dim,
                kernel_size=3,
                padding=1,
                groups=field_dim,
                bias=False,
            ),
            LayerNorm2d(field_dim),
            nn.GELU(),
            nn.Conv2d(field_dim, field_dim, kernel_size=1, bias=False),
        )
        self.field_norm = LayerNorm2d(field_dim)

        self.position_head = self._make_head(field_dim, 3)
        self.scale_head = self._make_head(field_dim, 3)
        self.rotation_head = self._make_head(field_dim, 4)
        self.field_descriptor_head = self._make_head(field_dim, field_dim)
        self.semantic_descriptor_head = self._make_head(field_dim, semantic_dim)
        self.confidence_head = self._make_head(field_dim, 1)
        self.semantic_uncertainty_head = self._make_head(field_dim, 1)

        complexity_hidden = max(field_dim // 2, 8)
        self.complexity_head = nn.Sequential(
            nn.Conv2d(field_dim + 2, complexity_hidden, kernel_size=1, bias=False),
            LayerNorm2d(complexity_hidden),
            nn.GELU(),
            nn.Conv2d(complexity_hidden, 1, kernel_size=1, bias=True),
        )

        self._initialize_geometry_heads(initial_scale)

    @staticmethod
    def _make_head(in_channels: int, out_channels: int) -> nn.Sequential:
        hidden_channels = max(in_channels // 2, out_channels)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            LayerNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True),
        )

    def _initialize_geometry_heads(self, initial_scale: float) -> None:
        nn.init.zeros_(self.position_head[-1].weight)
        nn.init.zeros_(self.position_head[-1].bias)

        nn.init.zeros_(self.rotation_head[-1].weight)
        nn.init.zeros_(self.rotation_head[-1].bias)

        scale_target = initial_scale - self.minimum_scale
        scale_bias = math.log(math.expm1(scale_target))
        nn.init.zeros_(self.scale_head[-1].weight)
        nn.init.constant_(self.scale_head[-1].bias, scale_bias)

        nn.init.zeros_(self.confidence_head[-1].bias)
        nn.init.zeros_(self.semantic_uncertainty_head[-1].bias)
        nn.init.zeros_(self.complexity_head[-1].bias)

    @staticmethod
    def _flatten_field(x: Tensor) -> Tensor:
        return x.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _local_depth_variation(normalized_depth: Tensor, valid_mask: Tensor) -> Tensor:
        dx = torch.abs(normalized_depth[:, :, :, 1:] - normalized_depth[:, :, :, :-1])
        dy = torch.abs(normalized_depth[:, :, 1:, :] - normalized_depth[:, :, :-1, :])

        dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
        dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")
        variation = torch.sqrt(dx.square() + dy.square() + 1e-8)
        variation = torch.tanh(variation)
        return variation * valid_mask.to(variation.dtype)

    def forward(
        self,
        rgb_feature: Tensor,
        depth_feature: Tensor,
        anchor_position: Tensor,
        normalized_coordinates: Tensor,
        depth_valid_mask: Tensor,
    ) -> FieldLevel:
        if rgb_feature.shape != depth_feature.shape:
            raise ValueError(
                "对应层 RGB 与 Depth 特征形状必须一致，"
                f"实际为 {tuple(rgb_feature.shape)} 与 {tuple(depth_feature.shape)}"
            )
        if rgb_feature.ndim != 4:
            raise ValueError("DINOv3 中间特征必须是 BCHW 张量")

        batch_size, _, height, width = rgb_feature.shape
        expected_geometry_shape = (batch_size, 3, height, width)
        expected_mask_shape = (batch_size, 1, height, width)
        if tuple(anchor_position.shape) != expected_geometry_shape:
            raise ValueError(
                f"anchor_position 应为 {expected_geometry_shape}，"
                f"实际为 {tuple(anchor_position.shape)}"
            )
        if tuple(normalized_coordinates.shape) != expected_geometry_shape:
            raise ValueError(
                f"normalized_coordinates 应为 {expected_geometry_shape}，"
                f"实际为 {tuple(normalized_coordinates.shape)}"
            )
        if tuple(depth_valid_mask.shape) != expected_mask_shape:
            raise ValueError(
                f"depth_valid_mask 应为 {expected_mask_shape}，"
                f"实际为 {tuple(depth_valid_mask.shape)}"
            )

        rgb_context = self.rgb_projection(rgb_feature)
        depth_context = self.depth_projection(depth_feature)
        coordinate_context = self.coordinate_encoder(
            normalized_coordinates, depth_valid_mask
        )

        geometry_context = depth_context + coordinate_context

        gamma, beta = self.geometry_to_semantic(geometry_context).chunk(2, dim=1)
        semantic_context = rgb_context * (1.0 + torch.tanh(gamma)) + beta

        geometry_gate = torch.sigmoid(self.semantic_to_geometry(semantic_context))
        geometry_context = geometry_context * (1.0 + geometry_gate)

        field_latent = self.field_constructor(
            torch.cat(
                [semantic_context, geometry_context, coordinate_context],
                dim=1,
            )
        )
        field_latent = self.field_norm(
            field_latent
            + semantic_context
            + geometry_context
            + self.local_field_refinement(field_latent)
        )

        position_delta = torch.tanh(self.position_head(field_latent))
        position = anchor_position + self.position_delta_scale * position_delta

        scale = F.softplus(self.scale_head(field_latent)) + self.minimum_scale

        quaternion = self.rotation_head(field_latent)
        identity = torch.zeros_like(quaternion)
        identity[:, 0:1] = 1.0
        rotation = F.normalize(quaternion + identity, dim=1, eps=1e-6)

        field_descriptor = self.field_descriptor_head(field_latent)
        semantic_descriptor = self.semantic_descriptor_head(
            field_latent + semantic_context
        )
        confidence = torch.sigmoid(self.confidence_head(field_latent))
        semantic_uncertainty = torch.sigmoid(
            self.semantic_uncertainty_head(field_latent)
        )

        normalized_depth = normalized_coordinates[:, 2:3]
        depth_variation = self._local_depth_variation(
            normalized_depth, depth_valid_mask
        )
        complexity = torch.sigmoid(
            self.complexity_head(
                torch.cat(
                    [field_latent, depth_variation, semantic_uncertainty],
                    dim=1,
                )
            )
        )

        return {
            "position": self._flatten_field(position),
            "scale": self._flatten_field(scale),
            "rotation": self._flatten_field(rotation),
            "field": self._flatten_field(field_descriptor),
            "semantic": self._flatten_field(semantic_descriptor),
            "confidence": self._flatten_field(confidence),
            "complexity": self._flatten_field(complexity),
            "semantic_uncertainty": self._flatten_field(semantic_uncertainty),
            "depth_variation": self._flatten_field(depth_variation),
            "depth_valid": self._flatten_field(
                depth_valid_mask.to(field_latent.dtype)
            ),
            "spatial_shape": (height, width),
        }


class NeuralGeometryFieldConstruction(nn.Module):
    """将多层 RGB/Depth 特征转换为 Neural Geometry Field pyramid。"""

    def __init__(
        self,
        in_channels: int = 768,
        field_dim: int = 256,
        semantic_dim: int = 128,
        num_levels: int = 4,
        coordinate_bands: int = 6,
        initial_scales: Optional[Sequence[float]] = None,
        minimum_scale: float = 1e-4,
        position_delta_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if num_levels < 1:
            raise ValueError("num_levels 必须大于等于 1")
        if initial_scales is None:
            initial_scales = tuple(0.04 + 0.02 * i for i in range(num_levels))
        if len(initial_scales) != num_levels:
            raise ValueError(
                f"initial_scales 数量必须等于 num_levels={num_levels}，"
                f"实际得到 {len(initial_scales)}"
            )

        self.num_levels = num_levels
        self.levels = nn.ModuleList(
            [
                NeuralGeometryFieldLevel(
                    in_channels=in_channels,
                    field_dim=field_dim,
                    semantic_dim=semantic_dim,
                    coordinate_bands=coordinate_bands,
                    initial_scale=float(initial_scales[level_index]),
                    minimum_scale=minimum_scale,
                    position_delta_scale=position_delta_scale,
                )
                for level_index in range(num_levels)
            ]
        )

    @staticmethod
    def _extract_scalar_depth(depth: Tensor) -> Tensor:
        if depth.ndim != 4:
            raise ValueError(f"depth 应为 BCHW，实际得到 {tuple(depth.shape)}")
        if depth.shape[1] == 1:
            return depth
        if depth.shape[1] == 3:
            return depth[:, 0:1]
        raise ValueError(
            "depth 仅支持 1 通道原始深度，或由深度复制得到的 3 通道输入，"
            f"实际通道数为 {depth.shape[1]}"
        )

    @staticmethod
    def _normalize_depth(depth: Tensor, valid_mask: Tensor) -> Tensor:
        valid_float = valid_mask.to(depth.dtype)
        count = valid_float.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        mean = (depth * valid_float).sum(dim=(2, 3), keepdim=True) / count
        variance = (
            (depth - mean).square() * valid_float
        ).sum(dim=(2, 3), keepdim=True) / count
        std = torch.sqrt(variance + 1e-6)
        normalized = (depth - mean) / std
        normalized = torch.clamp(normalized, min=-5.0, max=5.0)
        return torch.where(valid_mask, normalized, torch.zeros_like(normalized))

    @staticmethod
    def _parse_intrinsics(
        intrinsics: IntrinsicsType,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        if intrinsics is None:
            return None

        intrinsics_tensor = torch.as_tensor(intrinsics, device=device, dtype=dtype)
        if intrinsics_tensor.ndim == 1:
            if intrinsics_tensor.numel() != 4:
                raise ValueError("一维 intrinsics 必须为 [fx, fy, cx, cy]")
            intrinsics_tensor = intrinsics_tensor.unsqueeze(0).expand(batch_size, -1)
        elif intrinsics_tensor.ndim == 2:
            if intrinsics_tensor.shape != (batch_size, 4):
                raise ValueError(
                    "二维 intrinsics 必须为 [B, 4]，顺序为 [fx, fy, cx, cy]"
                )
        else:
            raise ValueError("intrinsics 仅支持 [4] 或 [B, 4]")

        return intrinsics_tensor

    def _build_geometry_inputs(
        self,
        raw_depth: Tensor,
        feature_size: Tuple[int, int],
        intrinsics: IntrinsicsType,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        scalar_depth = self._extract_scalar_depth(raw_depth)
        batch_size, _, original_height, original_width = scalar_depth.shape
        feature_height, feature_width = feature_size

        sampled_depth = F.interpolate(
            scalar_depth,
            size=feature_size,
            mode="bilinear",
            align_corners=False,
        )
        valid_mask = torch.isfinite(sampled_depth) & (sampled_depth > 0)
        clean_depth = torch.where(
            valid_mask, sampled_depth, torch.zeros_like(sampled_depth)
        )
        normalized_depth = self._normalize_depth(clean_depth, valid_mask)

        y_coordinates = torch.linspace(
            -1.0,
            1.0,
            feature_height,
            device=raw_depth.device,
            dtype=raw_depth.dtype,
        )
        x_coordinates = torch.linspace(
            -1.0,
            1.0,
            feature_width,
            device=raw_depth.device,
            dtype=raw_depth.dtype,
        )
        grid_y, grid_x = torch.meshgrid(
            y_coordinates, x_coordinates, indexing="ij"
        )
        grid_x = grid_x.view(1, 1, feature_height, feature_width).expand(
            batch_size, -1, -1, -1
        )
        grid_y = grid_y.view(1, 1, feature_height, feature_width).expand(
            batch_size, -1, -1, -1
        )

        normalized_coordinates = torch.cat(
            [grid_x, grid_y, normalized_depth], dim=1
        )

        parsed_intrinsics = self._parse_intrinsics(
            intrinsics,
            batch_size=batch_size,
            device=raw_depth.device,
            dtype=raw_depth.dtype,
        )

        if parsed_intrinsics is None:
            anchor_position = normalized_coordinates
        else:
            fx, fy, cx, cy = parsed_intrinsics.unbind(dim=1)
            fx = fx.view(batch_size, 1, 1, 1).clamp_min(1e-6)
            fy = fy.view(batch_size, 1, 1, 1).clamp_min(1e-6)
            cx = cx.view(batch_size, 1, 1, 1)
            cy = cy.view(batch_size, 1, 1, 1)

            pixel_u = (
                (
                    torch.arange(
                        feature_width,
                        device=raw_depth.device,
                        dtype=raw_depth.dtype,
                    )
                    + 0.5
                )
                * original_width
                / feature_width
                - 0.5
            )
            pixel_v = (
                (
                    torch.arange(
                        feature_height,
                        device=raw_depth.device,
                        dtype=raw_depth.dtype,
                    )
                    + 0.5
                )
                * original_height
                / feature_height
                - 0.5
            )
            pixel_v, pixel_u = torch.meshgrid(pixel_v, pixel_u, indexing="ij")
            pixel_u = pixel_u.view(1, 1, feature_height, feature_width)
            pixel_v = pixel_v.view(1, 1, feature_height, feature_width)

            camera_x = (pixel_u - cx) * clean_depth / fx
            camera_y = (pixel_v - cy) * clean_depth / fy
            camera_z = clean_depth
            anchor_position = torch.cat([camera_x, camera_y, camera_z], dim=1)
            anchor_position = torch.where(
                valid_mask.expand_as(anchor_position),
                anchor_position,
                torch.zeros_like(anchor_position),
            )

        return anchor_position, normalized_coordinates, valid_mask

    def forward(
        self,
        rgb_features: Sequence[Tensor],
        depth_features: Sequence[Tensor],
        raw_depth: Tensor,
        intrinsics: IntrinsicsType = None,
    ) -> List[FieldLevel]:
        if len(rgb_features) != self.num_levels:
            raise ValueError(
                f"需要 {self.num_levels} 层 RGB 特征，实际得到 {len(rgb_features)}"
            )
        if len(depth_features) != self.num_levels:
            raise ValueError(
                f"需要 {self.num_levels} 层 Depth 特征，实际得到 {len(depth_features)}"
            )

        field_pyramid: List[FieldLevel] = []
        for level_index, (rgb_feature, depth_feature, ngfc_level) in enumerate(
            zip(rgb_features, depth_features, self.levels)
        ):
            if rgb_feature.ndim != 4 or depth_feature.ndim != 4:
                raise ValueError("backbone 中间特征必须为 BCHW")
            feature_size = (rgb_feature.shape[-2], rgb_feature.shape[-1])
            anchor_position, normalized_coordinates, valid_mask = (
                self._build_geometry_inputs(
                    raw_depth=raw_depth,
                    feature_size=feature_size,
                    intrinsics=intrinsics,
                )
            )

            field_units = ngfc_level(
                rgb_feature=rgb_feature,
                depth_feature=depth_feature,
                anchor_position=anchor_position,
                normalized_coordinates=normalized_coordinates,
                depth_valid_mask=valid_mask,
            )
            field_units["level_index"] = level_index
            field_pyramid.append(field_units)

        return field_pyramid


class MultiScaleFieldPropagation(nn.Module):
    """
    在二维 neural field 上执行局部场传播。

    三个不同膨胀率的 depthwise 分支近似不同作用半径的场响应，再由
    pointwise projection 聚合。该模块替代 AGPG 的离散 primitive 生成，
    保留“场属性决定响应强度与传播范围”的核心思想。
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=channels,
                        bias=False,
                    ),
                    LayerNorm2d(channels),
                    nn.GELU(),
                )
                for dilation in (1, 2, 3)
            ]
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        propagated = [branch(x) for branch in self.branches]
        return self.fusion(torch.cat(propagated, dim=1))

def quaternion_to_rotation_matrix(quaternion: Tensor) -> Tensor:
    """将最后一维为 4 的单位四元数 [w, x, y, z] 转成旋转矩阵。"""
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion 的最后一维必须为 4")
    quaternion = F.normalize(quaternion, dim=-1, eps=1e-6)
    w, x, y, z = quaternion.unbind(dim=-1)

    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    row0 = torch.stack(
        [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1
    )
    row1 = torch.stack(
        [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)], dim=-1
    )
    row2 = torch.stack(
        [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz], dim=-1
    )
    return torch.stack([row0, row1, row2], dim=-2)


class HRFRPolicy(nn.Module):
    """Hierarchical Reinforced Field Routing (HRFR-v1).

    这一版本在 DINO 已经输出多层 RGB/Depth 特征之后进行策略路由，因此能够学习：
      1. 每个 DINO level 是否参与后续 NGFC/3DGS；
      2. 每个 level 使用 RGB-only / Depth-only / RGB+Depth；
      3. 每个 level 的 Gaussian budget；
      4. 每个 level 的局部 splatting 渲染窗口。

    注意：由于策略发生在 get_intermediate_layers 之后，本版本不宣称减少 DINO backbone
    本身的 FLOPs；后续若开放 DINO block forward，可直接把 level/modality action 前移。
    """

    MODALITY_RGB = 0
    MODALITY_DEPTH = 1
    MODALITY_BOTH = 2

    def __init__(
        self,
        backbone_dim: int,
        num_levels: int,
        max_gaussians: int,
        max_render_kernel: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if num_levels < 1:
            raise ValueError("num_levels 必须大于等于 1")
        if max_gaussians < 1:
            raise ValueError("max_gaussians 必须大于等于 1")
        if max_render_kernel < 3 or max_render_kernel % 2 == 0:
            raise ValueError("max_render_kernel 必须是大于等于 3 的奇数")

        self.num_levels = num_levels
        self.max_gaussians = max_gaussians
        self.max_render_kernel = max_render_kernel

        budget_choices = [1]
        if max_gaussians >= 2:
            budget_choices.append(2)
        if max_gaussians > 2:
            budget_choices.append(max_gaussians)
        budget_choices = sorted(set(budget_choices))

        render_choices = [kernel for kernel in (3, 5, 7, 9) if kernel <= max_render_kernel]
        if max_render_kernel not in render_choices:
            render_choices.append(max_render_kernel)
        render_choices = sorted(set(render_choices))

        self.register_buffer(
            "budget_choices",
            torch.tensor(budget_choices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "render_choices",
            torch.tensor(render_choices, dtype=torch.long),
            persistent=False,
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(3 * backbone_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.level_head = nn.Linear(hidden_dim, 2)
        self.modality_head = nn.Linear(hidden_dim, 3)
        self.budget_head = nn.Linear(hidden_dim, len(budget_choices))
        self.render_head = nn.Linear(hidden_dim, len(render_choices))
        self.value_head = nn.Linear(hidden_dim, 1)

        # 初始策略偏向完整计算，避免刚开始训练时过早丢失信息。
        nn.init.zeros_(self.level_head.weight)
        nn.init.constant_(self.level_head.bias, 0.0)
        with torch.no_grad():
            self.level_head.bias[1] = 1.5

        nn.init.zeros_(self.modality_head.weight)
        nn.init.zeros_(self.modality_head.bias)
        with torch.no_grad():
            self.modality_head.bias[self.MODALITY_BOTH] = 1.5

        nn.init.zeros_(self.budget_head.weight)
        nn.init.zeros_(self.budget_head.bias)
        nn.init.zeros_(self.render_head.weight)
        nn.init.zeros_(self.render_head.bias)

    @staticmethod
    def _sample_categorical(logits: Tensor, policy_mode: str) -> Tuple[Tensor, Tensor, Tensor]:
        distribution = torch.distributions.Categorical(logits=logits.float())
        if policy_mode == "sample":
            action = distribution.sample()
        else:
            action = logits.argmax(dim=-1)
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return action, log_prob.to(logits.dtype), entropy.to(logits.dtype)

    def evaluate_rollout(
        self,
        policy_state: Tensor,
        level_action: Tensor,
        modality_action: Tensor,
        budget_action: Tensor,
        render_action: Tensor,
    ) -> Dict[str, Tensor]:
        """用缓存的 DINO state 和固定 action 重新评估当前策略，供多 epoch PPO 使用。"""
        weight_dtype = self.state_encoder[0].weight.dtype
        policy_state = policy_state.to(dtype=weight_dtype)
        hidden = self.state_encoder(policy_state)
        level_logits = self.level_head(hidden)
        modality_logits = self.modality_head(hidden)
        budget_logits = self.budget_head(hidden)
        render_logits = self.render_head(hidden)

        def evaluate(logits: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
            distribution = torch.distributions.Categorical(logits=logits.float())
            return (
                distribution.log_prob(action).to(logits.dtype),
                distribution.entropy().to(logits.dtype),
            )

        level_logp, level_entropy = evaluate(level_logits, level_action)
        modality_logp, modality_entropy = evaluate(modality_logits, modality_action)
        budget_logp, budget_entropy = evaluate(budget_logits, budget_action)
        render_logp, render_entropy = evaluate(render_logits, render_action)
        log_prob = torch.stack(
            [level_logp, modality_logp, budget_logp, render_logp], dim=-1
        ).mean(dim=(1, 2))
        entropy = torch.stack(
            [level_entropy, modality_entropy, budget_entropy, render_entropy], dim=-1
        ).mean(dim=(1, 2))
        value = self.value_head(hidden).squeeze(-1).mean(dim=1)
        return {"log_prob": log_prob, "entropy": entropy, "value": value}

    def _warmup_route(self, rgb_features: Sequence[Tensor]) -> Dict[str, Tensor]:
        batch_size = rgb_features[0].shape[0]
        device = rgb_features[0].device
        dtype = rgb_features[0].dtype
        warmup_budget = min(2, self.max_gaussians)
        warmup_kernel = min(5, self.max_render_kernel)
        if warmup_kernel % 2 == 0:
            warmup_kernel -= 1
        return {
            "level_keep": torch.ones(batch_size, self.num_levels, device=device, dtype=dtype),
            "modality_action": torch.full(
                (batch_size, self.num_levels),
                self.MODALITY_BOTH,
                device=device,
                dtype=torch.long,
            ),
            "rgb_gate": torch.ones(batch_size, self.num_levels, device=device, dtype=dtype),
            "depth_gate": torch.ones(batch_size, self.num_levels, device=device, dtype=dtype),
            "gaussian_budget": torch.full(
                (batch_size, self.num_levels), warmup_budget, device=device, dtype=torch.long
            ),
            "render_kernel": torch.full(
                (batch_size, self.num_levels), warmup_kernel, device=device, dtype=torch.long
            ),
            "log_prob": torch.zeros(batch_size, device=device, dtype=dtype),
            "entropy": torch.zeros(batch_size, device=device, dtype=dtype),
            "value": torch.zeros(batch_size, device=device, dtype=dtype),
        }

    def forward(
        self,
        rgb_features: Sequence[Tensor],
        depth_features: Sequence[Tensor],
        policy_mode: str = "warmup",
    ) -> Dict[str, Tensor]:
        if policy_mode not in {"warmup", "sample", "deterministic"}:
            raise ValueError(f"不支持的 policy_mode={policy_mode}")
        if len(rgb_features) != self.num_levels or len(depth_features) != self.num_levels:
            raise ValueError("HRFR 输入层数与 num_levels 不一致")
        if policy_mode == "warmup":
            return self._warmup_route(rgb_features)

        hidden_states: List[Tensor] = []
        policy_states: List[Tensor] = []
        for rgb_feature, depth_feature in zip(rgb_features, depth_features):
            rgb_global = F.adaptive_avg_pool2d(rgb_feature, 1).flatten(1)
            depth_global = F.adaptive_avg_pool2d(depth_feature, 1).flatten(1)
            state = torch.cat(
                [rgb_global, depth_global, torch.abs(rgb_global - depth_global)], dim=1
            )
            # RL policy 不反向修改已经提取好的视觉表征，只通过 policy gradient 学习路由。
            detached_state = state.detach()
            policy_states.append(detached_state)
            hidden_states.append(self.state_encoder(detached_state))

        policy_state = torch.stack(policy_states, dim=1)  # [B,L,3C]
        hidden = torch.stack(hidden_states, dim=1)  # [B,L,D]
        level_logits = self.level_head(hidden)
        modality_logits = self.modality_head(hidden)
        budget_logits = self.budget_head(hidden)
        render_logits = self.render_head(hidden)
        value = self.value_head(hidden).squeeze(-1)

        level_action, level_logp, level_entropy = self._sample_categorical(
            level_logits, policy_mode
        )
        modality_action, modality_logp, modality_entropy = self._sample_categorical(
            modality_logits, policy_mode
        )
        budget_action, budget_logp, budget_entropy = self._sample_categorical(
            budget_logits, policy_mode
        )
        render_action, render_logp, render_entropy = self._sample_categorical(
            render_logits, policy_mode
        )

        level_keep = (level_action == 1).to(hidden.dtype)
        # 避免一个样本所有层都被删除。若发生，保留 keep-logit 最大的一层。
        no_level = level_keep.sum(dim=1) < 0.5
        if no_level.any():
            best_level = level_logits[..., 1].argmax(dim=1)
            level_keep = level_keep.clone()
            level_action = level_action.clone()
            level_keep[no_level, best_level[no_level]] = 1.0
            level_action[no_level, best_level[no_level]] = 1
            # used action 发生了安全修正，因此重新计算对应 log-probability。
            level_distribution = torch.distributions.Categorical(logits=level_logits.float())
            level_logp = level_distribution.log_prob(level_action).to(hidden.dtype)

        rgb_gate = (modality_action != self.MODALITY_DEPTH).to(hidden.dtype)
        depth_gate = (modality_action != self.MODALITY_RGB).to(hidden.dtype)
        rgb_gate = rgb_gate * level_keep
        depth_gate = depth_gate * level_keep

        budget_choices = self.budget_choices.to(device=hidden.device)
        render_choices = self.render_choices.to(device=hidden.device)
        gaussian_budget = budget_choices[budget_action]
        render_kernel = render_choices[render_action]

        all_logp = torch.stack(
            [level_logp, modality_logp, budget_logp, render_logp], dim=-1
        ).mean(dim=(1, 2))
        all_entropy = torch.stack(
            [level_entropy, modality_entropy, budget_entropy, render_entropy], dim=-1
        ).mean(dim=(1, 2))

        return {
            "level_keep": level_keep,
            "modality_action": modality_action,
            "rgb_gate": rgb_gate,
            "depth_gate": depth_gate,
            "gaussian_budget": gaussian_budget,
            "render_kernel": render_kernel,
            "log_prob": all_logp,
            "entropy": all_entropy,
            "value": value.mean(dim=1),
            "level_logits": level_logits,
            "modality_logits": modality_logits,
            "policy_state": policy_state,
            "level_action": level_action,
            "budget_action": budget_action,
            "render_action": render_action,
        }


class GaussianRenderingAgent(nn.Module):
    """RGRA: 独立于 RAGS 的渲染后语义贡献决策器。

    RAGS 在 splatting 前改变 Gaussian candidate topology/geometry；
    RGRA 在 splatting 后调节 rendered semantic feature 注入 neural field 的程度。
    0 = SUPPRESS, 1 = KEEP, 2 = REFINE (增强 Gaussian semantic contribution)。
    它不重复 DELETE/SPLIT 或改变 position/scale/rotation/opacity。
    """

    ACTION_SUPPRESS = 0
    ACTION_KEEP = 1
    ACTION_REFINE = 2

    def __init__(self, hidden_dim: int = 32, refine_gain: float = 1.25) -> None:
        super().__init__()
        if refine_gain < 1.0:
            raise ValueError("refine_gain 必须 >= 1")
        self.refine_gain = float(refine_gain)
        # source energy / rendered energy / discrepancy / coverage /
        # confidence / uncertainty / complexity / depth validity
        self.state_encoder = nn.Sequential(
            nn.Conv2d(8, hidden_dim, 1, bias=False),
            LayerNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            LayerNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.action_head = nn.Conv2d(hidden_dim, 3, 1)
        self.value_head = nn.Conv2d(hidden_dim, 1, 1)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        with torch.no_grad():
            self.action_head.bias[self.ACTION_KEEP] = 1.5
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    @staticmethod
    def _reduce(x: Tensor, mask: Tensor) -> Tensor:
        mask = mask.to(x.dtype)
        return (x * mask).flatten(1).sum(dim=1) / mask.flatten(1).sum(dim=1).clamp_min(1)

    @staticmethod
    def make_state(
        source: Tensor,
        rendered: Tensor,
        coverage: Tensor,
        confidence: Tensor,
        uncertainty: Tensor,
        complexity: Tensor,
        depth_valid: Tensor,
    ) -> Tensor:
        return torch.cat([
            source.detach().abs().mean(dim=1, keepdim=True),
            rendered.detach().abs().mean(dim=1, keepdim=True),
            (source.detach() - rendered.detach()).abs().mean(dim=1, keepdim=True),
            coverage.detach().clamp(0, 1),
            confidence.detach(),
            uncertainty.detach(),
            complexity.detach(),
            depth_valid.detach(),
        ], dim=1)

    def forward(self, state: Tensor, mask: Tensor, mode: str) -> Dict[str, Tensor]:
        if mode not in ("sample", "deterministic"):
            raise ValueError(f"RGRA 不支持 mode={mode}")
        # PPO 与 backbone 无需相互通过 state 传播梯度。
        state = state.detach()
        hidden = self.state_encoder(state)
        logits = self.action_head(hidden)
        distribution = torch.distributions.Categorical(logits=logits.permute(0, 2, 3, 1).float())
        action = distribution.sample() if mode == "sample" else logits.argmax(dim=1)
        valid = mask[:, 0].to(dtype=logits.dtype)
        log_prob = self._reduce(distribution.log_prob(action), valid)
        entropy = self._reduce(distribution.entropy(), valid)
        value = self._reduce(self.value_head(hidden)[:, 0], valid)
        active_ratio = self._reduce((action != self.ACTION_SUPPRESS).float(), valid)
        return {
            "action": action,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
            "active_ratio": active_ratio,
            "policy_state": state,
            "decision_mask": valid.detach(),
        }

    def evaluate_rollout(self, state: Tensor, action: Tensor, mask: Tensor) -> Dict[str, Tensor]:
        hidden = self.state_encoder(state.to(self.action_head.weight.dtype))
        logits = self.action_head(hidden)
        distribution = torch.distributions.Categorical(logits=logits.permute(0, 2, 3, 1).float())
        return {
            "log_prob": self._reduce(distribution.log_prob(action), mask),
            "entropy": self._reduce(distribution.entropy(), mask),
            "value": self._reduce(self.value_head(hidden)[:, 0], mask),
        }


class NeuralGeometryGaussianSplatting(nn.Module):
    """NGFC 引导的 RAGS 自适应语义 3D Gaussian Splatting。

    RAGS 将每个 field token 的 Gaussian 结构视为一个短时序决策过程。每一步的
    离散 action 为 DELETE / KEEP / SPLIT / ADJUST / STOP，并由 actor-critic 输出
    log-probability、entropy 和 state value，供外部 PPO loss 使用。
    """

    ACTION_DELETE = 0
    ACTION_KEEP = 1
    ACTION_SPLIT = 2
    ACTION_ADJUST = 3
    ACTION_STOP = 4
    NUM_ACTIONS = 5

    def __init__(
        self,
        channels: int,
        num_gaussians_per_token: int = 2,
        kernel_size: int = 5,
        minimum_screen_sigma: float = 0.35,
        maximum_screen_sigma: Optional[float] = None,
        position_offset_scale: float = 0.75,
        rags_steps: int = 2,
    ) -> None:
        super().__init__()
        if num_gaussians_per_token < 1:
            raise ValueError("num_gaussians_per_token 必须大于等于 1")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须是大于等于 3 的奇数")
        if minimum_screen_sigma <= 0:
            raise ValueError("minimum_screen_sigma 必须大于 0")
        if rags_steps < 1:
            raise ValueError("rags_steps 必须大于等于 1")

        self.channels = channels
        self.num_gaussians = num_gaussians_per_token
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.rags_steps = int(rags_steps)
        self.minimum_screen_sigma = float(minimum_screen_sigma)
        self.maximum_screen_sigma = float(
            maximum_screen_sigma
            if maximum_screen_sigma is not None
            else max(kernel_size / 2.0, minimum_screen_sigma)
        )
        if self.maximum_screen_sigma < self.minimum_screen_sigma:
            raise ValueError("maximum_screen_sigma 不能小于 minimum_screen_sigma")
        self.position_offset_scale = float(position_offset_scale)

        self.candidate_context = nn.Sequential(
            nn.Conv2d(channels + 5, channels, kernel_size=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
            ),
            LayerNorm2d(channels),
            nn.GELU(),
        )
        self.position_delta_head = nn.Conv2d(
            channels, self.num_gaussians * 3, kernel_size=1, bias=True
        )
        self.scale_delta_head = nn.Conv2d(
            channels, self.num_gaussians * 3, kernel_size=1, bias=True
        )
        self.rotation_delta_head = nn.Conv2d(
            channels, self.num_gaussians * 4, kernel_size=1, bias=True
        )
        self.feature_delta_head = nn.Conv2d(
            channels, self.num_gaussians * channels, kernel_size=1, bias=True
        )
        self.opacity_head = nn.Conv2d(
            channels, self.num_gaussians, kernel_size=1, bias=True
        )
        self.activation_head = (
            nn.Conv2d(channels, self.num_gaussians - 1, kernel_size=1, bias=True)
            if self.num_gaussians > 1
            else None
        )

        # RAGS actor-critic + continuous parameter adjustment heads.
        self.rags_policy_context = nn.Sequential(
            nn.Conv2d(channels + 3, channels, kernel_size=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
        )
        self.rags_action_head = nn.Conv2d(channels, self.NUM_ACTIONS, kernel_size=1)
        self.rags_value_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.rags_position_adjust_head = nn.Conv2d(channels, 3, kernel_size=1)
        self.rags_scale_adjust_head = nn.Conv2d(channels, 3, kernel_size=1)
        self.rags_rotation_adjust_head = nn.Conv2d(channels, 4, kernel_size=1)
        self.rags_opacity_adjust_head = nn.Conv2d(channels, 1, kernel_size=1)

        self.render_fusion = nn.Sequential(
            nn.Conv2d(2 * channels + 1, channels, kernel_size=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.output_norm = LayerNorm2d(channels)
        self._initialize_candidate_heads()
        self._initialize_rags_heads()
        # Third RL policy is attached AFTER the complete base network is constructed.
        self.rgra_agent: Optional[GaussianRenderingAgent] = None

    def _initialize_candidate_heads(self) -> None:
        for head in (
            self.position_delta_head,
            self.scale_delta_head,
            self.rotation_delta_head,
            self.feature_delta_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.opacity_head.weight)
        nn.init.constant_(self.opacity_head.bias, 0.0)
        if self.activation_head is not None:
            nn.init.zeros_(self.activation_head.weight)
            nn.init.constant_(self.activation_head.bias, -1.5)

    def _initialize_rags_heads(self) -> None:
        nn.init.zeros_(self.rags_action_head.weight)
        nn.init.zeros_(self.rags_action_head.bias)
        # 初始更偏向 KEEP / STOP，避免未训练策略大量删除结构。
        with torch.no_grad():
            self.rags_action_head.bias[self.ACTION_KEEP] = 1.0
            self.rags_action_head.bias[self.ACTION_STOP] = 0.5
        nn.init.zeros_(self.rags_value_head.weight)
        nn.init.zeros_(self.rags_value_head.bias)
        for head in (
            self.rags_position_adjust_head,
            self.rags_scale_adjust_head,
            self.rags_rotation_adjust_head,
            self.rags_opacity_adjust_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _parse_intrinsics(
        intrinsics: IntrinsicsType,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        if intrinsics is None:
            return None
        value = torch.as_tensor(intrinsics, device=device, dtype=dtype)
        if value.ndim == 1:
            if value.numel() != 4:
                raise ValueError("intrinsics 必须为 [fx, fy, cx, cy]")
            value = value.unsqueeze(0).expand(batch_size, -1)
        elif value.ndim == 2:
            if value.shape != (batch_size, 4):
                raise ValueError("batch intrinsics 必须为 [B, 4]")
        else:
            raise ValueError("intrinsics 仅支持 [4] 或 [B, 4]")
        return value

    def _reshape_candidate_head(self, x: Tensor, channels_per_candidate: int) -> Tensor:
        batch_size, _, height, width = x.shape
        return x.view(
            batch_size,
            self.num_gaussians,
            channels_per_candidate,
            height,
            width,
        )

    @staticmethod
    def _as_batch_long(
        value: Optional[Tensor],
        default: int,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if value is None:
            return torch.full((batch_size,), default, device=device, dtype=torch.long)
        if value.ndim == 0:
            return value.to(device=device, dtype=torch.long).expand(batch_size)
        if value.ndim != 1 or value.shape[0] != batch_size:
            raise ValueError("路由 budget/kernel 必须为标量或 [B]")
        return value.to(device=device, dtype=torch.long)

    def _construct_candidates(
        self,
        source: Tensor,
        position: Tensor,
        scale: Tensor,
        rotation: Tensor,
        confidence: Tensor,
        complexity: Tensor,
        uncertainty: Tensor,
        depth_variation: Tensor,
        depth_valid: Tensor,
    ) -> Dict[str, Tensor]:
        prior_state = torch.cat(
            [confidence, complexity, uncertainty, depth_variation, depth_valid], dim=1
        )
        context = self.candidate_context(torch.cat([source, prior_state], dim=1))

        position_delta = torch.tanh(
            self._reshape_candidate_head(self.position_delta_head(context), 3)
        )
        base_scale = scale.unsqueeze(1).clamp_min(1e-5)
        candidate_position = position.unsqueeze(1) + (
            self.position_offset_scale * base_scale * position_delta
        )

        scale_delta = 0.7 * torch.tanh(
            self._reshape_candidate_head(self.scale_delta_head(context), 3)
        )
        candidate_scale = (base_scale * torch.exp(scale_delta)).clamp_min(1e-5)

        rotation_delta = self._reshape_candidate_head(self.rotation_delta_head(context), 4)
        candidate_rotation = F.normalize(
            rotation.unsqueeze(1) + 0.25 * rotation_delta,
            dim=2,
            eps=1e-6,
        )

        feature_delta = self._reshape_candidate_head(
            self.feature_delta_head(context), self.channels
        )
        candidate_feature = source.unsqueeze(1) + feature_delta

        base_reliability = confidence * (1.0 - uncertainty) * depth_valid
        opacity = torch.sigmoid(self.opacity_head(context)).unsqueeze(2)
        opacity = opacity * base_reliability.unsqueeze(1)

        activation_parts = [torch.ones_like(opacity[:, :1])]
        if self.activation_head is not None:
            density_evidence = 1.5 * complexity + depth_variation + uncertainty - 1.0
            extra_activation = torch.sigmoid(
                self.activation_head(context).unsqueeze(2)
                + density_evidence.unsqueeze(1)
            )
            activation_parts.append(extra_activation)
        heuristic_activation = torch.cat(activation_parts, dim=1)

        return {
            "position": candidate_position,
            "scale": candidate_scale,
            "rotation": candidate_rotation,
            "feature": candidate_feature,
            "opacity": opacity,
            "heuristic_activation": heuristic_activation,
            "valid": depth_valid.unsqueeze(1).expand(
                -1, self.num_gaussians, -1, -1, -1
            ),
            "context": context,
        }

    def _run_rags(
        self,
        candidates: Dict[str, Tensor],
        gaussian_budget: Tensor,
        render_kernel: Tensor,
        policy_mode: str,
    ) -> Dict[str, Tensor]:
        context = candidates["context"]
        batch_size, _, height, width = context.shape
        device, dtype = context.device, context.dtype
        gaussian_budget = gaussian_budget.clamp(1, self.num_gaussians)

        candidate_indices = torch.arange(
            self.num_gaussians, device=device, dtype=torch.long
        ).view(1, self.num_gaussians, 1, 1, 1)
        budget_mask = (
            candidate_indices
            < gaussian_budget.view(batch_size, 1, 1, 1, 1)
        ).to(dtype)

        if policy_mode == "warmup":
            activation = candidates["heuristic_activation"] * budget_mask * candidates["valid"]
            candidates["opacity"] = candidates["opacity"] * activation
            candidates["activation"] = activation
            candidates["rags_actions"] = torch.full(
                (batch_size, self.rags_steps, height, width),
                self.ACTION_KEEP,
                device=device,
                dtype=torch.long,
            )
            candidates["rags_active_count"] = activation.sum(dim=1)
            candidates["rags_log_prob"] = torch.zeros(batch_size, device=device, dtype=dtype)
            candidates["rags_entropy"] = torch.zeros(batch_size, device=device, dtype=dtype)
            candidates["rags_value"] = torch.zeros(batch_size, device=device, dtype=dtype)
            candidates["rags_policy_states"] = torch.empty(0, device=device, dtype=dtype)
            candidates["rags_decision_masks"] = torch.empty(0, device=device, dtype=dtype)
            return candidates

        current_count = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)
        stopped = torch.zeros(batch_size, height, width, device=device, dtype=torch.bool)
        action_history: List[Tensor] = []
        policy_state_history: List[Tensor] = []
        decision_mask_history: List[Tensor] = []
        logp_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        entropy_sum = torch.zeros_like(logp_sum)
        value_sum = torch.zeros_like(logp_sum)
        decision_count = torch.zeros_like(logp_sum)

        budget_map = gaussian_budget.to(dtype).view(batch_size, 1, 1, 1).expand(
            -1, -1, height, width
        )
        render_norm = (
            render_kernel.to(dtype) / float(max(self.kernel_size, 1))
        ).view(batch_size, 1, 1, 1).expand(-1, -1, height, width)

        candidate_position = candidates["position"]
        candidate_scale = candidates["scale"]
        candidate_rotation = candidates["rotation"]
        candidate_opacity = candidates["opacity"]

        for _ in range(self.rags_steps):
            active_ratio = current_count / float(max(self.num_gaussians, 1))
            budget_ratio = budget_map / float(max(self.num_gaussians, 1))
            policy_state = torch.cat(
                [context.detach(), active_ratio.detach(), budget_ratio.detach(), render_norm.detach()],
                dim=1,
            )
            policy_state_history.append(policy_state.detach())
            hidden = self.rags_policy_context(policy_state)
            action_logits = self.rags_action_head(hidden)
            value_map = self.rags_value_head(hidden).squeeze(1)

            distribution = torch.distributions.Categorical(
                logits=action_logits.permute(0, 2, 3, 1).float()
            )
            if policy_mode == "sample":
                action = distribution.sample()
            else:
                action = action_logits.argmax(dim=1)

            alive = ~stopped
            decision_mask_history.append(alive.to(dtype).detach())
            log_prob = distribution.log_prob(action).to(dtype)
            entropy = distribution.entropy().to(dtype)
            alive_float = alive.to(dtype)
            spatial_denom = alive_float.flatten(1).sum(dim=1).clamp_min(1.0)
            logp_sum += (log_prob * alive_float).flatten(1).sum(dim=1) / spatial_denom
            entropy_sum += (entropy * alive_float).flatten(1).sum(dim=1) / spatial_denom
            value_sum += (value_map * alive_float).flatten(1).sum(dim=1) / spatial_denom
            decision_count += (spatial_denom > 0).to(dtype)

            # 已 STOP 的 token 在诊断中保持 STOP。
            action = torch.where(
                alive,
                action,
                torch.full_like(action, self.ACTION_STOP),
            )
            action_history.append(action)

            delete_mask = alive & (action == self.ACTION_DELETE)
            split_mask = alive & (action == self.ACTION_SPLIT)
            adjust_mask = alive & (action == self.ACTION_ADJUST)
            stop_mask = alive & (action == self.ACTION_STOP)

            current_count = torch.where(
                delete_mask.unsqueeze(1),
                torch.zeros_like(current_count),
                current_count,
            )
            current_count = torch.where(
                split_mask.unsqueeze(1),
                torch.maximum(current_count, budget_map),
                current_count,
            )

            if adjust_mask.any():
                adjust = adjust_mask.to(dtype).unsqueeze(1)
                pos_delta = 0.25 * torch.tanh(self.rags_position_adjust_head(hidden))
                scale_delta = 0.25 * torch.tanh(self.rags_scale_adjust_head(hidden))
                rot_delta = 0.10 * self.rags_rotation_adjust_head(hidden)
                opacity_delta = 0.50 * torch.tanh(self.rags_opacity_adjust_head(hidden))

                candidate_position = candidate_position + (
                    candidate_scale * pos_delta.unsqueeze(1) * adjust.unsqueeze(1)
                )
                candidate_scale = candidate_scale * torch.exp(
                    scale_delta.unsqueeze(1) * adjust.unsqueeze(1)
                )
                candidate_scale = candidate_scale.clamp_min(1e-5)
                candidate_rotation = F.normalize(
                    candidate_rotation + rot_delta.unsqueeze(1) * adjust.unsqueeze(1),
                    dim=2,
                    eps=1e-6,
                )
                candidate_opacity = (
                    candidate_opacity
                    * torch.exp(opacity_delta.unsqueeze(1) * adjust.unsqueeze(1))
                ).clamp(0.0, 0.995)

            stopped = stopped | delete_mask | stop_mask

        active_mask = (
            candidate_indices
            < current_count.round().long().unsqueeze(1)
        ).to(dtype)
        activation = active_mask * budget_mask * candidates["valid"]
        candidate_opacity = candidate_opacity * activation

        denom = decision_count.clamp_min(1.0)
        candidates["position"] = candidate_position
        candidates["scale"] = candidate_scale
        candidates["rotation"] = candidate_rotation
        candidates["opacity"] = candidate_opacity
        candidates["activation"] = activation
        candidates["rags_actions"] = torch.stack(action_history, dim=1)
        candidates["rags_active_count"] = activation.sum(dim=1)
        candidates["rags_log_prob"] = logp_sum / denom
        candidates["rags_entropy"] = entropy_sum / denom
        candidates["rags_value"] = value_sum / denom
        candidates["rags_policy_states"] = torch.stack(policy_state_history, dim=1)
        candidates["rags_decision_masks"] = torch.stack(decision_mask_history, dim=1)
        return candidates

    def evaluate_rags_rollout(
        self,
        policy_states: Tensor,
        actions: Tensor,
        decision_masks: Tensor,
    ) -> Dict[str, Tensor]:
        """重新评估缓存的 RAGS state/action，避免为 PPO replay 重跑 DINO/NGFC。"""
        if policy_states.ndim != 5:
            raise ValueError("RAGS policy_states 应为 [B,T,C,H,W]")
        batch_size, steps = policy_states.shape[:2]
        dtype = self.rags_action_head.weight.dtype
        logp_sum = torch.zeros(batch_size, device=policy_states.device, dtype=dtype)
        entropy_sum = torch.zeros_like(logp_sum)
        value_sum = torch.zeros_like(logp_sum)
        count = torch.zeros_like(logp_sum)

        for step in range(steps):
            state = policy_states[:, step].to(dtype=dtype)
            mask = decision_masks[:, step].to(dtype=dtype)
            hidden = self.rags_policy_context(state)
            logits = self.rags_action_head(hidden)
            value_map = self.rags_value_head(hidden).squeeze(1)
            distribution = torch.distributions.Categorical(
                logits=logits.permute(0, 2, 3, 1).float()
            )
            action = actions[:, step]
            log_prob = distribution.log_prob(action).to(dtype)
            entropy = distribution.entropy().to(dtype)
            denom = mask.flatten(1).sum(dim=1).clamp_min(1.0)
            logp_sum += (log_prob * mask).flatten(1).sum(dim=1) / denom
            entropy_sum += (entropy * mask).flatten(1).sum(dim=1) / denom
            value_sum += (value_map * mask).flatten(1).sum(dim=1) / denom
            count += (denom > 0).to(dtype)

        count = count.clamp_min(1.0)
        return {
            "log_prob": logp_sum / count,
            "entropy": entropy_sum / count,
            "value": value_sum / count,
        }

    def _project_gaussians(
        self,
        position: Tensor,
        scale: Tensor,
        rotation: Tensor,
        image_size: Tuple[int, int],
        intrinsics: IntrinsicsType,
    ) -> Dict[str, Tensor]:
        batch_size, num_gaussians, _, height, width = position.shape
        if num_gaussians != self.num_gaussians:
            raise RuntimeError("Gaussian candidate 数量不一致")

        position_last = position.permute(0, 1, 3, 4, 2).contiguous()
        scale_last = scale.permute(0, 1, 3, 4, 2).contiguous()
        rotation_last = rotation.permute(0, 1, 3, 4, 2).contiguous()

        rotation_matrix = quaternion_to_rotation_matrix(rotation_last)
        scale_square = scale_last.square()
        covariance_3d = torch.matmul(
            rotation_matrix * scale_square.unsqueeze(-2),
            rotation_matrix.transpose(-1, -2),
        )

        original_height, original_width = image_size
        parsed_intrinsics = self._parse_intrinsics(
            intrinsics,
            batch_size=batch_size,
            device=position.device,
            dtype=position.dtype,
        )

        x_coord = position_last[..., 0]
        y_coord = position_last[..., 1]
        z_coord = position_last[..., 2]

        if parsed_intrinsics is None:
            center_x = (x_coord + 1.0) * 0.5 * max(width - 1, 1)
            center_y = (y_coord + 1.0) * 0.5 * max(height - 1, 1)
            jacobian = torch.zeros(
                batch_size,
                self.num_gaussians,
                height,
                width,
                2,
                3,
                device=position.device,
                dtype=position.dtype,
            )
            jacobian[..., 0, 0] = 0.5 * max(width - 1, 1)
            jacobian[..., 1, 1] = 0.5 * max(height - 1, 1)
            projection_valid = torch.isfinite(position_last).all(dim=-1)
        else:
            fx, fy, cx, cy = parsed_intrinsics.unbind(dim=1)
            view_shape = (batch_size, 1, 1, 1)
            fx = fx.view(view_shape)
            fy = fy.view(view_shape)
            cx = cx.view(view_shape)
            cy = cy.view(view_shape)
            projection_valid = torch.isfinite(position_last).all(dim=-1) & (z_coord > 1e-4)
            safe_z = z_coord.clamp_min(1e-4)
            width_ratio = float(width) / float(original_width)
            height_ratio = float(height) / float(original_height)
            center_x = ((fx * x_coord / safe_z + cx) + 0.5) * width_ratio - 0.5
            center_y = ((fy * y_coord / safe_z + cy) + 0.5) * height_ratio - 0.5
            jacobian = torch.zeros(
                batch_size,
                self.num_gaussians,
                height,
                width,
                2,
                3,
                device=position.device,
                dtype=position.dtype,
            )
            jacobian[..., 0, 0] = fx / safe_z * width_ratio
            jacobian[..., 0, 2] = -fx * x_coord / safe_z.square() * width_ratio
            jacobian[..., 1, 1] = fy / safe_z * height_ratio
            jacobian[..., 1, 2] = -fy * y_coord / safe_z.square() * height_ratio

        covariance_2d = torch.matmul(
            torch.matmul(jacobian, covariance_3d), jacobian.transpose(-1, -2)
        )
        min_variance = self.minimum_screen_sigma ** 2
        max_variance = self.maximum_screen_sigma ** 2
        variance_x = covariance_2d[..., 0, 0].clamp(min_variance, max_variance)
        variance_y = covariance_2d[..., 1, 1].clamp(min_variance, max_variance)
        covariance_xy = covariance_2d[..., 0, 1]
        correlation_bound = 0.95 * torch.sqrt(variance_x * variance_y)
        covariance_xy = torch.maximum(
            torch.minimum(covariance_xy, correlation_bound), -correlation_bound
        )
        determinant = (variance_x * variance_y - covariance_xy.square()).clamp_min(1e-6)
        inverse_xx = variance_y / determinant
        inverse_yy = variance_x / determinant
        inverse_xy = -covariance_xy / determinant
        trace = variance_x + variance_y
        discriminant = torch.sqrt(
            (variance_x - variance_y).square() + 4.0 * covariance_xy.square() + 1e-8
        )
        major_sigma = torch.sqrt(
            (0.5 * (trace + discriminant)).clamp_min(min_variance)
        )

        def to_map(x: Tensor) -> Tensor:
            return x.unsqueeze(2).contiguous()

        return {
            "center_x": to_map(center_x),
            "center_y": to_map(center_y),
            "inverse_xx": to_map(inverse_xx),
            "inverse_xy": to_map(inverse_xy),
            "inverse_yy": to_map(inverse_yy),
            "depth": to_map(z_coord),
            "projection_valid": to_map(projection_valid.to(position.dtype)),
            "major_sigma": to_map(major_sigma),
        }

    def _unfold_candidates(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"candidate map 必须为 [B,G,C,H,W]，实际为 {tuple(x.shape)}")
        batch_size, num_gaussians, channels, height, width = x.shape
        patches = F.unfold(
            x.reshape(batch_size, num_gaussians * channels, height, width),
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        patches = patches.view(
            batch_size,
            num_gaussians,
            channels,
            self.kernel_size * self.kernel_size,
            height * width,
        )
        return (
            patches.permute(0, 2, 1, 3, 4)
            .reshape(
                batch_size,
                channels,
                num_gaussians * self.kernel_size * self.kernel_size,
                height * width,
            )
            .contiguous()
        )

    def _render_range_mask(
        self,
        render_kernel: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        offsets = torch.arange(self.kernel_size, device=device) - self.padding
        grid_y, grid_x = torch.meshgrid(offsets, offsets, indexing="ij")
        radius = torch.maximum(grid_x.abs(), grid_y.abs()).reshape(-1)
        radius = radius.repeat(self.num_gaussians).view(1, -1, 1)
        allowed_radius = ((render_kernel - 1) // 2).view(batch_size, 1, 1)
        return (radius <= allowed_radius).to(dtype)

    def _render_local_splats(
        self,
        candidate_feature: Tensor,
        opacity: Tensor,
        valid: Tensor,
        projection: Dict[str, Tensor],
        render_kernel: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        batch_size, _, channels, height, width = candidate_feature.shape
        source_feature = self._unfold_candidates(candidate_feature)

        def unfold_scalar(x: Tensor) -> Tensor:
            return self._unfold_candidates(x).squeeze(1)

        center_x = unfold_scalar(projection["center_x"])
        center_y = unfold_scalar(projection["center_y"])
        inverse_xx = unfold_scalar(projection["inverse_xx"])
        inverse_xy = unfold_scalar(projection["inverse_xy"])
        inverse_yy = unfold_scalar(projection["inverse_yy"])
        source_depth = unfold_scalar(projection["depth"])
        alpha_base = unfold_scalar(opacity)
        source_valid = unfold_scalar(valid * projection["projection_valid"])
        source_valid = source_valid * self._render_range_mask(
            render_kernel, batch_size, candidate_feature.device, candidate_feature.dtype
        )

        target_y, target_x = torch.meshgrid(
            torch.arange(height, device=candidate_feature.device, dtype=candidate_feature.dtype),
            torch.arange(width, device=candidate_feature.device, dtype=candidate_feature.dtype),
            indexing="ij",
        )
        target_x = target_x.reshape(1, 1, height * width)
        target_y = target_y.reshape(1, 1, height * width)
        delta_x = target_x - center_x
        delta_y = target_y - center_y
        mahalanobis = (
            inverse_xx * delta_x.square()
            + 2.0 * inverse_xy * delta_x * delta_y
            + inverse_yy * delta_y.square()
        )
        gaussian_response = torch.exp((-0.5 * mahalanobis).clamp(min=-20.0, max=0.0))
        alpha = (alpha_base * gaussian_response * source_valid).clamp(0.0, 0.995)

        large_depth = torch.finfo(source_depth.dtype).max / 1000.0
        sortable_depth = torch.where(
            source_valid > 0.5,
            source_depth,
            torch.full_like(source_depth, large_depth),
        )
        order = torch.argsort(sortable_depth, dim=1, descending=False)
        alpha = torch.gather(alpha, dim=1, index=order)
        ordered_feature = torch.gather(
            source_feature,
            dim=2,
            index=order.unsqueeze(1).expand(-1, channels, -1, -1),
        )
        one_minus_alpha = (1.0 - alpha).clamp_min(1e-6)
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[:, :1]), one_minus_alpha], dim=1),
            dim=1,
        )[:, :-1]
        compositing_weight = alpha * transmittance
        coverage = compositing_weight.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        rendered = (ordered_feature * compositing_weight.unsqueeze(1)).sum(dim=2)
        rendered = rendered / coverage.clamp_min(1e-6)
        rendered = rendered.view(batch_size, channels, height, width)
        coverage = coverage.view(batch_size, 1, height, width)
        return rendered, coverage

    def forward(
        self,
        source: Tensor,
        position: Tensor,
        scale: Tensor,
        rotation: Tensor,
        confidence: Tensor,
        complexity: Tensor,
        uncertainty: Tensor,
        depth_variation: Tensor,
        depth_valid: Tensor,
        image_size: Tuple[int, int],
        intrinsics: IntrinsicsType = None,
        gaussian_budget: Optional[Tensor] = None,
        render_kernel: Optional[Tensor] = None,
        policy_mode: str = "warmup",
        use_rgra_agent: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        batch_size = source.shape[0]
        gaussian_budget = self._as_batch_long(
            gaussian_budget, min(2, self.num_gaussians), batch_size, source.device
        )
        render_kernel = self._as_batch_long(
            render_kernel, min(5, self.kernel_size), batch_size, source.device
        )
        render_kernel = render_kernel.clamp(3, self.kernel_size)
        render_kernel = render_kernel - ((render_kernel + 1) % 2)

        candidates = self._construct_candidates(
            source=source,
            position=position,
            scale=scale,
            rotation=rotation,
            confidence=confidence,
            complexity=complexity,
            uncertainty=uncertainty,
            depth_variation=depth_variation,
            depth_valid=depth_valid,
        )
        candidates = self._run_rags(
            candidates,
            gaussian_budget=gaussian_budget,
            render_kernel=render_kernel,
            policy_mode=policy_mode,
        )
        projection = self._project_gaussians(
            position=candidates["position"],
            scale=candidates["scale"],
            rotation=candidates["rotation"],
            image_size=image_size,
            intrinsics=intrinsics,
        )
        rendered, coverage = self._render_local_splats(
            candidate_feature=candidates["feature"],
            opacity=candidates["opacity"],
            valid=candidates["valid"],
            projection=projection,
            render_kernel=render_kernel,
        )
        # RGRA acts on the RENDERED SEMANTIC CONTRIBUTION only.
        # It does not repeat RAGS topology control or change Gaussian geometry.
        rg_enabled = bool(use_rgra_agent) and policy_mode != "warmup"
        rg_stats: Optional[Dict[str, Tensor]] = None
        if rg_enabled:
            if self.rgra_agent is None:
                raise RuntimeError("模型初始化时未创建 RGRA，请设置 use_rgra_agent=True")
            rg_state = self.rgra_agent.make_state(
                source, rendered, coverage, confidence, uncertainty, complexity, depth_valid
            )
            rg_stats = self.rgra_agent(rg_state, depth_valid, mode=policy_mode)
            rg_action = rg_stats["action"]
            rg_gain = torch.where(
                rg_action == GaussianRenderingAgent.ACTION_SUPPRESS,
                0.0,
                torch.where(
                    rg_action == GaussianRenderingAgent.ACTION_REFINE,
                    self.rgra_agent.refine_gain,
                    1.0,
                ),
            ).unsqueeze(1).to(rendered.dtype)
            # SUPPRESS only removes the splatted semantic signal from fusion.
            # KEEP = original path, REFINE = increase local Gaussian contribution.
            rendered_for_fusion = rendered * rg_gain
            coverage_for_fusion = coverage * rg_gain.clamp(max=1.0)
        else:
            rendered_for_fusion = rendered
            coverage_for_fusion = coverage

        fused_update = self.render_fusion(torch.cat([source, rendered_for_fusion, coverage_for_fusion], dim=1))
        output = self.output_norm(source + fused_update)
        zero = source.new_zeros(batch_size)
        if rg_stats is None:
            rg_stats = {
                "action": torch.full(source.shape[:1] + source.shape[-2:],
                                     GaussianRenderingAgent.ACTION_KEEP,
                                     device=source.device, dtype=torch.long),
                "log_prob": zero, "entropy": zero, "value": zero,
                "active_ratio": torch.ones_like(zero),
                "policy_state": source.new_empty(0),
                "decision_mask": source.new_empty(0),
            }
        diagnostics = {
            "gaussian_opacity": candidates["opacity"],
            "gaussian_activation": candidates["activation"],
            "gaussian_coverage": coverage,
            "gaussian_major_sigma": projection["major_sigma"],
            "gaussian_center_x": projection["center_x"],
            "gaussian_center_y": projection["center_y"],
            "rendered_gaussian_feature": rendered,
            "rags_actions": candidates["rags_actions"],
            "rags_active_count": candidates["rags_active_count"],
            "rags_log_prob": candidates["rags_log_prob"],
            "rags_entropy": candidates["rags_entropy"],
            "rags_value": candidates["rags_value"],
            "rags_policy_states": candidates["rags_policy_states"],
            "rags_decision_masks": candidates["rags_decision_masks"],
            "gaussian_budget": gaussian_budget,
            "render_kernel": render_kernel,
            "rgra_action": rg_stats["action"],
            "rgra_log_prob": rg_stats["log_prob"],
            "rgra_entropy": rg_stats["entropy"],
            "rgra_value": rg_stats["value"],
            "rgra_active_ratio": rg_stats["active_ratio"],
            "rgra_policy_state": rg_stats["policy_state"],
            "rgra_decision_mask": rg_stats["decision_mask"],
        }
        return output, diagnostics


class FieldLevelToSegFeature(nn.Module):
    """
    将一个 NGFC level 转换为二维语义场，并通过 NGS 模块执行 3D Gaussian splatting。

    这里保留 NGFC 的 field / semantic / geometry / reliability 表达，但不再使用
    固定空洞卷积近似场传播；位置、尺度和旋转会直接构造 Gaussian mean 与
    covariance，并真实参与投影和 alpha compositing。
    """

    def __init__(
        self,
        field_dim: int,
        semantic_dim: int,
        out_dim: int,
        num_gaussians_per_token: int = 2,
        gaussian_kernel_size: int = 5,
        minimum_screen_sigma: float = 0.35,
        maximum_screen_sigma: Optional[float] = None,
        rags_steps: int = 2,
    ) -> None:
        super().__init__()
        self.field_dim = field_dim
        self.semantic_dim = semantic_dim

        self.field_projection = ConvNormAct(field_dim, out_dim)
        self.semantic_projection = ConvNormAct(semantic_dim, out_dim)
        self.geometry_projection = ConvNormAct(10, out_dim)
        self.prior_projection = ConvNormAct(5, out_dim)

        self.source_fusion = nn.Sequential(
            nn.Conv2d(4 * out_dim, 2 * out_dim, kernel_size=1, bias=False),
            LayerNorm2d(2 * out_dim),
            nn.GELU(),
            nn.Conv2d(2 * out_dim, out_dim, kernel_size=1, bias=False),
            LayerNorm2d(out_dim),
            nn.GELU(),
        )
        self.strength_head = nn.Sequential(
            nn.Conv2d(5, max(out_dim // 4, 8), kernel_size=1, bias=False),
            LayerNorm2d(max(out_dim // 4, 8)),
            nn.GELU(),
            nn.Conv2d(max(out_dim // 4, 8), out_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.gaussian_splatting = NeuralGeometryGaussianSplatting(
            channels=out_dim,
            num_gaussians_per_token=num_gaussians_per_token,
            kernel_size=gaussian_kernel_size,
            minimum_screen_sigma=minimum_screen_sigma,
            maximum_screen_sigma=maximum_screen_sigma,
            rags_steps=rags_steps,
        )

    @staticmethod
    def _unflatten(x: Tensor, spatial_shape: Tuple[int, int], name: str) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"{name} 应为 [B, N, C]，实际得到 {tuple(x.shape)}")
        height, width = spatial_shape
        if x.shape[1] != height * width:
            raise ValueError(
                f"{name} 的 token 数 {x.shape[1]} 与空间尺寸 {spatial_shape} 不一致"
            )
        return x.transpose(1, 2).reshape(
            x.shape[0], x.shape[2], height, width
        ).contiguous()

    @staticmethod
    def _masked_standardize(x: Tensor, valid: Tensor) -> Tensor:
        valid = valid.to(x.dtype)
        count = valid.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        mean = (x * valid).sum(dim=(2, 3), keepdim=True) / count
        variance = ((x - mean).square() * valid).sum(
            dim=(2, 3), keepdim=True
        ) / count
        standardized = (x - mean) / torch.sqrt(variance + 1e-6)
        return standardized.clamp(-5.0, 5.0) * valid

    def forward(
        self,
        field_units: FieldLevel,
        image_size: Tuple[int, int],
        intrinsics: IntrinsicsType = None,
        gaussian_budget: Optional[Tensor] = None,
        render_kernel: Optional[Tensor] = None,
        policy_mode: str = "warmup",
        use_rgra_agent: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        spatial_shape_value = field_units.get("spatial_shape")
        if not isinstance(spatial_shape_value, tuple) or len(spatial_shape_value) != 2:
            raise TypeError("field_units['spatial_shape'] 必须为 (H, W)")
        spatial_shape = (int(spatial_shape_value[0]), int(spatial_shape_value[1]))

        def get_tensor(key: str) -> Tensor:
            value = field_units.get(key)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"field_units['{key}'] 必须是 Tensor")
            return self._unflatten(value, spatial_shape, key)

        position = get_tensor("position")
        scale = get_tensor("scale")
        rotation = get_tensor("rotation")
        field_descriptor = get_tensor("field")
        semantic_descriptor = get_tensor("semantic")
        confidence = get_tensor("confidence")
        complexity = get_tensor("complexity")
        uncertainty = get_tensor("semantic_uncertainty")
        depth_variation = get_tensor("depth_variation")
        depth_valid = get_tensor("depth_valid").clamp(0.0, 1.0)

        normalized_position = self._masked_standardize(position, depth_valid)
        normalized_log_scale = self._masked_standardize(
            torch.log(scale.clamp_min(1e-6)), depth_valid
        )
        geometry_state = torch.cat(
            [normalized_position, normalized_log_scale, rotation], dim=1
        )
        prior_state = torch.cat(
            [confidence, complexity, uncertainty, depth_variation, depth_valid],
            dim=1,
        )

        field_source = self.field_projection(field_descriptor)
        semantic_source = self.semantic_projection(semantic_descriptor)
        geometry_source = self.geometry_projection(geometry_state)
        prior_source = self.prior_projection(prior_state)
        source = self.source_fusion(
            torch.cat(
                [field_source, semantic_source, geometry_source, prior_source],
                dim=1,
            )
        )

        reliability = confidence * (1.0 - uncertainty) * depth_valid
        activity = torch.clamp(complexity + depth_variation, 0.0, 1.0)
        learned_strength = self.strength_head(prior_state)
        physical_strength = reliability * (1.0 + activity)
        field_strength = learned_strength * (0.5 + physical_strength)
        activated_source = source * field_strength

        output, gaussian_diagnostics = self.gaussian_splatting(
            source=activated_source,
            position=position,
            scale=scale,
            rotation=rotation,
            confidence=confidence,
            complexity=complexity,
            uncertainty=uncertainty,
            depth_variation=depth_variation,
            depth_valid=depth_valid,
            image_size=image_size,
            intrinsics=intrinsics,
            gaussian_budget=gaussian_budget,
            render_kernel=render_kernel,
            policy_mode=policy_mode,
            use_rgra_agent=use_rgra_agent,
        )
        diagnostics = {
            "field_strength": field_strength,
            "reliability": reliability,
            "activity": activity,
            **gaussian_diagnostics,
        }
        return output, diagnostics

class CrossLevelFieldFusion(nn.Module):
    """对不同 DINO 深度对应 neural fields 执行 HRFR-aware 融合。"""

    def __init__(self, channels: int, num_levels: int) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.level_score_heads = nn.ModuleList(
            [nn.Conv2d(channels, 1, kernel_size=1, bias=True) for _ in range(num_levels)]
        )
        self.concat_fusion = nn.Sequential(
            nn.Conv2d(num_levels * channels, 2 * channels, kernel_size=1, bias=False),
            LayerNorm2d(2 * channels),
            nn.GELU(),
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
        )
        self.refinement = nn.Sequential(
            MultiScaleFieldPropagation(channels),
            LayerNorm2d(channels),
            nn.GELU(),
        )
        self.output_norm = LayerNorm2d(channels)

    def forward(
        self,
        level_features: Sequence[Tensor],
        level_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if len(level_features) != self.num_levels:
            raise ValueError(
                f"需要 {self.num_levels} 个 level feature，实际得到 {len(level_features)}"
            )
        target_height = max(feature.shape[-2] for feature in level_features)
        target_width = max(feature.shape[-1] for feature in level_features)
        target_size = (target_height, target_width)

        aligned_features: List[Tensor] = []
        scores: List[Tensor] = []
        for feature, score_head in zip(level_features, self.level_score_heads):
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature, size=target_size, mode="bilinear", align_corners=False
                )
            aligned_features.append(feature)
            scores.append(score_head(feature).mean(dim=(2, 3), keepdim=True))

        batch_size = aligned_features[0].shape[0]
        if level_mask is None:
            level_mask = torch.ones(
                batch_size,
                self.num_levels,
                device=aligned_features[0].device,
                dtype=aligned_features[0].dtype,
            )
        if level_mask.shape != (batch_size, self.num_levels):
            raise ValueError("level_mask 应为 [B,L]")
        level_mask = level_mask.to(aligned_features[0].dtype)

        level_scores = torch.stack(scores, dim=1)  # [B,L,1,1,1]
        mask5d = level_mask.view(batch_size, self.num_levels, 1, 1, 1)
        masked_scores = level_scores.masked_fill(mask5d < 0.5, -1e4)
        level_weights = torch.softmax(masked_scores, dim=1) * mask5d
        level_weights = level_weights / level_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        stacked = torch.stack(aligned_features, dim=1)
        weighted_field = (stacked * level_weights).sum(dim=1)
        masked_aligned = [
            feature * level_mask[:, i].view(batch_size, 1, 1, 1)
            for i, feature in enumerate(aligned_features)
        ]
        concatenated_field = self.concat_fusion(torch.cat(masked_aligned, dim=1))
        fused = weighted_field + concatenated_field
        fused = self.output_norm(fused + self.refinement(fused))
        return fused, level_weights.squeeze(-1).squeeze(-1).squeeze(-1)


class UpsampleRefineBlock(nn.Module):
    """2 倍上采样并进行局部边界细化。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            make_group_norm(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            make_group_norm(out_channels),
            nn.GELU(),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.block(x) + self.skip(x)


class NeuralFieldSegmentationHead(nn.Module):
    """NGFC -> RAGS semantic 3DGS -> HRFR cross-level fusion -> segmentation。"""

    def __init__(
        self,
        field_dim: int,
        semantic_dim: int,
        decoder_dim: int,
        num_levels: int,
        num_classes: int,
        num_upsample_stages: int = 4,
        num_gaussians_per_token: int = 2,
        gaussian_kernel_size: int = 5,
        minimum_screen_sigma: float = 0.35,
        maximum_screen_sigma: Optional[float] = None,
        rags_steps: int = 2,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes 必须大于等于 1")
        self.num_levels = num_levels
        self.level_adapters = nn.ModuleList(
            [
                FieldLevelToSegFeature(
                    field_dim=field_dim,
                    semantic_dim=semantic_dim,
                    out_dim=decoder_dim,
                    num_gaussians_per_token=num_gaussians_per_token,
                    gaussian_kernel_size=gaussian_kernel_size,
                    minimum_screen_sigma=minimum_screen_sigma,
                    maximum_screen_sigma=maximum_screen_sigma,
                    rags_steps=rags_steps,
                )
                for _ in range(num_levels)
            ]
        )
        self.cross_level_fusion = CrossLevelFieldFusion(decoder_dim, num_levels)

        upsample_channels: List[int] = []
        for stage_index in range(num_upsample_stages):
            if stage_index < 2:
                next_channels = decoder_dim
            elif stage_index == 2:
                next_channels = max(decoder_dim // 2, 32)
            else:
                next_channels = max(decoder_dim // 4, 32)
            upsample_channels.append(next_channels)

        blocks: List[nn.Module] = []
        current_channels = decoder_dim
        for next_channels in upsample_channels:
            blocks.append(UpsampleRefineBlock(current_channels, next_channels))
            current_channels = next_channels
        self.upsample_blocks = nn.ModuleList(blocks)
        head_hidden = max(current_channels // 2, 32)
        self.classifier = nn.Sequential(
            nn.Conv2d(current_channels, head_hidden, kernel_size=3, padding=1, bias=False),
            make_group_norm(head_hidden),
            nn.GELU(),
            nn.Conv2d(head_hidden, num_classes, kernel_size=1, bias=True),
        )

    def forward(
        self,
        field_pyramid: Sequence[FieldLevel],
        output_size: Tuple[int, int],
        intrinsics: IntrinsicsType = None,
        routing: Optional[Dict[str, Tensor]] = None,
        policy_mode: str = "warmup",
        return_auxiliary: bool = False,
        use_rgra_agent: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        if len(field_pyramid) != len(self.level_adapters):
            raise ValueError("field_pyramid 层数与分割头层数不一致")

        batch_size = next(
            value.shape[0]
            for value in field_pyramid[0].values()
            if isinstance(value, torch.Tensor)
        )
        if routing is None:
            device = next(self.parameters()).device
            level_keep = torch.ones(batch_size, self.num_levels, device=device)
            gaussian_budget = torch.full(
                (batch_size, self.num_levels), 2, device=device, dtype=torch.long
            )
            render_kernel = torch.full(
                (batch_size, self.num_levels), 5, device=device, dtype=torch.long
            )
        else:
            level_keep = routing["level_keep"]
            gaussian_budget = routing["gaussian_budget"]
            render_kernel = routing["render_kernel"]

        level_features: List[Tensor] = []
        diagnostics_by_key: Dict[str, List[Tensor]] = {
            "level_field_strengths": [],
            "level_reliabilities": [],
            "level_gaussian_opacities": [],
            "level_gaussian_activations": [],
            "level_gaussian_coverages": [],
            "level_gaussian_sigmas": [],
            "level_gaussian_centers_x": [],
            "level_gaussian_centers_y": [],
            "level_rags_actions": [],
            "level_rags_active_counts": [],
            "level_rags_policy_states": [],
            "level_rags_decision_masks": [],
            "level_rgra_actions": [],
            "level_rgra_policy_states": [],
            "level_rgra_decision_masks": [],
            "level_rgra_active_ratios": [],
        }
        rgra_logps: List[Tensor] = []
        rgra_entropies: List[Tensor] = []
        rgra_values: List[Tensor] = []
        rags_logps: List[Tensor] = []
        rags_entropies: List[Tensor] = []
        rags_values: List[Tensor] = []

        for level_index, (field_units, adapter) in enumerate(
            zip(field_pyramid, self.level_adapters)
        ):
            level_feature, diagnostics = adapter(
                field_units,
                image_size=output_size,
                intrinsics=intrinsics,
                gaussian_budget=gaussian_budget[:, level_index],
                render_kernel=render_kernel[:, level_index],
                policy_mode=policy_mode,
                use_rgra_agent=use_rgra_agent,
            )
            keep = level_keep[:, level_index].view(batch_size, 1, 1, 1).to(level_feature.dtype)
            level_feature = level_feature * keep
            level_features.append(level_feature)

            diagnostics_by_key["level_field_strengths"].append(diagnostics["field_strength"])
            diagnostics_by_key["level_reliabilities"].append(diagnostics["reliability"])
            diagnostics_by_key["level_gaussian_opacities"].append(diagnostics["gaussian_opacity"])
            diagnostics_by_key["level_gaussian_activations"].append(diagnostics["gaussian_activation"])
            diagnostics_by_key["level_gaussian_coverages"].append(diagnostics["gaussian_coverage"])
            diagnostics_by_key["level_gaussian_sigmas"].append(diagnostics["gaussian_major_sigma"])
            diagnostics_by_key["level_gaussian_centers_x"].append(diagnostics["gaussian_center_x"])
            diagnostics_by_key["level_gaussian_centers_y"].append(diagnostics["gaussian_center_y"])
            diagnostics_by_key["level_rags_actions"].append(diagnostics["rags_actions"])
            diagnostics_by_key["level_rags_active_counts"].append(diagnostics["rags_active_count"])
            diagnostics_by_key["level_rags_policy_states"].append(diagnostics["rags_policy_states"])
            diagnostics_by_key["level_rags_decision_masks"].append(diagnostics["rags_decision_masks"])
            diagnostics_by_key["level_rgra_actions"].append(diagnostics["rgra_action"])
            diagnostics_by_key["level_rgra_policy_states"].append(diagnostics["rgra_policy_state"])
            diagnostics_by_key["level_rgra_decision_masks"].append(diagnostics["rgra_decision_mask"])
            diagnostics_by_key["level_rgra_active_ratios"].append(diagnostics["rgra_active_ratio"])
            rgra_logps.append(diagnostics["rgra_log_prob"] * level_keep[:, level_index])
            rgra_entropies.append(diagnostics["rgra_entropy"] * level_keep[:, level_index])
            rgra_values.append(diagnostics["rgra_value"] * level_keep[:, level_index])
            rags_logps.append(diagnostics["rags_log_prob"] * level_keep[:, level_index])
            rags_entropies.append(diagnostics["rags_entropy"] * level_keep[:, level_index])
            rags_values.append(diagnostics["rags_value"] * level_keep[:, level_index])

        fused_field, level_weights = self.cross_level_fusion(
            level_features, level_mask=level_keep
        )
        decoded = fused_field
        for upsample_block in self.upsample_blocks:
            decoded = upsample_block(decoded)
        if decoded.shape[-2:] != output_size:
            decoded = F.interpolate(decoded, size=output_size, mode="bilinear", align_corners=False)
        logits = self.classifier(decoded)

        active_levels = level_keep.sum(dim=1).clamp_min(1.0)
        rags_log_prob = torch.stack(rags_logps, dim=1).sum(dim=1) / active_levels
        rags_entropy = torch.stack(rags_entropies, dim=1).sum(dim=1) / active_levels
        rags_value = torch.stack(rags_values, dim=1).sum(dim=1) / active_levels
        rgra_log_prob = torch.stack(rgra_logps, dim=1).sum(dim=1) / active_levels
        rgra_entropy = torch.stack(rgra_entropies, dim=1).sum(dim=1) / active_levels
        rgra_value = torch.stack(rgra_values, dim=1).sum(dim=1) / active_levels

        auxiliary: Dict[str, Any] = {}
        if return_auxiliary:
            auxiliary = {
                "level_weights": level_weights,
                "fused_field": fused_field,
                "decoded_feature": decoded,
                "rags_policy_stats": {
                    "log_prob": rags_log_prob,
                    "entropy": rags_entropy,
                    "value": rags_value,
                },
                "rgra_policy_stats": {
                    "log_prob": rgra_log_prob,
                    "entropy": rgra_entropy,
                    "value": rgra_value,
                },
                **diagnostics_by_key,
            }
        return logits, auxiliary


class Paper4NGFC3DGSSeg(nn.Module):
    """DINOv3 RGB-D + NGFC + HRFR-v1 + RAGS + semantic 3DGS。"""

    DEFAULT_REPO_DIR = "/data/Lsy/sam/lsy/mymodelnew/toolbox/Mymodels/DINO/dinov3"
    DEFAULT_WEIGHTS_PATH = (
        "/data/Lsy/sam/lsy/mymodelnew/toolbox/Mymodels/DINO/"
        "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    )

    def __init__(
        self,
        num_classes: int,
        freeze_backbone: bool = True,
        field_dim: int = 256,
        semantic_dim: int = 128,
        decoder_dim: int = 192,
        backbone: Optional[nn.Module] = None,
        backbone_dim: int = 768,
        feature_layers: Sequence[int] = (0, 3, 7, 11),
        repo_dir: str = DEFAULT_REPO_DIR,
        weights_path: str = DEFAULT_WEIGHTS_PATH,
        backbone_name: str = "dinov3_vitb16",
        coordinate_bands: int = 6,
        num_gaussians_per_token: int = 2,
        gaussian_kernel_size: int = 5,
        minimum_screen_sigma: float = 0.35,
        maximum_screen_sigma: Optional[float] = None,
        rags_steps: int = 2,
        policy_hidden_dim: int = 128,
        use_rgra_agent: bool = False,
        rgra_hidden_dim: int = 32,
        rgra_refine_gain: float = 1.25,
    ) -> None:
        super().__init__()
        if len(feature_layers) < 1:
            raise ValueError("feature_layers 不能为空")

        if backbone is None:
            if not os.path.isdir(repo_dir):
                raise FileNotFoundError(
                    f"DINOv3 本地仓库不存在：{repo_dir}\n"
                    "请通过 repo_dir 指定正确路径，或通过 backbone 参数传入已构建的 backbone。"
                )
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(
                    f"DINOv3 权重不存在：{weights_path}\n"
                    "请通过 weights_path 指定正确路径，或通过 backbone 参数传入已构建的 backbone。"
                )
            backbone = torch.hub.load(
                repo_dir, backbone_name, source="local", weights=weights_path
            )
        if not hasattr(backbone, "get_intermediate_layers"):
            raise TypeError("backbone 必须实现 get_intermediate_layers 方法")

        self.backbone = backbone
        self.num_classes = num_classes
        self.backbone_dim = backbone_dim
        self.feature_layers = tuple(int(index) for index in feature_layers)
        self.freeze_backbone = bool(freeze_backbone)
        self.max_gaussians = int(num_gaussians_per_token)
        self.max_render_kernel = int(gaussian_kernel_size)
        self.use_rgra_agent = bool(use_rgra_agent)

        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()

        self.hrfr_policy = HRFRPolicy(
            backbone_dim=backbone_dim,
            num_levels=len(self.feature_layers),
            max_gaussians=num_gaussians_per_token,
            max_render_kernel=gaussian_kernel_size,
            hidden_dim=policy_hidden_dim,
        )
        self.ngfc = NeuralGeometryFieldConstruction(
            in_channels=backbone_dim,
            field_dim=field_dim,
            semantic_dim=semantic_dim,
            num_levels=len(self.feature_layers),
            coordinate_bands=coordinate_bands,
        )
        self.seg_head = NeuralFieldSegmentationHead(
            field_dim=field_dim,
            semantic_dim=semantic_dim,
            decoder_dim=decoder_dim,
            num_levels=len(self.feature_layers),
            num_classes=num_classes,
            num_upsample_stages=4,
            num_gaussians_per_token=num_gaussians_per_token,
            gaussian_kernel_size=gaussian_kernel_size,
            minimum_screen_sigma=minimum_screen_sigma,
            maximum_screen_sigma=maximum_screen_sigma,
            rags_steps=rags_steps,
        )
        # Attach the additional Agent last: existing HRFR/RAGS/NGFC weights retain
        # their original RNG initialization, useful for strict A/B ablation.
        if self.use_rgra_agent:
            for adapter in self.seg_head.level_adapters:
                adapter.gaussian_splatting.rgra_agent = GaussianRenderingAgent(
                    hidden_dim=rgra_hidden_dim,
                    refine_gain=rgra_refine_gain,
                )

    def train(self, mode: bool = True) -> "Paper4NGFC3DGSSeg":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def get_policy_parameters(self) -> List[nn.Parameter]:
        parameters: List[nn.Parameter] = list(self.hrfr_policy.parameters())
        for adapter in self.seg_head.level_adapters:
            gaussian = adapter.gaussian_splatting
            policy_modules = (
                gaussian.rags_policy_context,
                gaussian.rags_action_head,
                gaussian.rags_value_head,
            )
            for module in policy_modules:
                parameters.extend(module.parameters())
            if gaussian.rgra_agent is not None:
                parameters.extend(gaussian.rgra_agent.parameters())
        # 去重，避免共享参数时 optimizer 重复注册。
        unique: Dict[int, nn.Parameter] = {}
        for parameter in parameters:
            if parameter.requires_grad:
                unique[id(parameter)] = parameter
        return list(unique.values())

    def get_main_parameters(self) -> List[nn.Parameter]:
        policy_ids = {id(parameter) for parameter in self.get_policy_parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in policy_ids
        ]

    def evaluate_policy_rollout(self, rollout: Dict[str, Any]) -> Dict[str, Tensor]:
        """只重算 actor/critic heads 的 PPO evaluator，不重新执行 DINO/NGFC/3DGS。"""
        hrfr = self.hrfr_policy.evaluate_rollout(
            rollout["hrfr_policy_state"],
            rollout["hrfr_level_action"],
            rollout["hrfr_modality_action"],
            rollout["hrfr_budget_action"],
            rollout["hrfr_render_action"],
        )
        level_keep = rollout["level_keep"]
        rags_logps: List[Tensor] = []
        rags_entropies: List[Tensor] = []
        rags_values: List[Tensor] = []
        for level_index, adapter in enumerate(self.seg_head.level_adapters):
            states = rollout["rags_policy_states"][level_index]
            actions = rollout["rags_actions"][level_index]
            masks = rollout["rags_decision_masks"][level_index]
            if states.numel() == 0:
                zero = torch.zeros_like(hrfr["log_prob"])
                stats = {"log_prob": zero, "entropy": zero, "value": zero}
            else:
                stats = adapter.gaussian_splatting.evaluate_rags_rollout(
                    states, actions, masks
                )
            keep = level_keep[:, level_index].to(stats["log_prob"].dtype)
            rags_logps.append(stats["log_prob"] * keep)
            rags_entropies.append(stats["entropy"] * keep)
            rags_values.append(stats["value"] * keep)

        active_levels = level_keep.sum(dim=1).clamp_min(1.0)
        rags_log_prob = torch.stack(rags_logps, dim=1).sum(dim=1) / active_levels
        rags_entropy = torch.stack(rags_entropies, dim=1).sum(dim=1) / active_levels
        rags_value = torch.stack(rags_values, dim=1).sum(dim=1) / active_levels
        # Recover the third policy only if it was enabled in this rollout.
        if rollout.get("rgra_enabled", False):
            rgra_logps: List[Tensor] = []
            rgra_entropies: List[Tensor] = []
            rgra_values: List[Tensor] = []
            for level_index, adapter in enumerate(self.seg_head.level_adapters):
                state = rollout["rgra_policy_states"][level_index]
                action = rollout["rgra_actions"][level_index]
                mask = rollout["rgra_decision_masks"][level_index]
                stats = adapter.gaussian_splatting.rgra_agent.evaluate_rollout(
                    state, action, mask
                )
                keep = level_keep[:, level_index].to(stats["log_prob"].dtype)
                rgra_logps.append(stats["log_prob"] * keep)
                rgra_entropies.append(stats["entropy"] * keep)
                rgra_values.append(stats["value"] * keep)
            rgra_log_prob = torch.stack(rgra_logps, dim=1).sum(dim=1) / active_levels
            rgra_entropy = torch.stack(rgra_entropies, dim=1).sum(dim=1) / active_levels
            rgra_value = torch.stack(rgra_values, dim=1).sum(dim=1) / active_levels
            # Independent clipped PPO terms: each agent has its own old/new ratio.
            # Their scores are normalized per module to keep trajectory sizes manageable.
            components = [
                {"name": "hrfr", "log_prob": hrfr["log_prob"],
                 "old_log_prob": rollout["old_hrfr_log_prob"],
                 "entropy": hrfr["entropy"], "value": hrfr["value"]},
                {"name": "rags", "log_prob": rags_log_prob,
                 "old_log_prob": rollout["old_rags_log_prob"],
                 "entropy": rags_entropy, "value": rags_value},
                {"name": "rgra", "log_prob": rgra_log_prob,
                 "old_log_prob": rollout["old_rgra_log_prob"],
                 "entropy": rgra_entropy, "value": rgra_value},
            ]
            return {
                "log_prob": (hrfr["log_prob"] + rags_log_prob + rgra_log_prob) / 3.0,
                "old_log_prob": rollout["old_log_prob"],
                "entropy": (hrfr["entropy"] + rags_entropy + rgra_entropy) / 3.0,
                "value": (hrfr["value"] + rags_value + rgra_value) / 3.0,
                "rgra_log_prob": rgra_log_prob,
                "components": components,
            }
        return {
            "log_prob": 0.5 * (hrfr["log_prob"] + rags_log_prob),
            "old_log_prob": rollout["old_log_prob"],
            "entropy": 0.5 * (hrfr["entropy"] + rags_entropy),
            "value": 0.5 * (hrfr["value"] + rags_value),
        }

    @staticmethod
    def _prepare_depth_for_backbone(depth: Tensor) -> Tensor:
        if depth.ndim != 4:
            raise ValueError(f"depth 应为 BCHW，实际得到 {tuple(depth.shape)}")
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        if depth.shape[1] == 1:
            return depth.repeat(1, 3, 1, 1)
        if depth.shape[1] == 3:
            return depth
        raise ValueError(f"depth 通道数必须为 1 或 3，实际得到 {depth.shape[1]}")

    def _extract_backbone_features(self, image: Tensor) -> List[Tensor]:
        context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with context:
            features = self.backbone.get_intermediate_layers(
                image, n=list(self.feature_layers), reshape=True
            )
        features = [features] if isinstance(features, torch.Tensor) else list(features)
        if len(features) != len(self.feature_layers):
            raise RuntimeError("backbone 返回层数与 feature_layers 不一致")
        for level_index, feature in enumerate(features):
            if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
                raise RuntimeError(f"第 {level_index} 层 backbone 特征必须为 BCHW Tensor")
            if feature.shape[1] != self.backbone_dim:
                raise RuntimeError(
                    f"第 {level_index} 层通道数为 {feature.shape[1]}，backbone_dim={self.backbone_dim}"
                )
        return features

    @staticmethod
    def _route_modalities(
        rgb_features: Sequence[Tensor],
        depth_features: Sequence[Tensor],
        routing: Dict[str, Tensor],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        routed_rgb: List[Tensor] = []
        routed_depth: List[Tensor] = []
        for level_index, (rgb_feature, depth_feature) in enumerate(
            zip(rgb_features, depth_features)
        ):
            rgb_gate = routing["rgb_gate"][:, level_index].view(-1, 1, 1, 1)
            depth_gate = routing["depth_gate"][:, level_index].view(-1, 1, 1, 1)
            routed_rgb.append(rgb_feature * rgb_gate.to(rgb_feature.dtype))
            routed_depth.append(depth_feature * depth_gate.to(depth_feature.dtype))
        return routed_rgb, routed_depth

    def forward(
        self,
        x: Tensor,
        depth: Tensor,
        intrinsics: IntrinsicsType = None,
        geometry_depth: Optional[Tensor] = None,
        return_field_pyramid: bool = False,
        return_auxiliary: bool = False,
        squeeze_single_batch: bool = False,
        policy_mode: str = "warmup",
        use_rgra_agent: Optional[bool] = None,
    ) -> Union[Tensor, Dict[str, object]]:
        del squeeze_single_batch
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"x 应为 [B, 3, H, W]，实际得到 {tuple(x.shape)}")
        if depth.ndim != 4:
            raise ValueError(f"depth 应为 BCHW，实际得到 {tuple(depth.shape)}")
        if x.shape[0] != depth.shape[0] or x.shape[-2:] != depth.shape[-2:]:
            raise ValueError("RGB 与 Depth 的 batch 和空间尺寸必须一致")

        _, _, height, width = x.shape
        depth_for_backbone = self._prepare_depth_for_backbone(depth)
        rgra_enabled = self.use_rgra_agent if use_rgra_agent is None else bool(use_rgra_agent)
        if rgra_enabled and not self.use_rgra_agent:
            raise RuntimeError("请在 paper4 构造时设置 use_rgra_agent=True 再开启前向 RGRA")
        if geometry_depth is None:
            # 仅作为兼容兜底；正式训练应显式提供未 Normalize 的 geometry_depth。
            geometry_depth = depth[:, :1]
        if geometry_depth.shape[0] != x.shape[0] or geometry_depth.shape[-2:] != x.shape[-2:]:
            raise ValueError("geometry_depth 与 RGB 的 batch 和空间尺寸必须一致")

        rgb_features = self._extract_backbone_features(x)
        depth_features = self._extract_backbone_features(depth_for_backbone)
        routing = self.hrfr_policy(rgb_features, depth_features, policy_mode=policy_mode)
        routed_rgb, routed_depth = self._route_modalities(
            rgb_features, depth_features, routing
        )

        field_pyramid = self.ngfc(
            rgb_features=routed_rgb,
            depth_features=routed_depth,
            raw_depth=geometry_depth,
            intrinsics=intrinsics,
        )
        logits, auxiliary = self.seg_head(
            field_pyramid=field_pyramid,
            output_size=(height, width),
            intrinsics=intrinsics,
            routing=routing,
            policy_mode=policy_mode,
            return_auxiliary=return_auxiliary,
            use_rgra_agent=rgra_enabled,
        )

        if not return_field_pyramid and not return_auxiliary:
            return logits

        outputs: Dict[str, object] = {"logits": logits}
        if return_field_pyramid:
            outputs["field_pyramid"] = field_pyramid
        if return_auxiliary:
            rags_stats = auxiliary.get("rags_policy_stats", {})
            hrfr_logp = routing["log_prob"]
            hrfr_entropy = routing["entropy"]
            hrfr_value = routing["value"]
            rags_logp = rags_stats.get("log_prob", torch.zeros_like(hrfr_logp))
            rags_entropy = rags_stats.get("entropy", torch.zeros_like(hrfr_entropy))
            rags_value = rags_stats.get("value", torch.zeros_like(hrfr_value))
            use_third = rgra_enabled and policy_mode != "warmup"
            if use_third:
                rgra_stats = auxiliary["rgra_policy_stats"]
                merged_logp = (hrfr_logp + rags_logp + rgra_stats["log_prob"]) / 3.0
                merged_entropy = (hrfr_entropy + rags_entropy + rgra_stats["entropy"]) / 3.0
                merged_value = (hrfr_value + rags_value + rgra_stats["value"]) / 3.0
            else:
                merged_logp = 0.5 * (hrfr_logp + rags_logp)
                merged_entropy = 0.5 * (hrfr_entropy + rags_entropy)
                merged_value = 0.5 * (hrfr_value + rags_value)
            policy_stats = {
                "log_prob": merged_logp,
                "old_log_prob": merged_logp.detach(),
                "entropy": merged_entropy,
                "value": merged_value,
            }
            policy_rollout = None
            if policy_mode != "warmup":
                policy_rollout = {
                    "old_log_prob": policy_stats["old_log_prob"],
                    "hrfr_policy_state": routing["policy_state"].detach(),
                    "hrfr_level_action": routing["level_action"].detach(),
                    "hrfr_modality_action": routing["modality_action"].detach(),
                    "hrfr_budget_action": routing["budget_action"].detach(),
                    "hrfr_render_action": routing["render_action"].detach(),
                    "level_keep": routing["level_keep"].detach(),
                    "rags_policy_states": [x.detach() for x in auxiliary["level_rags_policy_states"]],
                    "rags_actions": [x.detach() for x in auxiliary["level_rags_actions"]],
                    "rags_decision_masks": [x.detach() for x in auxiliary["level_rags_decision_masks"]],
                    "rgra_enabled": bool(use_third),
                }
                if use_third:
                    rgra_stats = auxiliary["rgra_policy_stats"]
                    policy_rollout.update({
                        "old_hrfr_log_prob": hrfr_logp.detach(),
                        "old_rags_log_prob": rags_logp.detach(),
                        "old_rgra_log_prob": rgra_stats["log_prob"].detach(),
                    })
                    policy_rollout.update({
                        "rgra_policy_states": [x.detach() for x in auxiliary["level_rgra_policy_states"]],
                        "rgra_actions": [x.detach() for x in auxiliary["level_rgra_actions"]],
                        "rgra_decision_masks": [x.detach() for x in auxiliary["level_rgra_decision_masks"]],
                    })
            outputs.update(auxiliary)
            outputs.update(
                {
                    "policy_stats": policy_stats,
                    "policy_rollout": policy_rollout,
                    "hrfr_level_keep": routing["level_keep"],
                    "hrfr_modality_action": routing["modality_action"],
                    "hrfr_gaussian_budget": routing["gaussian_budget"],
                    "hrfr_render_kernel": routing["render_kernel"],
                    "rgra_enabled": rgra_enabled,
                }
            )
        return outputs


# 保持旧类名与训练入口兼容。
Paper4NGFCSeg = Paper4NGFC3DGSSeg


class paper4(Paper4NGFC3DGSSeg):
    pass


class _MockDINOBackbone(nn.Module):
    """仅用于本文件自检，不参与正式训练。"""

    def __init__(self, embed_dim: int = 768, num_blocks: int = 12) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
                    LayerNorm2d(embed_dim),
                    nn.GELU(),
                )
                for _ in range(num_blocks)
            ]
        )

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: Sequence[int],
        reshape: bool = True,
    ) -> List[Tensor]:
        del reshape
        feature = self.patch_embed(x)
        outputs: List[Tensor] = []
        requested = set(int(index) for index in n)
        for index, block in enumerate(self.blocks):
            feature = feature + block(feature)
            if index in requested:
                outputs.append(feature)
        if len(outputs) != len(n):
            raise RuntimeError("Mock backbone 的层索引超出范围")
        return outputs


def _self_test() -> None:
    torch.manual_seed(7)
    use_real_backbone = (
        os.path.isdir(Paper4NGFC3DGSSeg.DEFAULT_REPO_DIR)
        and os.path.isfile(Paper4NGFC3DGSSeg.DEFAULT_WEIGHTS_PATH)
    )

    if use_real_backbone:
        model = paper4(num_classes=41, freeze_backbone=True)
        test_size = (224, 320)
        print("使用本地 DINOv3 执行自检。")
    else:
        model = paper4(
            num_classes=41,
            freeze_backbone=False,
            backbone=_MockDINOBackbone(embed_dim=768),
            backbone_dim=768,
            field_dim=64,
            semantic_dim=32,
            decoder_dim=64,
        )
        test_size = (128, 160)
        print("未发现本地 DINOv3，使用 Mock backbone 验证完整前向链路。")

    model.eval()
    rgb = torch.randn(1, 3, *test_size)
    raw_depth = torch.rand(1, 1, *test_size) + 0.1

    with torch.inference_mode():
        output = model(
            rgb,
            raw_depth,
            return_field_pyramid=True,
            return_auxiliary=True,
            policy_mode="warmup",
        )

    logits = output["logits"]
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (1, 41, *test_size), logits.shape
    assert torch.isfinite(logits).all(), "logits 中存在 NaN/Inf"

    total_params = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ) / 1e6

    print(f"输入 RGB:   {tuple(rgb.shape)}")
    print(f"输入 Depth: {tuple(raw_depth.shape)}")
    print(f"输出 logits:{tuple(logits.shape)}")
    print(f"总参数量:   {total_params:.2f}M")
    print(f"可训练参数: {trainable_params:.2f}M")
    print("NGFC + adaptive semantic 3DGS segmentation test passed.")


if __name__ == "__main__":
    _self_test()
