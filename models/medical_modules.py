import torch
import torch.nn.functional as F
from torch import nn


SAMPLE_TYPE_TO_ID = {
    "P": 0,
    "P-slice": 0,
    "PN": 1,
    "PN-slice": 1,
    "HN": 2,
    "HN-slice": 2,
    "N": 3,
    "N-slice": 3,
}


class ChannelWiseSliceEmbedding(nn.Module):
    """Adds learnable {-1, 0, +1} slice-position offsets to 2.5D input channels."""

    def __init__(self, init_scale=0.02):
        super().__init__()
        self.slice_embedding = nn.Parameter(
            torch.tensor([-1.0, 0.0, 1.0]).view(1, 3, 1, 1) * init_scale
        )

    def forward(self, x):
        if x.shape[1] != 3:
            raise ValueError("2.5D slice embedding expects exactly 3 input channels")
        return x + self.slice_embedding.to(dtype=x.dtype, device=x.device)


class EdgeGuidedSpatialAttention(nn.Module):
    """Sobel edge prior that boosts cortical edges and high-frequency fracture cues."""

    def __init__(self, strength=0.25):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.strength = strength

    def edge_prior(self, samples, organ_mask=None):
        center = samples[:, 1:2]
        gx = F.conv2d(center, self.sobel_x.to(center), padding=1)
        gy = F.conv2d(center, self.sobel_y.to(center), padding=1)
        edge = torch.sqrt(gx.square() + gy.square() + 1e-6)
        edge = edge / edge.flatten(1).amax(dim=1).clamp(min=1e-6).view(-1, 1, 1, 1)
        if organ_mask is not None:
            if organ_mask.shape[1] >= 3:
                organ_mask = organ_mask[:, 1:2]
            edge = edge * F.interpolate(organ_mask.to(device=edge.device, dtype=edge.dtype),
                                        size=edge.shape[-2:], mode="nearest")
        return edge.clamp(0, 1)

    def forward(self, features, samples, organ_mask=None):
        edge = self.edge_prior(samples, organ_mask=organ_mask)
        enhanced = []
        for feature in features:
            prior = F.interpolate(edge, size=feature.shape[-2:], mode="bilinear", align_corners=False)
            enhanced.append(feature * (1.0 + self.strength * prior))
        return enhanced, edge


def normalize_organ_mask(organ_mask, batch_size, height, width, device, dtype, channels=3):
    if organ_mask is None:
        return torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)
    if organ_mask.ndim == 3:
        organ_mask = organ_mask.unsqueeze(1)
    organ_mask = organ_mask.to(device=device, dtype=dtype)
    if organ_mask.shape[-2:] != (height, width):
        organ_mask = F.interpolate(organ_mask, size=(height, width), mode="nearest")
    if organ_mask.shape[1] == 1 and channels > 1:
        organ_mask = organ_mask.expand(-1, channels, -1, -1)
    elif organ_mask.shape[1] != channels:
        raise ValueError(
            f"Organ mask must have 1 or {channels} channels, got shape {tuple(organ_mask.shape)}"
        )
    return (organ_mask > 0).to(dtype=dtype)


def dilate_organ_mask(organ_mask, pixels=5):
    if organ_mask is None or pixels <= 0:
        return organ_mask
    if organ_mask.ndim == 3:
        organ_mask = organ_mask.unsqueeze(1)
    kernel = int(pixels) * 2 + 1
    return F.max_pool2d(organ_mask.float(), kernel_size=kernel, stride=1, padding=int(pixels)).clamp(0, 1)


class OrganFeatureGating(nn.Module):
    """Patient-specific soft ROI gating for P2/P3/P4 features."""

    def __init__(self, strength=1.0, outside_scale=0.25, dilation=5):
        super().__init__()
        self.strength = strength
        self.outside_scale = outside_scale
        self.dilation = dilation

    def forward(self, features, organ_mask):
        if organ_mask is None:
            return features, []
        if organ_mask.shape[1] >= 3:
            organ_mask = organ_mask[:, 1:2]
        organ_mask = dilate_organ_mask(organ_mask, self.dilation)
        gated = []
        priors = []
        for feature in features:
            prior = F.interpolate(organ_mask.to(device=feature.device, dtype=feature.dtype),
                                  size=feature.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
            gate = self.outside_scale + (1.0 - self.outside_scale) * prior
            gated.append(feature * gate * (1.0 + self.strength * prior))
            priors.append(prior)
        return gated, priors


class FractureHeatmapBranch(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, feature):
        return self.head(feature)


class AdjacentSliceFeatureMemory(nn.Module):
    """Fuse query features with spatial features sampled from explicit t-1/t/t+1 slices."""

    def __init__(self, hidden_dim, alpha=1.0, beta=0.5, gamma=0.5, mid_dim=64):
        super().__init__()
        self.slice_encoder = nn.Sequential(
            nn.Conv2d(1, mid_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, mid_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, mid_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.register_buffer(
            "slice_offsets",
            torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32).view(1, 3, 1, 1, 1),
        )
        self.slice_position_direction = nn.Parameter(torch.empty(1, 1, hidden_dim, 1, 1))
        nn.init.normal_(self.slice_position_direction, std=0.02)
        self.query_gate = nn.Linear(hidden_dim * 2, 3)
        self.memory_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.base_weights = nn.Parameter(torch.tensor([beta, alpha, gamma], dtype=torch.float32))

    def forward(self, queries, reference_points, samples, organ_mask=None, feature_size=None):
        if samples is None or samples.shape[1] != 3 or reference_points is None:
            zero = queries.sum() * 0.0
            return queries, {"slice_memory_consistency": zero}

        bs, num_queries, hidden_dim = queries.shape
        _, _, height, width = samples.shape
        encoded_content = self.slice_encoder(samples.reshape(bs * 3, 1, height, width))
        if feature_size is not None and encoded_content.shape[-2:] != tuple(feature_size):
            encoded_content = F.interpolate(encoded_content, size=feature_size, mode="bilinear", align_corners=False)
        encoded_content = encoded_content.view(bs, 3, hidden_dim, encoded_content.shape[-2], encoded_content.shape[-1])

        if organ_mask is not None:
            organ_mask = normalize_organ_mask(
                organ_mask, bs, height, width, samples.device, samples.dtype, channels=3
            )
            mask = organ_mask.reshape(bs * 3, 1, height, width)
            mask = F.interpolate(mask, size=encoded_content.shape[-2:], mode="nearest").view(
                bs, 3, 1, *encoded_content.shape[-2:]
            )
            encoded_content = encoded_content * (0.2 + 0.8 * mask)

        encoded = encoded_content + self.slice_offsets.to(dtype=encoded_content.dtype, device=encoded_content.device) * \
            self.slice_position_direction.to(dtype=encoded_content.dtype, device=encoded_content.device)

        grid = reference_points.clamp(0, 1).mul(2).sub(1).view(bs, num_queries, 1, 2)
        sampled_slices = []
        sampled_content_slices = []
        for slice_id in range(3):
            sampled = F.grid_sample(
                encoded[:, slice_id], grid, mode="bilinear",
                padding_mode="zeros", align_corners=False
            )[:, :, :, 0].transpose(1, 2)
            sampled_slices.append(sampled)
            sampled_content = F.grid_sample(
                encoded_content[:, slice_id], grid, mode="bilinear",
                padding_mode="zeros", align_corners=False
            )[:, :, :, 0].transpose(1, 2)
            sampled_content_slices.append(sampled_content)
        sampled = torch.stack(sampled_slices, dim=2)  # [B, Q, 3, C]
        sampled_content = torch.stack(sampled_content_slices, dim=2)

        center_features = sampled[:, :, 1]
        logits = self.query_gate(torch.cat([queries, center_features], dim=-1))
        logits = logits + self.base_weights.view(1, 1, 3)
        weights = torch.softmax(logits, dim=-1)
        memory = (sampled * weights.unsqueeze(-1)).sum(dim=2)
        fused = self.norm(queries + self.memory_proj(memory))

        center_content = sampled_content[:, :, 1]
        consistency = (
            F.smooth_l1_loss(sampled_content[:, :, 0], center_content, reduction="none").mean(dim=-1)
            + F.smooth_l1_loss(sampled_content[:, :, 2], center_content, reduction="none").mean(dim=-1)
        ) * 0.5
        return fused, {
            "slice_memory_consistency": consistency.mean(),
            "slice_memory_weights": weights.detach().mean(dim=(0, 1)),
        }


class RegionPriorEmbedding(nn.Module):
    """Embeds per-sample organ-mask prior values for dynamic query initialization."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.LayerNorm(hidden_dim))

    def forward(self, values):
        return self.proj(values.unsqueeze(-1))


def build_box_center_heatmap(targets, size, device):
    heatmap = torch.zeros((len(targets), 1, size[0], size[1]), device=device)
    for batch_idx, target in enumerate(targets):
        boxes = target.get("boxes")
        if boxes is None or len(boxes) == 0:
            continue
        centers = boxes[:, :2].to(device)
        xs = (centers[:, 0] * size[1]).long().clamp(0, size[1] - 1)
        ys = (centers[:, 1] * size[0]).long().clamp(0, size[0] - 1)
        heatmap[batch_idx, 0, ys, xs] = 1.0
    return heatmap
