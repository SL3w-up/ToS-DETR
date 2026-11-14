from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


SAMPLE_TYPE_ORDER = ("P", "PN", "N")
SAMPLE_TYPE_TO_ID = {name: idx for idx, name in enumerate(SAMPLE_TYPE_ORDER)}
UNKNOWN_SAMPLE_TYPE = "UNKNOWN"
UNKNOWN_SAMPLE_TYPE_ID = -1


def normalize_sample_type(value, annotations: Optional[Sequence[dict]] = None) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        normalized = value.strip().upper()
        normalized = normalized.replace("_", "-")
        if normalized.endswith("-SLICE"):
            normalized = normalized[:-6]
        if normalized in SAMPLE_TYPE_TO_ID:
            return normalized
    return UNKNOWN_SAMPLE_TYPE


def sample_type_to_id(sample_type: Optional[str]) -> int:
    if sample_type is None:
        return UNKNOWN_SAMPLE_TYPE_ID
    return SAMPLE_TYPE_TO_ID.get(normalize_sample_type(sample_type), UNKNOWN_SAMPLE_TYPE_ID)


def boxes_to_centers(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 2))
    return boxes[..., :2]


def build_heatmap_targets(
    targets: List[dict],
    size: Tuple[int, int],
    device: torch.device,
    sigma: float = 2.0,
) -> torch.Tensor:
    height, width = size
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    heatmaps = []
    for target in targets:
        heatmap = torch.zeros((height, width), device=device, dtype=torch.float32)
        boxes = target.get("boxes")
        if boxes is None or boxes.numel() == 0:
            heatmaps.append(heatmap)
            continue
        centers = boxes_to_centers(boxes.to(device))
        for center in centers:
            cx = center[0].clamp(0, 1) * max(width - 1, 1)
            cy = center[1].clamp(0, 1) * max(height - 1, 1)
            dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
            heatmap = torch.maximum(heatmap, torch.exp(-dist2 / (2 * sigma ** 2)))
        heatmaps.append(heatmap)
    return torch.stack(heatmaps, dim=0).unsqueeze(1)


def sample_map_at_points(feature_map: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    if points.numel() == 0:
        return feature_map.new_zeros(points.shape[:-1])
    grid = points.clamp(0, 1) * 2 - 1
    grid = grid.unsqueeze(2)
    sampled = F.grid_sample(feature_map, grid, align_corners=False, mode="bilinear")
    return sampled.squeeze(1).squeeze(-1)


def move_target_to_device(value, device, non_blocking: bool = False):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {key: move_target_to_device(item, device, non_blocking=non_blocking)
                for key, item in value.items()}
    if isinstance(value, list):
        return [move_target_to_device(item, device, non_blocking=non_blocking) for item in value]
    if isinstance(value, tuple):
        return tuple(move_target_to_device(item, device, non_blocking=non_blocking) for item in value)
    return value
