# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def aligned_iou_loss(boxes1, boxes2, loss_type="eiou", eps=1e-7):
    """
    IoU-family losses for matched xyxy boxes.

    DIoU adds center-distance regularization; CIoU adds aspect-ratio
    regularization; EIoU separately regularizes center, width, and height.
    EIoU is the default transition loss for small fracture boxes because the
    width/height terms are less entangled than GIoU's enclosing-area penalty.
    """
    assert boxes1.shape == boxes2.shape
    assert boxes1.shape[-1] == 4

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    w1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=eps)
    h1 = (boxes1[:, 3] - boxes1[:, 1]).clamp(min=eps)
    w2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=eps)
    h2 = (boxes2[:, 3] - boxes2[:, 1]).clamp(min=eps)
    area1 = w1 * h1
    area2 = w2 * h2
    union = (area1 + area2 - inter).clamp(min=eps)
    iou = inter / union

    if loss_type == "iou":
        return 1 - iou

    c_lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    c_rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    c_wh = (c_rb - c_lt).clamp(min=eps)
    c2 = (c_wh[:, 0] ** 2 + c_wh[:, 1] ** 2).clamp(min=eps)

    b1_center = (boxes1[:, :2] + boxes1[:, 2:]) * 0.5
    b2_center = (boxes2[:, :2] + boxes2[:, 2:]) * 0.5
    rho2 = ((b1_center - b2_center) ** 2).sum(dim=1)
    diou = 1 - iou + rho2 / c2

    if loss_type == "diou":
        return diou

    if loss_type == "ciou":
        v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)
        return diou + alpha * v

    if loss_type == "eiou":
        cw2 = (c_wh[:, 0] ** 2).clamp(min=eps)
        ch2 = (c_wh[:, 1] ** 2).clamp(min=eps)
        return diou + (w1 - w2) ** 2 / cw2 + (h1 - h2) ** 2 / ch2

    raise ValueError(f"Unsupported IoU loss type: {loss_type}")


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
