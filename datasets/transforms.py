# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torch.nn.functional as torch_F
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util import box_ops
from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def _as_chw_mask(mask):
    if isinstance(mask, torch.Tensor):
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3 and mask.shape[0] not in (1, 3) and mask.shape[-1] in (1, 3):
            mask = mask.permute(2, 0, 1)
        mask = (mask.float() > 0).float()
        if mask.shape[0] == 1:
            mask = mask.expand(3, -1, -1)
        elif mask.shape[0] != 3:
            raise ValueError(f"Organ mask must have 1 or 3 channels, got shape {tuple(mask.shape)}")
        return mask
    return mask


def _filter_target_by_keep(target, keep):
    fields = ["boxes", "labels", "area", "iscrowd", "masks"]
    target = target.copy()
    for field in fields:
        if field in target and isinstance(target[field], torch.Tensor) and target[field].shape[0] == keep.shape[0]:
            target[field] = target[field][keep]
    return target


def _mask_to_tensor(mask):
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        return _as_chw_mask(mask)
    return (F.to_tensor(mask) > 0).float()


def _resize_mask(mask, size):
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        mask = _as_chw_mask(mask)
        return (interpolate(mask[None].float(), size, mode="nearest")[0] > 0.5).float()
    return F.resize(mask, size, interpolation=T.InterpolationMode.NEAREST)


def _crop_mask(mask, region):
    if mask is None:
        return None
    return F.crop(mask, *region)


def _hflip_mask(mask):
    if mask is None:
        return None
    return F.hflip(mask)


def _pad_mask(mask, padding):
    if mask is None:
        return None
    return F.pad(mask, padding)


def _rotate_mask(mask, angle):
    if mask is None:
        return None
    return F.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST)


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    if "organ_mask" in target:
        target["organ_mask"] = _crop_mask(target["organ_mask"], region)

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    if "organ_mask" in target:
        target["organ_mask"] = _hflip_mask(target["organ_mask"])

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    if "organ_mask" in target:
        target["organ_mask"] = _resize_mask(target["organ_mask"], size)

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    if "organ_mask" in target:
        target["organ_mask"] = _pad_mask(target["organ_mask"], (0, 0, padding[0], padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        if target is not None and "organ_mask" in target:
            target = target.copy()
            target["organ_mask"] = _mask_to_tensor(target["organ_mask"])
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


def _rotate_points(points: torch.Tensor, center: torch.Tensor, angle_rad: float) -> torch.Tensor:
    angle_tensor = points.new_tensor(angle_rad)
    cos_theta = torch.cos(angle_tensor)
    sin_theta = torch.sin(angle_tensor)
    rotation = torch.stack([
        torch.stack([cos_theta, -sin_theta]),
        torch.stack([sin_theta, cos_theta]),
    ])
    return (points - center) @ rotation.T + center


class RandomRotationBoxSafe(object):
    def __init__(self, degrees: float = 5.0, p: float = 0.3):
        self.degrees = degrees
        self.p = p

    def __call__(self, img, target):
        if random.random() >= self.p or self.degrees <= 0:
            return img, target
        angle = random.uniform(-self.degrees, self.degrees)
        rotated = F.rotate(img, angle, interpolation=T.InterpolationMode.BILINEAR)
        if target is None:
            return rotated, target

        target = target.copy()
        if "organ_mask" in target:
            target["organ_mask"] = _rotate_mask(target["organ_mask"], angle)
        if "boxes" not in target or target["boxes"].numel() == 0:
            return rotated, target

        width, height = img.size
        center = target["boxes"].new_tensor([width / 2.0, height / 2.0])
        angle_rad = angle * torch.pi / 180.0
        boxes = target["boxes"]
        corners = torch.stack([
            boxes[:, [0, 1]],
            boxes[:, [2, 1]],
            boxes[:, [2, 3]],
            boxes[:, [0, 3]],
        ], dim=1)
        rotated_corners = _rotate_points(corners.reshape(-1, 2), center, angle_rad).reshape(-1, 4, 2)
        min_xy = rotated_corners.min(dim=1).values
        max_xy = rotated_corners.max(dim=1).values
        rotated_boxes = torch.cat([min_xy, max_xy], dim=1)
        rotated_boxes[:, 0::2].clamp_(0, width)
        rotated_boxes[:, 1::2].clamp_(0, height)
        target["boxes"] = rotated_boxes
        if "area" in target:
            wh = (rotated_boxes[:, 2:] - rotated_boxes[:, :2]).clamp(min=0)
            target["area"] = wh[:, 0] * wh[:, 1]
        keep = (rotated_boxes[:, 2] > rotated_boxes[:, 0]) & (rotated_boxes[:, 3] > rotated_boxes[:, 1])
        if not keep.all():
            target = _filter_target_by_keep(target, keep)
        return rotated, target


class SafeElasticDeformation(object):
    def __init__(self, p: float = 0.0, alpha: float = 2.0, sigma: float = 6.0, safe_mode: bool = True):
        self.p = p
        self.alpha = alpha
        self.sigma = sigma
        self.safe_mode = safe_mode

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        if not isinstance(image, torch.Tensor):
            return image, target
        if self.safe_mode and target is not None and target.get("boxes") is not None and target["boxes"].numel() > 0:
            return image, target

        _, height, width = image.shape
        device = image.device
        noise = torch.randn(1, 2, height, width, device=device, dtype=image.dtype)
        kernel_size = max(int(self.sigma) * 2 + 1, 3)
        displacement = torch_F.avg_pool2d(noise, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        displacement = displacement * (self.alpha / max(height, width))
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=image.dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=image.dtype),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        grid = grid + displacement.permute(0, 2, 3, 1)
        deformed = torch_F.grid_sample(image.unsqueeze(0), grid, align_corners=False, padding_mode="border").squeeze(0)
        if target is not None and "organ_mask" in target:
            target = target.copy()
            organ_mask = _as_chw_mask(target["organ_mask"]).to(device=device, dtype=image.dtype)
            target["organ_mask"] = torch_F.grid_sample(
                organ_mask.unsqueeze(0), grid, align_corners=False,
                padding_mode="zeros", mode="nearest"
            ).squeeze(0).clamp(0, 1)
        return deformed, target


class ROICopyPaste(object):
    def __init__(self, p: float = 0.15, max_iou: float = 0.1, max_translate: float = 0.08,
                 min_organ_overlap: float = 0.5):
        self.p = p
        self.max_iou = max_iou
        self.max_translate = max_translate
        self.min_organ_overlap = min_organ_overlap

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        if not isinstance(image, torch.Tensor) or target is None or "boxes" not in target or target["boxes"].numel() == 0:
            return image, target

        boxes = target["boxes"]
        index = random.randrange(boxes.shape[0])
        box = boxes[index]
        _, height, width = image.shape
        x1, y1, x2, y2 = box.round().long().tolist()
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, width)
        y2 = min(y2, height)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return image, target

        patch = image[:, y1:y2, x1:x2].clone()
        dx = int(random.uniform(-self.max_translate, self.max_translate) * width)
        dy = int(random.uniform(-self.max_translate, self.max_translate) * height)
        new_x1 = max(min(x1 + dx, width - (x2 - x1)), 0)
        new_y1 = max(min(y1 + dy, height - (y2 - y1)), 0)
        new_x2 = new_x1 + (x2 - x1)
        new_y2 = new_y1 + (y2 - y1)
        new_box = box.new_tensor([[new_x1, new_y1, new_x2, new_y2]])
        if boxes.numel() > 0:
            ious = box_ops.box_iou(new_box, boxes)[0]
            if ious.max().item() > self.max_iou:
                return image, target
        if "organ_mask" in target and self.min_organ_overlap > 0:
            organ_mask = _as_chw_mask(target["organ_mask"])
            center_mask = organ_mask[1:2] if organ_mask.shape[0] >= 3 else organ_mask[:1]
            region = center_mask[:, new_y1:new_y2, new_x1:new_x2]
            if region.numel() == 0 or float(region.float().mean().item()) < self.min_organ_overlap:
                return image, target

        image = image.clone()
        image[:, new_y1:new_y2, new_x1:new_x2] = patch
        target = target.copy()
        target["boxes"] = torch.cat([boxes, new_box.to(boxes.dtype)], dim=0)
        if "labels" in target:
            target["labels"] = torch.cat([target["labels"], target["labels"][index:index + 1]], dim=0)
        if "area" in target:
            area = (new_box[:, 2] - new_box[:, 0]) * (new_box[:, 3] - new_box[:, 1])
            target["area"] = torch.cat([target["area"], area.to(target["area"].dtype)], dim=0)
        if "iscrowd" in target:
            target["iscrowd"] = torch.cat([target["iscrowd"], target["iscrowd"][index:index + 1]], dim=0)
        if "masks" in target:
            new_mask = torch.zeros_like(target["masks"][index:index + 1])
            new_mask[:, new_y1:new_y2, new_x1:new_x2] = target["masks"][index:index + 1, y1:y2, x1:x2]
            target["masks"] = torch.cat([target["masks"], new_mask], dim=0)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
