# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.fracture_ops import normalize_sample_type, sample_type_to_id
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False,
                 local_rank=0, local_size=1, single_class=True, pn_slice_window=5,
                 organ_mask_root='auto', organ_mask_suffixes=None,
                 organ_mask_extensions=None, organ_mask_missing_as_ones=True,
                 organ_mask_recursive=False, strict_organ_masks=False,
                 organ_mask_missing_warning_limit=10, require_case_metadata=True,
                 validate_organ_masks=False, organ_mask_validate_max_samples=2048):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, single_class=single_class)
        self.img_folder = Path(img_folder)
        self.organ_mask_root = self._resolve_organ_mask_root(organ_mask_root)
        self.organ_mask_suffixes = organ_mask_suffixes or ["", "_mask", "_seg", "-mask", "-seg"]
        self.organ_mask_extensions = organ_mask_extensions or [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy", ".pt", ".pth"]
        self.organ_mask_missing_as_ones = organ_mask_missing_as_ones
        self.strict_organ_masks = strict_organ_masks
        self.organ_mask_missing_warning_limit = organ_mask_missing_warning_limit
        self.organ_mask_index = self._build_organ_mask_index() if organ_mask_recursive else {}
        self.organ_mask_missing_count = 0
        self.require_case_metadata = require_case_metadata
        self.organ_mask_validate_max_samples = max(int(organ_mask_validate_max_samples), 0)
        self._validate_case_metadata()
        if validate_organ_masks:
            self._validate_organ_masks_exist()
        self.sample_types = []
        self.case_has_positive = defaultdict(bool)
        positive_slices_by_case = {}
        for image_id in self.ids:
            image_info = self.coco.imgs.get(image_id, {})
            annotations = self.coco.imgToAnns.get(image_id, [])
            has_valid_boxes = self._has_valid_boxes(image_info, annotations)
            if not has_valid_boxes:
                continue
            case_id = image_info.get("case_id")
            if case_id is not None:
                self.case_has_positive[str(case_id)] = True
            slice_index = image_info.get("slice_index")
            if case_id is not None and slice_index is not None:
                positive_slices_by_case.setdefault(case_id, set()).add(int(slice_index))

        for image_id in self.ids:
            image_info = self.coco.imgs.get(image_id, {})
            annotations = self.coco.imgToAnns.get(image_id, [])
            has_boxes = self._has_valid_boxes(image_info, annotations)
            sample_type = normalize_sample_type(
                image_info.get("sample_type") or image_info.get("sampleType") or image_info.get("type"),
            )
            if has_boxes:
                sample_type = "P"
            elif sample_type == "P":
                sample_type = "UNKNOWN"
            if not has_boxes and sample_type == "UNKNOWN":
                case_id = image_info.get("case_id")
                slice_index = image_info.get("slice_index")
                positive_slices = positive_slices_by_case.get(case_id, set())
                is_near_positive = (
                    pn_slice_window > 0
                    and slice_index is not None
                    and any(abs(int(slice_index) - pos_slice) <= pn_slice_window for pos_slice in positive_slices)
                )
                sample_type = "PN" if is_near_positive else "N"
            self.sample_types.append(sample_type)
        self.has_sample_types = len(self.sample_types) > 0
        self.case_to_indices = defaultdict(list)
        self.case_slice_to_index = {}
        for idx, image_id in enumerate(self.ids):
            image_info = self.coco.imgs.get(image_id, {})
            case_id = image_info.get("case_id")
            slice_index = image_info.get("slice_index")
            if case_id is not None:
                self.case_to_indices[str(case_id)].append(idx)
                if slice_index is not None:
                    self.case_slice_to_index[(str(case_id), int(slice_index))] = idx

    @staticmethod
    def _has_valid_boxes(image_info, annotations):
        width = float(image_info.get("width", 0) or 0)
        height = float(image_info.get("height", 0) or 0)
        for obj in annotations:
            if obj.get("iscrowd", 0):
                continue
            bbox = obj.get("bbox")
            if bbox is None or len(bbox) < 4:
                continue
            x0, y0, bw, bh = [float(v) for v in bbox[:4]]
            x1 = x0 + bw
            y1 = y0 + bh
            if width > 0:
                x0 = min(max(x0, 0.0), width)
                x1 = min(max(x1, 0.0), width)
            if height > 0:
                y0 = min(max(y0, 0.0), height)
                y1 = min(max(y1, 0.0), height)
            if x1 > x0 and y1 > y0:
                return True
        return False

    def _validate_case_metadata(self):
        missing_case = []
        missing_slice = []
        for image_id in self.ids:
            image_info = self.coco.imgs.get(image_id, {})
            if "case_id" not in image_info:
                missing_case.append(image_info.get("file_name", str(image_id)))
            if "slice_index" not in image_info:
                missing_slice.append(image_info.get("file_name", str(image_id)))
        if not missing_case and not missing_slice:
            return
        message = (
            f"COCO medical metadata is incomplete for {self.img_folder}: "
            f"missing case_id={len(missing_case)}, missing slice_index={len(missing_slice)}. "
            f"Examples case_id={missing_case[:5]}, slice_index={missing_slice[:5]}"
        )
        if self.require_case_metadata:
            raise ValueError(message)
        print(f"Warning: {message}; falling back to per-image metadata where needed.")

    def _validate_organ_masks_exist(self):
        if self.organ_mask_root is None:
            raise FileNotFoundError(
                "strict organ-mask validation is enabled, but organ_mask_root resolved to None"
            )
        if not self.organ_mask_root.exists():
            raise FileNotFoundError(
                f"strict organ-mask validation is enabled, but organ_mask_root does not exist: {self.organ_mask_root}"
            )
        missing = []
        ids_to_check = list(self.ids)
        if self.organ_mask_validate_max_samples > 0:
            ids_to_check = ids_to_check[:self.organ_mask_validate_max_samples]
        for image_id in ids_to_check:
            image_info = self.coco.imgs.get(image_id, {})
            if not any(path.exists() and path.is_file() for path in self._candidate_mask_paths(image_info)):
                missing.append(image_info.get("file_name", str(image_id)))
                if len(missing) >= self.organ_mask_missing_warning_limit:
                    break
        if missing:
            raise FileNotFoundError(
                f"Missing organ masks under {self.organ_mask_root} for {self.img_folder}. "
                f"First missing examples: {missing}"
            )

    def _resolve_organ_mask_root(self, organ_mask_root):
        if organ_mask_root is None:
            return None
        if isinstance(organ_mask_root, str) and organ_mask_root.lower() in ("", "none", "off", "false"):
            return None
        if organ_mask_root != "auto":
            return Path(organ_mask_root)

        coco_root = self.img_folder.parent
        candidates = [
            coco_root / "anatomy_masks",
            coco_root / "organ_masks",
            coco_root / "masks",
            coco_root.parent / f"{coco_root.parent.name}-mask",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _build_organ_mask_index(self):
        if self.organ_mask_root is None or not self.organ_mask_root.exists():
            return {}
        index = {}
        duplicate_stems = set()
        for path in self.organ_mask_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in set(self.organ_mask_extensions):
                if path.stem in duplicate_stems:
                    continue
                if path.stem in index and index[path.stem] != path:
                    duplicate_stems.add(path.stem)
                    index.pop(path.stem, None)
                    continue
                index[path.stem] = path
        return index

    def _candidate_mask_paths(self, image_info):
        field_names = (
            "organ_mask", "organ_mask_path", "organ_mask_file", "organ_mask_file_name",
            "anatomy_mask", "anatomy_mask_path", "mask_path", "mask_file_name",
        )
        roots = [self.organ_mask_root] if self.organ_mask_root is not None else []
        candidates = []
        for field in field_names:
            value = image_info.get(field)
            if not value:
                continue
            path = Path(value)
            if path.is_absolute():
                candidates.append(path)
            for root in roots:
                candidates.append(root / path)

        file_name = image_info.get("file_name")
        if not file_name:
            return candidates
        image_path = Path(file_name)
        stem = image_path.stem
        split = self.img_folder.name
        stems = {stem}
        case_id = image_info.get("case_id")
        slice_index = image_info.get("slice_index")
        num_slices = image_info.get("num_slices")
        if case_id is not None and slice_index is not None:
            stems.update({
                f"{case_id}_{int(slice_index)}",
                f"{case_id}_seg_{int(slice_index)}",
                f"{case_id}_mask_{int(slice_index)}",
            })
            if num_slices is not None:
                stems.update({
                    f"{case_id}_{int(slice_index)}_{int(num_slices)}",
                    f"{case_id}_seg_{int(slice_index)}_{int(num_slices)}",
                    f"{case_id}_mask_{int(slice_index)}_{int(num_slices)}",
                })

        for root in roots:
            candidates.extend([root / split / image_path.name, root / image_path.name])
            for candidate_stem in stems:
                for suffix in self.organ_mask_suffixes:
                    for ext in self.organ_mask_extensions:
                        candidates.append(root / split / f"{candidate_stem}{suffix}{ext}")
                        candidates.append(root / f"{candidate_stem}{suffix}{ext}")
                indexed = self.organ_mask_index.get(candidate_stem)
                if indexed is not None:
                    candidates.append(indexed)
        return candidates

    @staticmethod
    def _tensor_mask_from_payload(payload):
        mask = torch.as_tensor(payload)
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

    @staticmethod
    def _safe_load_tensor_payload(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError(
                f"PyTorch does not support safe weights_only loading for organ mask {path}. "
                "Convert tensor masks to .npy or use image masks to avoid pickle loading."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Safe organ-mask tensor load failed for {path}. "
                "Convert tensor masks to .npy/image format, or regenerate the mask file."
            ) from exc

    def _load_organ_mask(self, image_info, width, height):
        for path in self._candidate_mask_paths(image_info):
            if not path.exists() or not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".npy":
                return self._tensor_mask_from_payload(np.load(path)), str(path), True
            if suffix in (".pt", ".pth"):
                payload = self._safe_load_tensor_payload(path)
                if isinstance(payload, dict):
                    for key in ("organ_mask", "anatomy_mask", "mask", "seg"):
                        if key in payload:
                            payload = payload[key]
                            break
                return self._tensor_mask_from_payload(payload), str(path), True
            with Image.open(path) as mask_image:
                return mask_image.copy(), str(path), True

        self.organ_mask_missing_count += 1
        if self.strict_organ_masks:
            raise FileNotFoundError(
                f"Missing organ mask for image_id={image_info.get('id', 'unknown')} "
                f"file_name={image_info.get('file_name', 'unknown')} under {self.organ_mask_root}"
            )
        if self.organ_mask_missing_count <= self.organ_mask_missing_warning_limit:
            print(
                f"Warning: missing organ mask for file_name={image_info.get('file_name', 'unknown')}; "
                f"using {'all-ones' if self.organ_mask_missing_as_ones else 'all-zeros'} fallback."
            )
        if self.organ_mask_missing_as_ones:
            return Image.new("L", (width, height), color=255), "", False
        return Image.new("L", (width, height), color=0), "", False

    def get_sample_type(self, idx):
        return self.sample_types[idx]

    def get_case_id(self, idx):
        image_info = self.coco.imgs.get(self.ids[idx], {})
        case_id = image_info.get("case_id", self.ids[idx])
        return str(case_id)

    def get_slice_index(self, idx):
        image_info = self.coco.imgs.get(self.ids[idx], {})
        value = image_info.get("slice_index")
        return int(value) if value is not None else idx

    def get_case_neighbor_indices(self, idx, window=5, include_self=True):
        case_id = self.get_case_id(idx)
        slice_index = self.get_slice_index(idx)
        indices = []
        for neighbor_idx in self.case_to_indices.get(case_id, []):
            if not include_self and neighbor_idx == idx:
                continue
            if abs(self.get_slice_index(neighbor_idx) - slice_index) <= window:
                indices.append(neighbor_idx)
        return indices

    def get_case_label(self, idx):
        return int(self.case_has_positive.get(self.get_case_id(idx), False))

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        sample_type = self.sample_types[idx]
        if sample_type == "UNKNOWN":
            sample_type = "P" if len(target["boxes"]) > 0 else "N"
        target["sample_type"] = sample_type
        target["sample_type_id"] = torch.tensor(sample_type_to_id(sample_type), dtype=torch.int64)
        image_info = self.coco.imgs.get(image_id, {})
        organ_mask, organ_mask_path, organ_mask_available = self._load_organ_mask(
            image_info, width=img.size[0], height=img.size[1]
        )
        target["organ_mask"] = organ_mask
        target["organ_mask_available"] = torch.tensor(int(organ_mask_available), dtype=torch.int64)
        target["organ_mask_path"] = organ_mask_path
        if "case_id" in image_info:
            target["case_id"] = image_info["case_id"]
            target["case_label"] = torch.tensor(self.get_case_label(idx), dtype=torch.int64)
        else:
            target["case_label"] = torch.tensor(int(len(target["boxes"]) > 0), dtype=torch.int64)
        if "slice_index" in image_info:
            target["slice_index"] = torch.tensor(int(image_info["slice_index"]), dtype=torch.int64)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, single_class=True):
        self.return_masks = return_masks
        self.single_class = single_class

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.single_class:
            classes = torch.zeros((len(anno),), dtype=torch.int64)
        else:
            classes = [obj["category_id"] for obj in anno]
            classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


class FixedImageSize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        return T.resize(image, target, (self.size[1], self.size[0]))


def make_coco_transforms(image_set, args):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(args.pixel_mean, args.pixel_std)
    ])
    fixed_resize = FixedImageSize((args.input_height, args.input_width))
    train_transforms = [
        fixed_resize,
        T.RandomHorizontalFlip(),
        T.RandomRotationBoxSafe(args.rotation_degrees, p=args.rotation_prob),
        T.ToTensor(),
        T.SafeElasticDeformation(
            p=args.elastic_prob,
            alpha=args.elastic_alpha,
            sigma=args.elastic_sigma,
            safe_mode=args.elastic_safe_mode,
        ),
        T.ROICopyPaste(
            p=args.roi_copy_paste_prob,
            max_iou=args.roi_copy_paste_iou_thresh,
            max_translate=args.roi_copy_paste_max_translate,
            min_organ_overlap=args.roi_copy_paste_min_organ_overlap,
        ),
        T.Normalize(args.pixel_mean, args.pixel_std),
    ]

    if image_set == 'train':
        return T.Compose(train_transforms)

    if image_set == 'val':
        return T.Compose([
            fixed_resize,
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set, args), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(),
                            single_class=args.fracture_single_class,
                            pn_slice_window=args.pn_slice_window,
                            organ_mask_root=args.organ_mask_root,
                            organ_mask_suffixes=args.organ_mask_suffixes,
                            organ_mask_extensions=args.organ_mask_extensions,
                            organ_mask_missing_as_ones=args.organ_mask_missing_as_ones,
                            organ_mask_recursive=args.organ_mask_recursive,
                            strict_organ_masks=args.strict_organ_masks,
                            organ_mask_missing_warning_limit=args.organ_mask_missing_warning_limit,
                            require_case_metadata=args.require_case_metadata,
                            validate_organ_masks=args.strict_organ_masks and image_set != 'train',
                            organ_mask_validate_max_samples=args.organ_mask_validate_max_samples)
    return dataset
