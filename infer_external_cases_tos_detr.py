#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

import datasets.coco as coco_dataset
import util.misc as utils
from models import build_model
from util.fracture_eval import aggregate_case_predictions


def parse_args():
    parser = argparse.ArgumentParser("External case inference for ToS-DETR")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-images", default=0, type=int)
    parser.add_argument("--save-positive-case-images", action="store_true")
    parser.add_argument("--positive-case-image-dir-name", default="positive_case_images")
    parser.add_argument("--save-positive-case-labels", action="store_true")
    parser.add_argument("--positive-case-label-dir-name", default="positive_case_labels")
    parser.add_argument("--filter-isolated-slice-detections", action="store_true")
    parser.add_argument("--isolation-slice-gap", default=1, type=int)
    parser.add_argument("--save-threshold-filtered-boxes", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--threshold-filtered-boxes-file-name", default="threshold_filtered_case_boxes.json")
    return parser.parse_args()


def parse_case_and_slice(file_name: str):
    stem = Path(file_name).stem
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0], int(parts[1]), int(parts[2])
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1]), int(parts[-1]), -1
    return stem, -1, -1


def jsonable(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.view(-1)[0].item())
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


class ExternalOrganMaskDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, transform, max_images: int = 0):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        allowed_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        self.image_paths = sorted([path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in allowed_suffixes])
        if max_images and max_images > 0:
            self.image_paths = self.image_paths[:max_images]
        if not self.image_paths:
            suffixes = ", ".join(sorted(allowed_suffixes))
            raise FileNotFoundError(f"No supported image files ({suffixes}) found in {image_dir}")
        missing_masks = [path.name for path in self.image_paths if not (mask_dir / path.name).exists()]
        if missing_masks:
            raise FileNotFoundError(
                f"{len(missing_masks)} image files do not have same-name masks under {mask_dir}; "
                f"first missing: {missing_masks[:5]}"
            )
        grouped = defaultdict(list)
        for image_path in self.image_paths:
            case_id, slice_index, _ = parse_case_and_slice(image_path.name)
            grouped[case_id].append((slice_index, image_path.name))
        self.case_ids = sorted(grouped)
        self.case_to_order = {}
        for case_id, rows in grouped.items():
            for order, (_, file_name) in enumerate(sorted(rows)):
                self.case_to_order[(case_id, file_name)] = order

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / image_path.name
        image = Image.open(image_path).convert("RGB")
        organ_mask = Image.open(mask_path).convert("RGB")
        if organ_mask.size != image.size:
            organ_mask = organ_mask.resize(image.size, resample=Image.NEAREST)
        orig_w, orig_h = image.size
        case_id, slice_index, total_hint = parse_case_and_slice(image_path.name)
        target = {
            "image_id": torch.tensor([index], dtype=torch.int64),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
            "size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "file_name": image_path.name,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "case_id": case_id,
            "slice_index": torch.tensor(slice_index, dtype=torch.int64),
            "num_slices": torch.tensor(total_hint if total_hint > 0 else 1, dtype=torch.int64),
            "slice_order": torch.tensor(self.case_to_order[(case_id, image_path.name)], dtype=torch.int64),
            "organ_mask": organ_mask,
            "organ_mask_available": torch.tensor(1, dtype=torch.int64),
            "organ_mask_path": str(mask_path),
        }
        return self.transform(image, target)


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def aggregate_kwargs(args):
    signature = inspect.signature(aggregate_case_predictions)
    candidates = {
        "alpha": getattr(args, "case_alpha", 0.7),
        "beta": getattr(args, "case_beta", 0.3),
        "topk": getattr(args, "case_topk_lesions", 5),
        "score_threshold": getattr(args, "fracture_score_threshold", 0.3),
        "iou_threshold": getattr(args, "case_association_iou", 0.2),
        "min_continuity": getattr(args, "case_min_continuity", 2),
        "continuity_norm": getattr(args, "case_continuity_norm", 5.0),
        "center_distance_threshold": getattr(args, "case_center_distance_threshold", 28.0),
        "fallback_weight": getattr(args, "case_fallback_weight", 0.0),
    }
    return {key: value for key, value in candidates.items() if key in signature.parameters}


def prepare_for_aggregation(item):
    converted = dict(item)
    if "box" in converted:
        converted["box"] = torch.as_tensor(converted["box"], dtype=torch.float32)
    for key in ("score", "organ_score", "uncertainty", "certainty"):
        if key in converted:
            converted[key] = float(converted[key])
    return converted


def filter_isolated_slice_detections(detections_by_file: Dict[str, List[dict]], slice_gap: int = 1):
    slice_gap = max(1, int(slice_gap))
    case_to_slices = defaultdict(set)
    file_to_case_slice = {}
    raw_box_count = 0
    for file_name, detections in detections_by_file.items():
        case_id, slice_index, _ = parse_case_and_slice(file_name)
        file_to_case_slice[file_name] = (case_id, slice_index)
        raw_box_count += len(detections)
        if detections:
            case_to_slices[case_id].add(slice_index)

    filtered_by_file = {}
    removed_slice_count = 0
    removed_box_count = 0
    kept_box_count = 0
    kept_slice_count = 0
    for file_name, detections in detections_by_file.items():
        case_id, slice_index = file_to_case_slice[file_name]
        if not detections:
            filtered_by_file[file_name] = []
            continue
        positive_slices = case_to_slices[case_id]
        has_neighbor = any(
            (slice_index - gap in positive_slices) or (slice_index + gap in positive_slices)
            for gap in range(1, slice_gap + 1)
        )
        if has_neighbor:
            filtered_by_file[file_name] = detections
            kept_slice_count += 1
            kept_box_count += len(detections)
        else:
            filtered_by_file[file_name] = []
            removed_slice_count += 1
            removed_box_count += len(detections)

    return filtered_by_file, {
        "enabled": True,
        "isolation_slice_gap": slice_gap,
        "raw_box_count": raw_box_count,
        "kept_box_count": kept_box_count,
        "removed_box_count": removed_box_count,
        "kept_slice_count": kept_slice_count,
        "removed_slice_count": removed_slice_count,
    }


def clamp_box(box, width: int, height: int):
    x0, y0, x1, y1 = [float(value) for value in box]
    x0 = max(0.0, min(float(width - 1), x0))
    y0 = max(0.0, min(float(height - 1), y0))
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def draw_detections(image_path: Path, output_path: Path, detections: List[dict]):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for detection in detections:
        box = detection.get("box")
        if box is None:
            continue
        x0, y0, x1, y1 = clamp_box(box, width, height)
        draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=2)
        score = float(detection.get("score", 0.0))
        label = f"{score:.3f}"
        text_bbox = draw.textbbox((0, 0), label)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        tx = max(0, min(width - text_w - 2, int(x0)))
        ty = max(0, int(y0) - text_h - 4)
        draw.rectangle((tx, ty, tx + text_w + 2, ty + text_h + 2), fill=(255, 0, 0))
        draw.text((tx + 1, ty + 1), label, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def detection_to_yolo_line(detection: dict, width: int, height: int):
    box = detection.get("box")
    if box is None:
        return None
    x0, y0, x1, y1 = clamp_box(box, width, height)
    box_w = max(0.0, x1 - x0)
    box_h = max(0.0, y1 - y0)
    if box_w <= 0.0 or box_h <= 0.0:
        return None
    x_center = (x0 + x1) * 0.5 / max(float(width), 1.0)
    y_center = (y0 + y1) * 0.5 / max(float(height), 1.0)
    norm_w = box_w / max(float(width), 1.0)
    norm_h = box_h / max(float(height), 1.0)
    values = [
        0,
        min(max(x_center, 0.0), 1.0),
        min(max(y_center, 0.0), 1.0),
        min(max(norm_w, 0.0), 1.0),
        min(max(norm_h, 0.0), 1.0),
    ]
    return f"{values[0]} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f} {values[4]:.6f}"


def detection_to_yolo_values(detection: dict, width: int, height: int):
    box = detection.get("box")
    if box is None:
        return None
    x0, y0, x1, y1 = clamp_box(box, width, height)
    box_w = max(0.0, x1 - x0)
    box_h = max(0.0, y1 - y0)
    if box_w <= 0.0 or box_h <= 0.0:
        return None
    x_center = (x0 + x1) * 0.5 / max(float(width), 1.0)
    y_center = (y0 + y1) * 0.5 / max(float(height), 1.0)
    norm_w = box_w / max(float(width), 1.0)
    norm_h = box_h / max(float(height), 1.0)
    return {
        "class": 0,
        "x_center": min(max(x_center, 0.0), 1.0),
        "y_center": min(max(y_center, 0.0), 1.0),
        "width": min(max(norm_w, 0.0), 1.0),
        "height": min(max(norm_h, 0.0), 1.0),
    }


def save_threshold_filtered_case_boxes(args, dataset, detections_by_file, score_threshold: float):
    image_by_name = {path.name: path for path in dataset.image_paths}
    cases = defaultdict(lambda: {
        "case_id": "",
        "file_count": 0,
        "box_count": 0,
        "files": defaultdict(list),
    })
    total_box_count = 0
    kept_file_count = 0

    for file_name in sorted(detections_by_file):
        image_path = image_by_name.get(file_name)
        if image_path is None:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        case_id, slice_index, _ = parse_case_and_slice(file_name)
        kept_boxes = []
        for box_index, detection in enumerate(detections_by_file.get(file_name, [])):
            score = float(detection.get("score", 0.0))
            if score < score_threshold:
                continue
            box = detection.get("box")
            if box is None:
                continue
            x0, y0, x1, y1 = clamp_box(box, width, height)
            yolo_values = detection_to_yolo_values(detection, width, height)
            if yolo_values is None:
                continue
            kept_boxes.append({
                "box_index": int(box_index),
                "score": score,
                "box_xyxy": [x0, y0, x1, y1],
                "box_yolo_normalized": yolo_values,
                "slice_index": int(detection.get("slice_index", slice_index)),
                "slice_order": int(detection.get("slice_order", -1)),
            })
        if not kept_boxes:
            continue
        case_row = cases[case_id]
        case_row["case_id"] = case_id
        if file_name not in case_row["files"]:
            case_row["file_count"] += 1
        case_row["files"][file_name].extend(kept_boxes)
        case_row["box_count"] += len(kept_boxes)
        total_box_count += len(kept_boxes)
        kept_file_count += 1

    serializable_cases = {}
    for case_id in sorted(cases):
        row = cases[case_id]
        serializable_cases[case_id] = {
            "case_id": row["case_id"],
            "file_count": int(row["file_count"]),
            "box_count": int(row["box_count"]),
            "files": dict(sorted(row["files"].items())),
        }

    output = {
        "format": {
            "box_xyxy": "pixel coordinates [x0, y0, x1, y1] after clamping to the corresponding input image",
            "box_yolo_normalized": "class, x_center, y_center, width, height normalized to the corresponding input image size",
            "score": "model fracture score for this retained box",
        },
        "score_threshold": float(score_threshold),
        "case_count": len(serializable_cases),
        "file_count": kept_file_count,
        "box_count": total_box_count,
        "cases": serializable_cases,
    }
    output_path = args.output_dir / args.threshold_filtered_boxes_file_name
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output_file": str(output_path),
        "case_count": len(serializable_cases),
        "file_count": kept_file_count,
        "box_count": total_box_count,
        "score_threshold": float(score_threshold),
    }


def write_detection_labels(image_path: Path, output_path: Path, detections: List[dict]):
    with Image.open(image_path) as image:
        width, height = image.size
    lines = []
    for detection in detections:
        line = detection_to_yolo_line(detection, width, height)
        if line is not None:
            lines.append(line)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content:
        content += "\n"
    output_path.write_text(content, encoding="utf-8")
    return len(lines)


def save_positive_case_labels(args, dataset, positive_case_ids, detections_by_file):
    output_dir = args.output_dir / args.positive_case_label_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_txt in output_dir.glob("*.txt"):
        old_txt.unlink()

    positive_case_ids = set(positive_case_ids)
    manifest = {
        "output_dir": str(output_dir),
        "format": "class x_center y_center width height; normalized to the corresponding input image size",
        "positive_case_count": len(positive_case_ids),
        "saved_label_file_count": 0,
        "nonempty_label_file_count": 0,
        "empty_label_file_count": 0,
        "total_box_count": 0,
        "cases": defaultdict(lambda: {
            "saved_label_file_count": 0,
            "nonempty_label_file_count": 0,
            "empty_label_file_count": 0,
            "total_box_count": 0,
        }),
    }
    for image_path in dataset.image_paths:
        case_id, _, _ = parse_case_and_slice(image_path.name)
        if case_id not in positive_case_ids:
            continue
        output_path = output_dir / f"{image_path.stem}.txt"
        detections = detections_by_file.get(image_path.name, [])
        box_count = write_detection_labels(image_path, output_path, detections)
        manifest["saved_label_file_count"] += 1
        manifest["total_box_count"] += box_count
        manifest["cases"][case_id]["saved_label_file_count"] += 1
        manifest["cases"][case_id]["total_box_count"] += box_count
        if box_count > 0:
            manifest["nonempty_label_file_count"] += 1
            manifest["cases"][case_id]["nonempty_label_file_count"] += 1
        else:
            manifest["empty_label_file_count"] += 1
            manifest["cases"][case_id]["empty_label_file_count"] += 1

    manifest["cases"] = dict(sorted(manifest["cases"].items()))
    (args.output_dir / "positive_case_labels_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def save_positive_case_images(args, dataset, positive_case_ids, detections_by_file):
    output_dir = args.output_dir / args.positive_case_image_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_png in output_dir.glob("*.png"):
        old_png.unlink()

    positive_case_ids = set(positive_case_ids)
    manifest = {
        "output_dir": str(output_dir),
        "positive_case_count": len(positive_case_ids),
        "saved_image_count": 0,
        "visualized_image_count": 0,
        "copied_image_count": 0,
        "cases": defaultdict(lambda: {
            "saved_image_count": 0,
            "visualized_image_count": 0,
            "copied_image_count": 0,
        }),
    }
    for image_path in dataset.image_paths:
        case_id, _, _ = parse_case_and_slice(image_path.name)
        if case_id not in positive_case_ids:
            continue
        output_path = output_dir / image_path.name
        detections = detections_by_file.get(image_path.name, [])
        if detections:
            draw_detections(image_path, output_path, detections)
            manifest["visualized_image_count"] += 1
            manifest["cases"][case_id]["visualized_image_count"] += 1
        else:
            shutil.copy2(image_path, output_path)
            manifest["copied_image_count"] += 1
            manifest["cases"][case_id]["copied_image_count"] += 1
        manifest["saved_image_count"] += 1
        manifest["cases"][case_id]["saved_image_count"] += 1

    manifest["cases"] = dict(sorted(manifest["cases"].items()))
    (args.output_dir / "positive_case_images_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint["args"]
    ckpt_args.device = args.device
    ckpt_args.eval = True
    ckpt_args.distributed = False
    ckpt_args.backbone_pretrained = False
    ckpt_args.backbone_pretrained_path = ""
    ckpt_args.pretrained_model = ""

    transform = coco_dataset.make_coco_transforms("val", ckpt_args)
    dataset = ExternalOrganMaskDataset(args.image_dir, args.mask_dir, transform, max_images=args.max_images)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=utils.collate_fn,
    )

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    torch.backends.cudnn.benchmark = True
    model, _, postprocessors = build_model(ckpt_args)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    fracture_post = postprocessors["fracture"]

    case_detections: Dict[str, List[dict]] = {case_id: [] for case_id in dataset.case_ids}
    detections_by_file: Dict[str, List[dict]] = {}
    slice_summaries = []
    total_batches = len(loader)
    with torch.no_grad():
        for batch_idx, (samples, targets) in enumerate(loader, start=1):
            samples = samples.to(device)
            model_targets = move_to_device(targets, device)
            target_sizes = torch.stack([target["orig_size"] for target in targets], dim=0).to(device)
            with torch.autocast(device_type="cuda", enabled=args.amp and device.type == "cuda"):
                outputs = model(samples, targets=model_targets)
            predictions = fracture_post(outputs, target_sizes, model_targets)
            for target, prediction in zip(targets, predictions):
                case_id = str(target["case_id"])
                slice_index = int(target["slice_index"].item())
                slice_order = int(target["slice_order"].item())
                detections = []
                for detection in prediction.get("detections", []):
                    item = jsonable(detection)
                    item["case_id"] = case_id
                    item["slice_index"] = int(item.get("slice_index", slice_index))
                    item["slice_order"] = slice_order
                    detections.append(item)
                detections_by_file[target["file_name"]] = detections
                case_detections.setdefault(case_id, []).extend(detections)
                slice_summaries.append({
                    "file_name": target["file_name"],
                    "case_id": case_id,
                    "slice_index": slice_index,
                    "slice_order": slice_order,
                    "num_detections": len(detections),
                    "max_score": max((float(item.get("score", 0.0)) for item in detections), default=0.0),
                })
            if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == total_batches:
                print(f"[ToS-DETR progress] {batch_idx}/{total_batches} batches", flush=True)

    filter_summary = {
        "enabled": False,
        "isolation_slice_gap": int(args.isolation_slice_gap),
    }
    if args.filter_isolated_slice_detections:
        detections_by_file, filter_summary = filter_isolated_slice_detections(
            detections_by_file,
            slice_gap=args.isolation_slice_gap,
        )
        case_detections = {case_id: [] for case_id in dataset.case_ids}
        for file_name, detections in detections_by_file.items():
            case_id, _, _ = parse_case_and_slice(file_name)
            case_detections.setdefault(case_id, []).extend(detections)

    kwargs = aggregate_kwargs(ckpt_args)
    score_threshold = float(kwargs.get("score_threshold", getattr(ckpt_args, "fracture_score_threshold", 0.3)))
    case_threshold = float(getattr(ckpt_args, "case_threshold", 0.5))
    case_predictions = {}
    for case_id in sorted(case_detections):
        detections = [prepare_for_aggregation(item) for item in case_detections[case_id]]
        summary = aggregate_case_predictions(detections, **kwargs)
        predicted_positive = float(summary.get("case_score", 0.0)) >= case_threshold
        case_predictions[case_id] = {
            **jsonable(summary),
            "predicted_positive": bool(predicted_positive),
            "num_slice_detections": len(detections),
        }

    positives = [case_id for case_id, row in case_predictions.items() if row["predicted_positive"]]
    negatives = [case_id for case_id, row in case_predictions.items() if not row["predicted_positive"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    threshold_filtered_boxes_manifest = None
    if args.save_threshold_filtered_boxes:
        threshold_filtered_boxes_manifest = save_threshold_filtered_case_boxes(
            args,
            dataset,
            detections_by_file,
            score_threshold=score_threshold,
        )
    summary = {
        "model": "ToS-DETR",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "image_dir": str(args.image_dir),
        "mask_dir": str(args.mask_dir),
        "num_images": len(dataset),
        "num_cases": len(case_predictions),
        "predicted_positive_cases": len(positives),
        "predicted_negative_cases": len(negatives),
        "case_threshold": case_threshold,
        "box_score_threshold": score_threshold,
        "aggregate_kwargs": kwargs,
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "amp": bool(args.amp),
        "isolated_slice_filter": filter_summary,
    }
    if threshold_filtered_boxes_manifest is not None:
        summary["threshold_filtered_boxes"] = threshold_filtered_boxes_manifest
    (args.output_dir / "case_predictions.json").write_text(json.dumps(case_predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "slice_summaries.jsonl").open("w", encoding="utf-8") as handle:
        for item in slice_summaries:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    for file_name in ("predicted_positive_cases.txt", "positive_cases.txt"):
        (args.output_dir / file_name).write_text("\n".join(positives), encoding="utf-8")
    for file_name in ("predicted_negative_cases.txt", "negative_cases.txt"):
        (args.output_dir / file_name).write_text("\n".join(negatives), encoding="utf-8")
    if args.save_positive_case_images:
        manifest = save_positive_case_images(args, dataset, positives, detections_by_file)
        summary["positive_case_image_dir"] = manifest["output_dir"]
        summary["positive_case_saved_images"] = manifest["saved_image_count"]
        summary["positive_case_visualized_images"] = manifest["visualized_image_count"]
        summary["positive_case_copied_images"] = manifest["copied_image_count"]
    if args.save_positive_case_labels:
        manifest = save_positive_case_labels(args, dataset, positives, detections_by_file)
        summary["positive_case_label_dir"] = manifest["output_dir"]
        summary["positive_case_saved_label_files"] = manifest["saved_label_file_count"]
        summary["positive_case_nonempty_label_files"] = manifest["nonempty_label_file_count"]
        summary["positive_case_empty_label_files"] = manifest["empty_label_file_count"]
        summary["positive_case_total_label_boxes"] = manifest["total_box_count"]
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
    main()
