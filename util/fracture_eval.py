import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from util import box_ops
from util.misc import all_gather


DEFAULT_SCORE_THRESHOLD = 0.3
DEFAULT_ASSOC_IOU_THRESHOLD = 0.2
DEFAULT_MIN_CONTINUITY = 2
DEFAULT_MAX_SLICE_GAP = 1
DEFAULT_TOPK_LESIONS = 5
DEFAULT_CASE_ALPHA = 0.7
DEFAULT_CASE_BETA = 0.3
DEFAULT_CONTINUITY_NORM = 5.0
DEFAULT_CASE_THRESHOLD = 0.5
DEFAULT_CASE_FALLBACK_WEIGHT = 0.0
DEFAULT_CENTER_DISTANCE_THRESHOLD = 28.0
DEFAULT_SIZE_RATIO_THRESHOLD = 0.45
DEFAULT_MAX_FROC_THRESHOLDS = 256
DEFAULT_FROC_MIN_SCORE = 0.05
DEFAULT_FROC_MAX_DETECTIONS_PER_SLICE = 20


def _case_key(target: dict) -> str:
    case_id = target.get("case_id", target.get("image_id", "unknown"))
    if isinstance(case_id, torch.Tensor):
        case_id = case_id.flatten()[0].item()
    return str(case_id)


def _image_key(target: dict) -> str:
    image_id = target.get("image_id")
    if isinstance(image_id, torch.Tensor):
        image_id = image_id.flatten()[0].item()
    if image_id is not None:
        return str(image_id)
    return f"{_case_key(target)}:{target.get('slice_index', 'unknown')}"


def _detach_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _detach_to_cpu(item) for key, item in value.items()
                if key != "organ_mask"}
    if isinstance(value, list):
        return [_detach_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(item) for item in value)
    return value


def _box_center(box: torch.Tensor) -> torch.Tensor:
    return torch.stack([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])


def _box_size(box: torch.Tensor) -> torch.Tensor:
    return (box[2:] - box[:2]).clamp(min=1e-6)


def _adaptive_center_threshold(
    box: torch.Tensor,
    prev_box: torch.Tensor,
    center_distance_threshold: float,
) -> float:
    base_threshold = float(center_distance_threshold)
    box_diag = torch.norm(_box_size(box), p=2).item()
    prev_diag = torch.norm(_box_size(prev_box), p=2).item()
    size_scaled = 0.75 * max((box_diag + prev_diag) * 0.5, 1.0)
    min_threshold = min(base_threshold, 8.0)
    return max(min_threshold, min(base_threshold, size_scaled))


def _track_compatible(
    box: torch.Tensor,
    prev_box: torch.Tensor,
    iou_threshold: float,
    center_distance_threshold: float,
    size_ratio_threshold: float,
) -> bool:
    iou = box_ops.box_iou(box.view(1, 4), prev_box.view(1, 4))[0].item()
    if iou >= iou_threshold:
        return True
    center_distance = torch.norm(_box_center(box) - _box_center(prev_box), p=2).item()
    center_threshold = _adaptive_center_threshold(box, prev_box, center_distance_threshold)
    box_size = _box_size(box)
    prev_size = _box_size(prev_box)
    min_size = torch.minimum(box_size, prev_size)
    max_size = torch.maximum(box_size, prev_size)
    size_similarity = float((min_size / max_size.clamp(min=1e-6)).mean().item())
    return center_distance <= center_threshold and size_similarity >= size_ratio_threshold


def associate_cross_slice_lesions(
    detections: Sequence[dict],
    iou_threshold: float = DEFAULT_ASSOC_IOU_THRESHOLD,
    min_continuity: int = DEFAULT_MIN_CONTINUITY,
    max_slice_gap: int = DEFAULT_MAX_SLICE_GAP,
    center_distance_threshold: float = DEFAULT_CENTER_DISTANCE_THRESHOLD,
    size_ratio_threshold: float = DEFAULT_SIZE_RATIO_THRESHOLD,
) -> List[dict]:
    sorted_detections = sorted(detections, key=lambda item: int(item.get("slice_index", 0)))
    groups: List[List[dict]] = []
    for detection in sorted_detections:
        box = detection.get("box")
        slice_index = int(detection.get("slice_index", 0))
        matched_group = None
        for group_id, group in enumerate(groups):
            previous = group[-1]
            if abs(slice_index - int(previous.get("slice_index", 0))) > max(int(max_slice_gap), 1):
                continue
            prev_box = previous.get("box")
            if box is None or prev_box is None:
                continue
            if _track_compatible(
                box, prev_box,
                iou_threshold=iou_threshold,
                center_distance_threshold=center_distance_threshold,
                size_ratio_threshold=size_ratio_threshold,
            ):
                matched_group = group_id
                break
        if matched_group is None:
            groups.append([detection])
        else:
            groups[matched_group].append(detection)

    associated = []
    for group_id, group in enumerate(groups):
        is_true_lesion = len({int(item.get("slice_index", 0)) for item in group}) >= min_continuity
        for item in group:
            associated.append({**item, "association_id": group_id, "passes_continuity": is_true_lesion})
    return associated


def _score_value(item: dict, score_key: str = "score") -> float:
    score = item.get(score_key, 0.0)
    if isinstance(score, torch.Tensor):
        return float(score.detach().cpu().view(-1)[0].item())
    return float(score)


def _filter_froc_candidates(
    detections: Sequence[dict],
    min_score: float,
    max_detections_per_slice: int,
) -> List[dict]:
    candidates = [
        detection for detection in detections
        if _score_value(detection) >= min_score
    ]
    candidates.sort(key=_score_value, reverse=True)
    if max_detections_per_slice and max_detections_per_slice > 0:
        candidates = candidates[:max_detections_per_slice]
    return candidates


def _summarize_lesion_groups(
    detections: Sequence[dict],
    score_key: str = "score",
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = DEFAULT_ASSOC_IOU_THRESHOLD,
    min_continuity: int = DEFAULT_MIN_CONTINUITY,
    max_slice_gap: int = DEFAULT_MAX_SLICE_GAP,
    center_distance_threshold: float = DEFAULT_CENTER_DISTANCE_THRESHOLD,
) -> List[dict]:
    filtered = [item for item in detections if _score_value(item, score_key) >= score_threshold]
    if not filtered:
        return []
    associated = associate_cross_slice_lesions(
        filtered,
        iou_threshold=iou_threshold,
        min_continuity=min_continuity,
        max_slice_gap=max_slice_gap,
        center_distance_threshold=center_distance_threshold,
    )
    lesion_groups = defaultdict(list)
    for item in associated:
        lesion_groups[item.get("association_id")].append(item)

    lesions = []
    for group_id, group in lesion_groups.items():
        # One lesion track can have multiple detections on a slice. Use the
        # strongest per-slice score, then average across its continuous extent.
        per_slice_score = {}
        for item in group:
            slice_index = int(item.get("slice_index", 0))
            per_slice_score[slice_index] = max(per_slice_score.get(slice_index, 0.0), _score_value(item, score_key))
        continuity = len(per_slice_score)
        slice_scores = list(per_slice_score.values())
        lesion_score = float(torch.tensor(slice_scores, dtype=torch.float32).mean().item()) if slice_scores else 0.0
        has_continuity = continuity >= min_continuity
        lesions.append({
            "association_id": group_id,
            "items": group,
            "score": lesion_score,
            "max_score": max(slice_scores) if slice_scores else 0.0,
            "continuity": continuity,
            "passes_continuity": has_continuity,
            "primary": has_continuity,
            "isolated": not has_continuity,
        })
    return lesions


def aggregate_case_predictions(
    detections: Sequence[dict],
    score_key: str = "score",
    alpha: float = DEFAULT_CASE_ALPHA,
    beta: float = DEFAULT_CASE_BETA,
    topk: int = DEFAULT_TOPK_LESIONS,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = DEFAULT_ASSOC_IOU_THRESHOLD,
    min_continuity: int = DEFAULT_MIN_CONTINUITY,
    max_slice_gap: int = DEFAULT_MAX_SLICE_GAP,
    continuity_norm: float = DEFAULT_CONTINUITY_NORM,
    center_distance_threshold: float = DEFAULT_CENTER_DISTANCE_THRESHOLD,
    fallback_weight: float = DEFAULT_CASE_FALLBACK_WEIGHT,
) -> Dict[str, float]:
    lesions = _summarize_lesion_groups(
        detections,
        score_key=score_key,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        min_continuity=min_continuity,
        max_slice_gap=max_slice_gap,
        center_distance_threshold=center_distance_threshold,
    )
    if not lesions:
        return {
            "case_score": 0.0,
            "lesion_count": 0.0,
            "primary_lesion_count": 0.0,
            "isolated_lesion_count": 0.0,
            "continuity_score": 0.0,
            "topk_lesion_score": 0.0,
            "primary_topk_lesion_score": 0.0,
            "fallback_topk_lesion_score": 0.0,
            "max_continuity": 0.0,
            "fallback_case_score": 0.0,
            "primary_case_score": 0.0,
        }

    primary_lesions = [item for item in lesions if item.get("passes_continuity", False)]
    fallback_lesions = [item for item in lesions if not item.get("passes_continuity", False)]

    def _topk_mean(items: Sequence[dict]) -> float:
        if not items:
            return 0.0
        lesion_scores = torch.tensor([item["score"] for item in items], dtype=torch.float32)
        return float(lesion_scores.topk(min(topk, lesion_scores.numel())).values.mean().item())

    primary_topk_score = _topk_mean(primary_lesions)
    fallback_topk_score = _topk_mean(fallback_lesions)
    primary_max_continuity = max((item["continuity"] for item in primary_lesions), default=0)
    fallback_max_continuity = max((item["continuity"] for item in fallback_lesions), default=0)
    max_continuity = max(primary_max_continuity, fallback_max_continuity)
    primary_continuity_score = min(
        math.log1p(float(primary_max_continuity)) / math.log1p(max(float(continuity_norm), 1.0)),
        1.0,
    )
    fallback_case_score = float(fallback_topk_score)
    primary_case_score = float(alpha * primary_topk_score + beta * primary_continuity_score)
    case_score = primary_case_score + fallback_weight * fallback_case_score
    return {
        "case_score": float(case_score),
        "lesion_count": float(len(lesions)),
        "primary_lesion_count": float(len(primary_lesions)),
        "isolated_lesion_count": float(len(fallback_lesions)),
        "continuity_score": primary_continuity_score,
        "all_track_continuity_score": min(
            math.log1p(float(max_continuity)) / math.log1p(max(float(continuity_norm), 1.0)),
            1.0,
        ),
        "topk_lesion_score": float(primary_topk_score if primary_lesions else fallback_topk_score),
        "primary_topk_lesion_score": float(primary_topk_score),
        "fallback_topk_lesion_score": float(fallback_topk_score),
        "max_continuity": float(max_continuity),
        "fallback_case_score": float(fallback_case_score),
        "primary_case_score": float(primary_case_score),
    }


def _target_boxes_xyxy(target: dict, device: torch.device) -> torch.Tensor:
    boxes = target.get("boxes")
    if boxes is None or boxes.numel() == 0:
        return torch.zeros((0, 4), device=device)
    boxes = box_ops.box_cxcywh_to_xyxy(boxes.to(device))
    if "orig_size" in target:
        h, w = target["orig_size"].to(device).float()
        boxes = boxes * torch.stack([w, h, w, h])
    return boxes


def _target_has_boxes(target: dict) -> bool:
    boxes = target.get("boxes")
    return boxes is not None and boxes.numel() > 0


def _target_case_labels(targets: Sequence[dict]) -> Dict[str, int]:
    case_labels = defaultdict(int)
    for target in targets:
        label = target.get("case_label")
        if isinstance(label, torch.Tensor):
            label = int(label.detach().cpu().view(-1)[0].item())
        elif label is None:
            label = int(_target_has_boxes(target))
        else:
            label = int(label)
        case_labels[_case_key(target)] = max(case_labels[_case_key(target)], label)
    return dict(case_labels)


def _match_detections(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, iou_threshold: float = 0.2):
    if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
        return 0, pred_boxes.shape[0], target_boxes.shape[0]
    ious = box_ops.box_iou(pred_boxes, target_boxes)[0]
    matched_targets = set()
    true_positive = 0
    for pred_index in range(ious.shape[0]):
        best_iou, best_target = torch.max(ious[pred_index], dim=0)
        target_index = int(best_target.item())
        if best_iou.item() >= iou_threshold and target_index not in matched_targets:
            matched_targets.add(target_index)
            true_positive += 1
    false_positive = pred_boxes.shape[0] - true_positive
    false_negative = target_boxes.shape[0] - true_positive
    return true_positive, false_positive, false_negative


def _count_lesions(lesions_by_case: Dict[str, List[dict]]) -> int:
    return sum(len(lesions) for lesions in lesions_by_case.values())


def _filter_primary_lesions(lesions_by_case: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    return {
        case_id: [lesion for lesion in lesions if lesion.get("passes_continuity", False)]
        for case_id, lesions in lesions_by_case.items()
    }


def compute_lesion_metrics(
    predictions: Sequence[dict],
    targets: Sequence[dict],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = DEFAULT_ASSOC_IOU_THRESHOLD,
    min_continuity: int = DEFAULT_MIN_CONTINUITY,
) -> Dict[str, float]:
    pred_cases = _prediction_lesions_by_case(
        predictions,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        min_continuity=min_continuity,
        source_key="detections",
    )
    gt_cases = _target_lesions_by_case(
        targets,
        iou_threshold=iou_threshold,
        min_continuity=1,
    )

    def _accumulate(candidate_pred_cases):
        total_tp = total_fp = total_fn = 0
        for case_id in set(candidate_pred_cases) | set(gt_cases):
            tp, fp, fn = _match_lesion_tracks(
                candidate_pred_cases.get(case_id, []),
                gt_cases.get(case_id, []),
                iou_threshold=iou_threshold,
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn
        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        return precision, recall, total_tp, total_fp, total_fn

    precision, recall, total_tp, total_fp, total_fn = _accumulate(pred_cases)
    primary_pred_cases = _filter_primary_lesions(pred_cases)
    primary_precision, primary_recall, primary_tp, primary_fp, primary_fn = _accumulate(primary_pred_cases)

    isolated_fp = 0
    for case_id, lesions in pred_cases.items():
        isolated_fp += sum(1 for lesion in lesions if not lesion.get("passes_continuity", False))

    return {
        "lesion_precision": precision,
        "lesion_recall": recall,
        "lesion_track_precision": precision,
        "lesion_track_recall": recall,
        "lesion_track_tp": float(total_tp),
        "lesion_track_fp": float(total_fp),
        "lesion_track_fn": float(total_fn),
        "primary_lesion_precision": primary_precision,
        "primary_lesion_recall": primary_recall,
        "primary_lesion_track_precision": primary_precision,
        "primary_lesion_track_recall": primary_recall,
        "primary_lesion_track_tp": float(primary_tp),
        "primary_lesion_track_fp": float(primary_fp),
        "primary_lesion_track_fn": float(primary_fn),
        "isolated_pred_lesion_count": float(isolated_fp),
        "gt_lesion_track_count": float(_count_lesions(gt_cases)),
        "pred_lesion_track_count": float(_count_lesions(pred_cases)),
        "primary_pred_lesion_track_count": float(_count_lesions(primary_pred_cases)),
    }


def _compute_froc_for_cases(
    pred_cases: Dict[str, List[dict]],
    gt_cases: Dict[str, List[dict]],
    case_count: int,
    fp_per_case: Iterable[float],
    iou_threshold: float,
    prefix: str,
) -> Dict[str, float]:
    lesion_scores = [
        lesion["score"]
        for lesions in pred_cases.values()
        for lesion in lesions
    ]
    froc_stats = {}
    for fp in fp_per_case:
        froc_stats[f"{prefix}froc@{fp:g}"] = 0.0
        froc_stats[f"{prefix}lesion_track_froc@{fp:g}"] = 0.0
    if not lesion_scores:
        return froc_stats

    thresholds = torch.unique(torch.tensor(lesion_scores, dtype=torch.float32)).sort(descending=True).values
    best_recall = {fp: 0.0 for fp in fp_per_case}
    total_gt = _count_lesions(gt_cases)

    for threshold in thresholds:
        total_tp = total_fp = total_fn = 0
        for case_id in set(pred_cases) | set(gt_cases):
            kept_predictions = [
                lesion for lesion in pred_cases.get(case_id, [])
                if lesion["score"] >= float(threshold.item())
            ]
            gt_lesions = gt_cases.get(case_id, [])
            tp, fp, fn = _match_lesion_tracks(kept_predictions, gt_lesions, iou_threshold=iou_threshold)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        fp_case = total_fp / max(case_count, 1)
        recall = total_tp / max(total_gt, total_tp + total_fn, 1)
        for fp_target in fp_per_case:
            if fp_case <= fp_target:
                best_recall[fp_target] = max(best_recall[fp_target], recall)

    for fp_target, recall in best_recall.items():
        froc_stats[f"{prefix}froc@{fp_target:g}"] = recall
        froc_stats[f"{prefix}lesion_track_froc@{fp_target:g}"] = recall
    return froc_stats


def compute_froc(
    predictions: Sequence[dict],
    targets: Sequence[dict],
    fp_per_case: Iterable[float] = (0.5, 1.0, 2.0, 4.0),
    iou_threshold: float = DEFAULT_ASSOC_IOU_THRESHOLD,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    min_continuity: int = DEFAULT_MIN_CONTINUITY,
    max_thresholds: int = DEFAULT_MAX_FROC_THRESHOLDS,
    min_score: float = DEFAULT_FROC_MIN_SCORE,
    max_detections_per_slice: int = DEFAULT_FROC_MAX_DETECTIONS_PER_SLICE,
) -> Dict[str, float]:
    case_count = max(len({_case_key(target) for target in targets}), 1)
    gt_cases = _target_lesions_by_case(targets, iou_threshold=iou_threshold, min_continuity=1)
    if min_score is None:
        min_score = score_threshold
    min_score = max(float(min_score), 0.0)
    max_detections_per_slice = max(int(max_detections_per_slice), 0)

    case_detections = defaultdict(list)
    detection_scores = []
    for prediction in predictions:
        froc_detections = prediction.get("all_detections")
        if froc_detections is None:
            froc_detections = prediction.get("raw_detections")
        if froc_detections is None:
            froc_detections = prediction.get("detections", [])
        for detection in _filter_froc_candidates(
            froc_detections,
            min_score=min_score,
            max_detections_per_slice=max_detections_per_slice,
        ):
            score = _score_value(detection)
            detection_scores.append(score)
            case_detections[str(detection.get("case_id", "unknown"))].append(detection)

    froc_stats = {}
    for fp in fp_per_case:
        froc_stats[f"froc@{fp:g}"] = 0.0
        froc_stats[f"lesion_track_froc@{fp:g}"] = 0.0
        froc_stats[f"primary_froc@{fp:g}"] = 0.0
        froc_stats[f"primary_lesion_track_froc@{fp:g}"] = 0.0
    if not detection_scores:
        return froc_stats

    score_tensor = torch.unique(torch.tensor(detection_scores, dtype=torch.float32)).sort(descending=True).values
    if max_thresholds and max_thresholds > 0 and score_tensor.numel() > max_thresholds:
        pick = torch.linspace(0, score_tensor.numel() - 1, max_thresholds).round().long()
        thresholds = score_tensor[pick].unique(sorted=True).flip(0)
    else:
        thresholds = score_tensor
    total_gt = _count_lesions(gt_cases)
    best_recall = {fp: 0.0 for fp in fp_per_case}
    best_primary_recall = {fp: 0.0 for fp in fp_per_case}

    for threshold in thresholds:
        threshold_value = float(threshold.item())
        pred_cases = {
            case_id: _summarize_lesion_groups(
                detections,
                score_threshold=threshold_value,
                iou_threshold=iou_threshold,
                min_continuity=min_continuity,
            )
            for case_id, detections in case_detections.items()
        }
        primary_pred_cases = _filter_primary_lesions(pred_cases)
        for primary_only, candidate_cases, tracker in (
            (False, pred_cases, best_recall),
            (True, primary_pred_cases, best_primary_recall),
        ):
            total_tp = total_fp = total_fn = 0
            for case_id in set(candidate_cases) | set(gt_cases):
                kept_predictions = candidate_cases.get(case_id, [])
                gt_lesions = gt_cases.get(case_id, [])
                tp, fp, fn = _match_lesion_tracks(kept_predictions, gt_lesions, iou_threshold=iou_threshold)
                total_tp += tp
                total_fp += fp
                total_fn += fn
            fp_case = total_fp / case_count
            recall = total_tp / max(total_gt, total_tp + total_fn, 1)
            for fp_target in fp_per_case:
                if fp_case <= fp_target:
                    tracker[fp_target] = max(tracker[fp_target], recall)

    for fp_target in fp_per_case:
        froc_stats[f"froc@{fp_target:g}"] = best_recall[fp_target]
        froc_stats[f"lesion_track_froc@{fp_target:g}"] = best_recall[fp_target]
        froc_stats[f"primary_froc@{fp_target:g}"] = best_primary_recall[fp_target]
        froc_stats[f"primary_lesion_track_froc@{fp_target:g}"] = best_primary_recall[fp_target]
    return froc_stats


def _prediction_lesions_by_case(
    predictions: Sequence[dict],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = DEFAULT_ASSOC_IOU_THRESHOLD,
    min_continuity: int = DEFAULT_MIN_CONTINUITY,
    source_key: str = "detections",
) -> Dict[str, List[dict]]:
    case_groups = defaultdict(list)
    for prediction in predictions:
        detections = prediction.get(source_key)
        if detections is None:
            detections = prediction.get("lesions", [])
        for detection in detections:
            case_groups[str(detection.get("case_id", "unknown"))].append(detection)

    return {
        case_id: _summarize_lesion_groups(
            detections,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            min_continuity=min_continuity,
        )
        for case_id, detections in case_groups.items()
    }


def _target_lesions_by_case(
    targets: Sequence[dict],
    iou_threshold: float = DEFAULT_ASSOC_IOU_THRESHOLD,
    min_continuity: int = 1,
) -> Dict[str, List[dict]]:
    case_groups = defaultdict(list)
    for target in targets:
        case_id = _case_key(target)
        slice_index_value = target.get("slice_index", 0)
        if isinstance(slice_index_value, torch.Tensor):
            slice_index_value = slice_index_value.flatten()[0].item()
        boxes = _target_boxes_xyxy(target, device=target["boxes"].device if target.get("boxes") is not None else torch.device("cpu"))
        for box in boxes.detach().cpu():
            case_groups[case_id].append({
                "score": 1.0,
                "box": box,
                "slice_index": int(slice_index_value),
                "case_id": case_id,
            })

    return {
        case_id: _summarize_lesion_groups(
            detections,
            score_threshold=0.0,
            iou_threshold=iou_threshold,
            min_continuity=min_continuity,
        )
        for case_id, detections in case_groups.items()
    }


def _match_lesion_tracks(
    pred_lesions: Sequence[dict],
    gt_lesions: Sequence[dict],
    iou_threshold: float,
    center_distance_threshold: float = DEFAULT_CENTER_DISTANCE_THRESHOLD,
    size_ratio_threshold: float = DEFAULT_SIZE_RATIO_THRESHOLD,
):
    if not pred_lesions or not gt_lesions:
        return 0, len(pred_lesions), len(gt_lesions)
    matched_gt = set()
    true_positive = 0
    for pred in sorted(pred_lesions, key=lambda item: item["score"], reverse=True):
        best_iou = 0.0
        best_gt = None
        for gt_index, gt in enumerate(gt_lesions):
            if gt_index in matched_gt:
                continue
            iou = _track_max_iou(
                pred,
                gt,
                iou_threshold=iou_threshold,
                center_distance_threshold=center_distance_threshold,
                size_ratio_threshold=size_ratio_threshold,
            )
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_index
        if best_iou >= iou_threshold and best_gt is not None:
            matched_gt.add(best_gt)
            true_positive += 1
    false_positive = len(pred_lesions) - true_positive
    false_negative = len(gt_lesions) - true_positive
    return true_positive, false_positive, false_negative


def _track_max_iou(
    pred_lesion: dict,
    gt_lesion: dict,
    iou_threshold: float = DEFAULT_ASSOC_IOU_THRESHOLD,
    center_distance_threshold: float = DEFAULT_CENTER_DISTANCE_THRESHOLD,
    size_ratio_threshold: float = DEFAULT_SIZE_RATIO_THRESHOLD,
) -> float:
    max_iou = 0.0
    gt_by_slice = defaultdict(list)
    for item in gt_lesion.get("items", []):
        gt_by_slice[int(item.get("slice_index", 0))].append(item.get("box"))
    for pred_item in pred_lesion.get("items", []):
        pred_box = pred_item.get("box")
        if pred_box is None:
            continue
        for gt_box in gt_by_slice.get(int(pred_item.get("slice_index", 0)), []):
            if gt_box is None:
                continue
            iou = box_ops.box_iou(pred_box.view(1, 4), gt_box.view(1, 4))[0].item()
            max_iou = max(max_iou, float(iou))
    return max_iou


def compute_slice_metrics(predictions: Sequence[dict], targets: Sequence[dict], score_threshold: float = DEFAULT_SCORE_THRESHOLD) -> Dict[str, float]:
    tp = fp = fn = 0
    for prediction, target in zip(predictions, targets):
        pred_positive = bool((prediction["scores"] >= score_threshold).any().item())
        gt_positive = target.get("boxes") is not None and target["boxes"].numel() > 0
        tp += int(pred_positive and gt_positive)
        fp += int(pred_positive and not gt_positive)
        fn += int((not pred_positive) and gt_positive)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"slice_precision": precision, "slice_recall": recall, "slice_f1": f1}


def _binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return 0.5
    comparisons = (pos[:, None] > neg[None]).float() + 0.5 * (pos[:, None] == neg[None]).float()
    return float(comparisons.mean().item())


def compute_case_metrics(case_scores: Dict[str, float], case_labels: Dict[str, int], threshold: float = DEFAULT_CASE_THRESHOLD) -> Dict[str, float]:
    if not case_scores:
        return {"case_auroc": 0.5, "case_sensitivity": 0.0, "case_specificity": 0.0, "case_mcc": 0.0}
    keys = sorted(case_scores)
    scores = torch.tensor([case_scores[key] for key in keys], dtype=torch.float32)
    labels = torch.tensor([case_labels.get(key, 0) for key in keys], dtype=torch.int64)
    preds = (scores >= threshold).long()
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    denom = max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1)
    mcc = (tp * tn - fp * fn) / (denom ** 0.5)
    return {
        "case_auroc": _binary_auc(scores, labels),
        "case_sensitivity": sensitivity,
        "case_specificity": specificity,
        "case_mcc": mcc,
    }


def compute_negative_case_metrics(
    case_summaries: Dict[str, Dict[str, float]],
    case_labels: Dict[str, int],
    threshold: float = DEFAULT_CASE_THRESHOLD,
) -> Dict[str, float]:
    negative_cases = [case_id for case_id, label in case_labels.items() if label == 0]
    if not negative_cases:
        return {
            "negative_case_count": 0.0,
            "negative_case_fp_per_case": 0.0,
            "negative_case_primary_fp_per_case": 0.0,
            "negative_case_fp_rate": 0.0,
        }
    false_lesions = 0.0
    false_primary_lesions = 0.0
    false_positive_cases = 0
    for case_id in negative_cases:
        summary = case_summaries.get(case_id, {})
        false_lesions += float(summary.get("lesion_count", 0.0))
        false_primary_lesions += float(summary.get("primary_lesion_count", summary.get("lesion_count", 0.0)))
        false_positive_cases += int(float(summary.get("case_score", 0.0)) >= threshold)
    return {
        "negative_case_count": float(len(negative_cases)),
        "negative_case_fp_per_case": false_lesions / max(len(negative_cases), 1),
        "negative_case_primary_fp_per_case": false_primary_lesions / max(len(negative_cases), 1),
        "negative_case_fp_rate": false_positive_cases / max(len(negative_cases), 1),
    }


class FractureMetricEvaluator(object):
    def __init__(
        self,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        iou_threshold: float = DEFAULT_ASSOC_IOU_THRESHOLD,
        min_continuity: int = DEFAULT_MIN_CONTINUITY,
        topk_lesions: int = DEFAULT_TOPK_LESIONS,
        case_alpha: float = DEFAULT_CASE_ALPHA,
        case_beta: float = DEFAULT_CASE_BETA,
        continuity_norm: float = DEFAULT_CONTINUITY_NORM,
        case_threshold: float = DEFAULT_CASE_THRESHOLD,
        max_froc_thresholds: int = DEFAULT_MAX_FROC_THRESHOLDS,
        compute_froc_metrics: bool = True,
        froc_min_score: float = DEFAULT_FROC_MIN_SCORE,
        froc_max_detections_per_slice: int = DEFAULT_FROC_MAX_DETECTIONS_PER_SLICE,
    ):
        self.predictions: List[dict] = []
        self.targets: List[dict] = []
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.min_continuity = min_continuity
        self.topk_lesions = topk_lesions
        self.case_alpha = case_alpha
        self.case_beta = case_beta
        self.continuity_norm = continuity_norm
        self.case_threshold = case_threshold
        self.max_froc_thresholds = max_froc_thresholds
        self.compute_froc_metrics = compute_froc_metrics
        self.froc_min_score = froc_min_score
        self.froc_max_detections_per_slice = froc_max_detections_per_slice

    def update(self, predictions: Sequence[dict], targets: Sequence[dict]):
        self.predictions.extend(_detach_to_cpu(prediction) for prediction in predictions)
        self.targets.extend(_detach_to_cpu(target) for target in targets)

    def synchronize_between_processes(self):
        gathered = all_gather({
            "predictions": self.predictions,
            "targets": self.targets,
        })
        merged = {}
        for payload in gathered:
            for prediction, target in zip(payload.get("predictions", []), payload.get("targets", [])):
                merged.setdefault(_image_key(target), (prediction, target))
        if not merged:
            return
        ordered = sorted(merged.values(), key=lambda item: (_case_key(item[1]), int(item[1].get("slice_index", 0))))
        self.predictions = [prediction for prediction, _ in ordered]
        self.targets = [target for _, target in ordered]

    def summarize(self) -> Dict[str, float]:
        stats = {}
        stats.update(compute_slice_metrics(self.predictions, self.targets, score_threshold=self.score_threshold))
        stats.update(compute_lesion_metrics(
            self.predictions,
            self.targets,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold,
        ))
        if self.compute_froc_metrics:
            stats.update(compute_froc(
                self.predictions,
                self.targets,
                iou_threshold=self.iou_threshold,
                score_threshold=self.score_threshold,
                min_continuity=self.min_continuity,
                max_thresholds=self.max_froc_thresholds,
                min_score=self.froc_min_score,
                max_detections_per_slice=self.froc_max_detections_per_slice,
            ))

        case_groups = defaultdict(list)
        case_labels = defaultdict(int)
        for prediction, target in zip(self.predictions, self.targets):
            case_id = _case_key(target)
            case_key = str(case_id)
            detections = prediction.get("detections")
            if detections is None:
                detections = prediction.get("lesions", [])
                if detections and "items" in detections[0]:
                    flattened = []
                    for lesion in detections:
                        flattened.extend(lesion.get("items", []))
                    detections = flattened
            case_groups[case_key].extend(detections)
            target_case_label = target.get("case_label")
            if isinstance(target_case_label, torch.Tensor):
                target_case_label = int(target_case_label.detach().cpu().view(-1)[0].item())
            elif target_case_label is None:
                target_case_label = int(target.get("boxes") is not None and target["boxes"].numel() > 0)
            else:
                target_case_label = int(target_case_label)
            case_labels[case_key] = max(case_labels[case_key], target_case_label)
            case_groups.setdefault(case_key, [])

        if case_labels:
            case_summaries = {
                key: aggregate_case_predictions(
                    detections,
                    alpha=self.case_alpha,
                    beta=self.case_beta,
                    topk=self.topk_lesions,
                    score_threshold=self.score_threshold,
                    iou_threshold=self.iou_threshold,
                    min_continuity=self.min_continuity,
                    continuity_norm=self.continuity_norm,
                )
                for key, detections in case_groups.items()
            }
            case_scores = {key: summary["case_score"] for key, summary in case_summaries.items()}
            stats["case_score_mean"] = float(sum(case_scores.values()) / max(len(case_scores), 1))
            stats.update(compute_case_metrics(case_scores, case_labels, threshold=self.case_threshold))
            stats.update(compute_negative_case_metrics(case_summaries, case_labels, threshold=self.case_threshold))
        else:
            stats["case_score_mean"] = 0.0
            stats.update(compute_case_metrics({}, {}))
            stats.update(compute_negative_case_metrics({}, {}))
        return stats


class CaseLevelAggregator(object):
    """Stateful test-time case aggregator shared by evaluation and export code."""

    def __init__(
        self,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        iou_threshold: float = DEFAULT_ASSOC_IOU_THRESHOLD,
        min_continuity: int = DEFAULT_MIN_CONTINUITY,
        topk_lesions: int = DEFAULT_TOPK_LESIONS,
        case_alpha: float = DEFAULT_CASE_ALPHA,
        case_beta: float = DEFAULT_CASE_BETA,
        continuity_norm: float = DEFAULT_CONTINUITY_NORM,
        case_threshold: float = DEFAULT_CASE_THRESHOLD,
    ):
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.min_continuity = min_continuity
        self.topk_lesions = topk_lesions
        self.case_alpha = case_alpha
        self.case_beta = case_beta
        self.continuity_norm = continuity_norm
        self.case_threshold = case_threshold
        self.case_detections: Dict[str, List[dict]] = defaultdict(list)

    def update(self, prediction: dict):
        for detection in prediction.get("detections", []):
            case_id = str(detection.get("case_id", "unknown"))
            self.case_detections[case_id].append(_detach_to_cpu(detection))

    def update_many(self, predictions: Sequence[dict]):
        for prediction in predictions:
            self.update(prediction)

    def summarize_cases(self) -> Dict[str, Dict[str, float]]:
        summaries = {}
        for case_id, detections in self.case_detections.items():
            summary = aggregate_case_predictions(
                detections,
                alpha=self.case_alpha,
                beta=self.case_beta,
                topk=self.topk_lesions,
                score_threshold=self.score_threshold,
                iou_threshold=self.iou_threshold,
                min_continuity=self.min_continuity,
                continuity_norm=self.continuity_norm,
            )
            summary["case_pred"] = float(summary["case_score"] >= self.case_threshold)
            summaries[case_id] = summary
        return summaries
