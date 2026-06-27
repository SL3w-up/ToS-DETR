# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from util.fracture_eval import FractureMetricEvaluator
from util.fracture_ops import move_target_to_device


EVAL_LOSSES = ('labels', 'boxes', 'centers', 'heatmap', 'region', 'cardinality')
CORE_FRACTURE_METRIC_KEYS = {
    'slice_precision',
    'slice_recall',
    'slice_f1',
    'primary_lesion_track_precision',
    'primary_lesion_track_recall',
    'primary_lesion_track_froc@1',
    'case_score_mean',
    'case_auroc',
    'case_sensitivity',
    'case_specificity',
    'case_mcc',
    'negative_case_count',
    'negative_case_primary_fp_per_case',
    'negative_case_fp_per_case',
    'negative_case_fp_rate',
}


def _core_fracture_metric_keys(args):
    keys = set(CORE_FRACTURE_METRIC_KEYS)
    if args is not None:
        for attr in ('checkpoint_fp_key', 'checkpoint_all_fp_key',
                     'checkpoint_recall_key', 'froc_primary_key'):
            value = getattr(args, attr, None)
            if value:
                keys.add(value)
    return keys


class _temporary_criterion_losses(object):
    def __init__(self, criterion, losses):
        self.criterion = criterion
        self.losses = list(losses)
        self.previous_losses = None

    def __enter__(self):
        if self.criterion is None or not hasattr(self.criterion, "losses"):
            return
        self.previous_losses = list(self.criterion.losses)
        self.criterion.losses = list(self.losses)

    def __exit__(self, exc_type, exc, traceback):
        if self.previous_losses is not None:
            self.criterion.losses = self.previous_losses
        return False


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples, targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced.get('class_error', 0.0))
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    compute_coco_metrics = args is None or getattr(args, "compute_coco_metrics", True)
    coco_evaluator = CocoEvaluator(base_ds, iou_types) if compute_coco_metrics and iou_types else None
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    fracture_evaluator = None
    if args is not None and getattr(args, "compute_fracture_metrics", False):
        fracture_evaluator = FractureMetricEvaluator(
            score_threshold=args.fracture_score_threshold,
            iou_threshold=args.case_association_iou,
            min_continuity=args.case_min_continuity,
            topk_lesions=args.case_topk_lesions,
            case_alpha=args.case_alpha,
            case_beta=args.case_beta,
            continuity_norm=args.case_continuity_norm,
            case_threshold=args.case_threshold,
            max_froc_thresholds=args.max_froc_thresholds,
            compute_froc_metrics=getattr(args, "compute_froc_metrics", True),
            froc_min_score=getattr(args, "froc_min_score", 0.05),
            froc_max_detections_per_slice=getattr(args, "froc_max_detections_per_slice", 0),
        )

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    if args is not None and getattr(args, "eval_losses", None):
        eval_losses = tuple(args.eval_losses)
    elif hasattr(criterion, "get_active_losses"):
        eval_losses = tuple(criterion.get_active_losses())
    else:
        eval_losses = EVAL_LOSSES
    with _temporary_criterion_losses(criterion, eval_losses):
        for samples, targets in metric_logger.log_every(data_loader, 10, header):
            samples = samples.to(device)
            targets = [{k: move_target_to_device(v, device) for k, v in t.items()} for t in targets]

            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            stage_loss = sum(loss_dict_reduced_scaled.values())
            metric_updates = {
                'loss': stage_loss,
                'stage_loss': stage_loss,
                **loss_dict_reduced_scaled,
            }
            core_loss_terms = [
                value for key, value in loss_dict_reduced_scaled.items()
                if key.startswith(('loss_ce', 'loss_bbox', 'loss_giou', 'loss_wise_iou', 'loss_center'))
            ]
            if core_loss_terms:
                metric_updates['core_loss'] = sum(core_loss_terms)
            if args is None or getattr(args, "verbose_eval_losses", False):
                metric_updates.update({f'{k}_unscaled': v
                                       for k, v in loss_dict_reduced.items()})
            metric_logger.update(**metric_updates)
            metric_logger.update(class_error=loss_dict_reduced.get('class_error', 0.0))

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            if coco_evaluator is not None:
                results = postprocessors['bbox'](outputs, orig_target_sizes)
                if 'segm' in postprocessors.keys():
                    target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                    results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
                res = {target['image_id'].item(): output for target, output in zip(targets, results)}
                coco_evaluator.update(res)
            if fracture_evaluator is not None and 'fracture' in postprocessors:
                fracture_predictions = postprocessors['fracture'](outputs, orig_target_sizes, targets)
                fracture_evaluator.update(fracture_predictions, targets)

            if panoptic_evaluator is not None:
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
                for i, target in enumerate(targets):
                    image_id = target["image_id"].item()
                    file_name = f"{image_id:012d}.png"
                    res_pano[i]["image_id"] = image_id
                    res_pano[i]["file_name"] = file_name

                panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()
    if fracture_evaluator is not None:
        fracture_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    if fracture_evaluator is not None:
        fracture_stats = fracture_evaluator.summarize()
        if args is not None and not getattr(args, "verbose_fracture_metrics", False):
            core_keys = _core_fracture_metric_keys(args)
            fracture_stats = {
                key: value for key, value in fracture_stats.items()
                if key in core_keys
            }
        stats.update(fracture_stats)
    return stats, coco_evaluator


@torch.no_grad()
def mine_dynamic_hard_negatives(model, postprocessors, data_loader, device, args):
    """Mine no-GT slices with high-confidence predictions for next-epoch HN sampling."""
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Mine HN:'
    local_candidates = []
    examined_negative = 0

    for samples, targets in metric_logger.log_every(data_loader, 20, header):
        samples = samples.to(device)
        targets = [{k: move_target_to_device(v, device) for k, v in t.items()} for t in targets]
        outputs = model(samples, targets)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        predictions = postprocessors['fracture'](outputs, orig_target_sizes, targets)

        for prediction, target in zip(predictions, targets):
            has_gt = target.get("boxes") is not None and target["boxes"].numel() > 0
            if has_gt:
                continue
            examined_negative += 1
            scores = prediction.get("scores")
            if scores is None or scores.numel() == 0:
                continue
            max_score = float(scores.max().detach().cpu().item())
            if max_score >= args.dynamic_hn_score_threshold:
                image_id = int(target["image_id"].flatten()[0].detach().cpu().item())
                local_candidates.append((max_score, image_id))

    gathered = utils.all_gather({
        "candidates": local_candidates,
        "examined_negative": examined_negative,
    })
    merged = {}
    examined_total = 0
    for rank_payload in gathered:
        examined_total += int(rank_payload.get("examined_negative", 0))
        for score, image_id in rank_payload.get("candidates", []):
            merged[int(image_id)] = max(float(score), merged.get(int(image_id), 0.0))

    ranked = sorted(merged.items(), key=lambda item: item[1], reverse=True)
    if args.dynamic_hn_max_slices and args.dynamic_hn_max_slices > 0:
        ranked = ranked[:args.dynamic_hn_max_slices]
    hard_negative_ids = [image_id for image_id, _ in ranked]
    stats = {
        "dynamic_hn_examined_negative_slices": float(examined_total),
        "dynamic_hn_candidate_slices": float(len(merged)),
        "dynamic_hn_selected_slices": float(len(hard_negative_ids)),
        "dynamic_hn_max_score": float(ranked[0][1]) if ranked else 0.0,
    }
    return hard_negative_ids, stats
