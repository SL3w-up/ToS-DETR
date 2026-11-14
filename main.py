# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PRETRAINED_MODEL = os.environ.get("TOS_DETR_PRETRAINED_MODEL", "")
DEFAULT_BACKBONE_PRETRAINED = os.environ.get("TOS_DETR_BACKBONE_PRETRAINED", "")
DEFAULT_COCO_PATH = os.environ.get("TOS_DETR_COCO_PATH", str(PROJECT_ROOT / "data" / "coco"))

# Reference schedule of the released checkpoint: 5-epoch heatmap warm-up followed
# by full detector training under a 43-epoch cosine LR schedule (5 warmup epochs,
# min_lr_factor 0.05). The best medical checkpoint was selected from this run by
# the checkpoint-selection rule below; the schedule length itself is 43 epochs.
TWO_STAGE_SCHEDULE_DEFAULT_STAGE1_EPOCHS = 5
DEFAULT_TOTAL_EPOCHS = 43


def load_checkpoint(checkpoint_path, allow_unsafe=False):
    if checkpoint_path.startswith('https'):
        return torch.hub.load_state_dict_from_url(
            checkpoint_path, map_location='cpu', check_hash=True)
    try:
        from torch.serialization import safe_globals
        with safe_globals([argparse.Namespace]):
            return torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except TypeError:
        if allow_unsafe:
            return torch.load(checkpoint_path, map_location='cpu')
        raise RuntimeError(
            "This PyTorch version cannot safely load checkpoints with weights_only=True. "
            "Use --allow_unsafe_checkpoint only for trusted local checkpoints."
        )
    except Exception as exc:
        if allow_unsafe and "Weights only load failed" in str(exc):
            return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        raise RuntimeError(
            f"Safe checkpoint load failed for {checkpoint_path}. "
            "Use --allow_unsafe_checkpoint only for trusted local checkpoints."
        ) from exc


def adapt_transformer_pretrained_tensor(key, value, target):
    if key == 'transformer.level_embed' and value.ndim == 2 and target.ndim == 2:
        if value.shape[0] >= target.shape[0] and value.shape[1] == target.shape[1]:
            return value[:target.shape[0]].clone()

    old_levels = 4
    new_levels = 3
    num_heads = 8
    num_points = 4
    if 'sampling_offsets.weight' in key and value.ndim == 2 and value.shape[1] == target.shape[1]:
        if value.shape[0] == num_heads * old_levels * num_points * 2 and target.shape[0] == num_heads * new_levels * num_points * 2:
            return value.view(num_heads, old_levels, num_points, 2, value.shape[1])[:, :new_levels].reshape_as(target).clone()
    if 'sampling_offsets.bias' in key and value.ndim == 1:
        if value.shape[0] == num_heads * old_levels * num_points * 2 and target.shape[0] == num_heads * new_levels * num_points * 2:
            return value.view(num_heads, old_levels, num_points, 2)[:, :new_levels].reshape_as(target).clone()
    if 'attention_weights.weight' in key and value.ndim == 2 and value.shape[1] == target.shape[1]:
        if value.shape[0] == num_heads * old_levels * num_points and target.shape[0] == num_heads * new_levels * num_points:
            return value.view(num_heads, old_levels, num_points, value.shape[1])[:, :new_levels].reshape_as(target).clone()
    if 'attention_weights.bias' in key and value.ndim == 1:
        if value.shape[0] == num_heads * old_levels * num_points and target.shape[0] == num_heads * new_levels * num_points:
            return value.view(num_heads, old_levels, num_points)[:, :new_levels].reshape_as(target).clone()
    return None


def load_shape_compatible_pretrained(model, checkpoint_path, include_keywords=None, allow_unsafe=False):
    checkpoint = load_checkpoint(checkpoint_path, allow_unsafe=allow_unsafe)
    state_dict = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_state = model.state_dict()
    include_keywords = include_keywords or []
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if include_keywords and not any(keyword in key for keyword in include_keywords):
            skipped.append(key)
            continue
        if key not in model_state:
            skipped.append(key)
            continue
        target = model_state[key]
        if tuple(target.shape) == tuple(value.shape):
            compatible[key] = value
            continue
        adapted = adapt_transformer_pretrained_tensor(key, value, target)
        if adapted is None:
            skipped.append(key)
            continue
        compatible[key] = adapted
    missing_keys, unexpected_keys = model.load_state_dict(compatible, strict=False)
    print(
        f"Loaded {len(compatible)} shape-compatible pretrained tensors from {checkpoint_path}; "
        f"skipped {len(skipped)} filtered/incompatible/unmatched tensors."
    )
    if missing_keys:
        print(f"Pretrained missing keys after partial load: {missing_keys[:50]}")
    if unexpected_keys:
        print(f"Pretrained unexpected keys after partial load: {unexpected_keys[:50]}")
    return compatible, skipped


def build_lr_scheduler(optimizer, args):
    if args.lr_scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    warmup_epochs = max(int(args.warmup_epochs), 0)
    # --cosine_horizon decouples the LR curve length from --epochs so partial
    # runs still follow the reference 43-epoch cosine schedule.
    total_epochs = max(int(args.cosine_horizon), 1)
    min_factor = float(args.min_lr_factor)

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        cosine_span = max(total_epochs - warmup_epochs, 1)
        progress = min(max(epoch - warmup_epochs, 0) / cosine_span, 1.0)
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_epoch_training_stage(epoch, args):
    if epoch < args.stage1_epochs:
        return 'stage1_heatmap'
    return 'stage2_full'


def get_stage_sample_type_ratios(stage, args):
    return list(args.sample_type_ratios)


def _set_requires_grad_by_name(model, enabled_keywords, trainable=True):
    for name, parameter in model.named_parameters():
        if any(keyword in name for keyword in enabled_keywords):
            parameter.requires_grad_(trainable)


def configure_model_for_stage(model_without_ddp, criterion, optimizer, stage, args):
    """Apply stage behavior: heatmap warm-up freezes everything but the heatmap path."""
    for parameter in model_without_ddp.parameters():
        parameter.requires_grad_(True)

    if stage == 'stage1_heatmap':
        for parameter in model_without_ddp.parameters():
            parameter.requires_grad_(False)
        _set_requires_grad_by_name(
            model_without_ddp,
            [
                'backbone',
                'input_proj',
                'slice_embedding',
                'fracture_heatmap',
            ],
            trainable=True,
        )

    criterion.set_training_stage(stage, include_masks=args.masks)

    trainable_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print(
        f"Training stage: {stage}; active_losses={criterion.get_active_losses()}; "
        f"trainable_params={trainable_params}"
    )
    return trainable_params


def build_checkpoint_payload(
    model_without_ddp,
    optimizer,
    lr_scheduler,
    epoch,
    args,
    best_medical_recall,
    best_medical_fp,
    best_medical_froc,
    best_fallback_recall,
    best_fallback_fp,
    best_fallback_froc,
    extra=None,
):
    payload = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'args': args,
        'best_medical_recall': best_medical_recall,
        'best_medical_fp': best_medical_fp,
        'best_medical_froc': best_medical_froc,
        'best_fallback_recall': best_fallback_recall,
        'best_fallback_fp': best_fallback_fp,
        'best_fallback_froc': best_fallback_froc,
    }
    if extra:
        payload.update(extra)
    return payload


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=DEFAULT_TOTAL_EPOCHS, type=int)
    parser.add_argument('--lr_scheduler', default='cosine', choices=('cosine', 'step'),
                        help='Use warmup+cosine by default; step keeps the original Deformable DETR schedule')
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--min_lr_factor', default=0.05, type=float,
                        help='Final lr as a fraction of the initial lr for cosine schedule')
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--fracture_single_class', dest='fracture_single_class', action='store_true')
    parser.add_argument('--multi_class_labels', dest='fracture_single_class', action='store_false')
    parser.set_defaults(fracture_single_class=True)

    # * Backbone
    parser.add_argument('--backbone', default='convnext_tiny', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--backbone_pretrained', dest='backbone_pretrained', action='store_true')
    parser.add_argument('--no_backbone_pretrained', dest='backbone_pretrained', action='store_false',
                        help='Disable timm/torchvision backbone weight download; useful on offline machines')
    parser.set_defaults(backbone_pretrained=True)
    parser.add_argument('--backbone_pretrained_path', default=DEFAULT_BACKBONE_PRETRAINED, type=str,
                        help='Local timm backbone checkpoint file; avoids network download when set')
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=3, type=int, help='number of feature levels')
    parser.add_argument('--slice_embedding_scale', default=0.02, type=float)
    parser.add_argument('--edge_attention_strength', default=0.25, type=float)
    parser.add_argument('--organ_mask_input_strength', default=0.20, type=float,
                        help='Softly boost image intensities inside the 2.5D organ mask before the backbone')
    parser.add_argument('--organ_feature_gating_strength', default=1.0, type=float,
                        help='Feature-level organ ROI boost for P2/P3/P4')
    parser.add_argument('--organ_feature_outside_scale', default=0.25, type=float,
                        help='Retained feature scale outside the organ ROI; nonzero keeps context')
    parser.add_argument('--organ_mask_dilation', default=5, type=int,
                        help='Pixel dilation radius for organ ROI gating/query/matching')
    parser.add_argument('--dynamic_query_fraction', default=0.70, type=float,
                        help='Fraction of query slots initialized from ROI-aware heatmap/coverage points')
    parser.add_argument('--coverage_query_fraction', default=0.20, type=float,
                        help='Fraction of query slots used as uniform ROI coverage queries')
    parser.add_argument('--query_region_floor', default=0.02, type=float,
                        help='Minimum heatmap sampling score outside the ROI to avoid empty-query collapse')
    parser.add_argument('--use_slice_memory', dest='use_slice_memory', action='store_true')
    parser.add_argument('--no_use_slice_memory', dest='use_slice_memory', action='store_false')
    parser.set_defaults(use_slice_memory=True)
    parser.add_argument('--dynamic_queries', dest='dynamic_queries', action='store_true')
    parser.add_argument('--no_dynamic_queries', dest='dynamic_queries', action='store_false')
    parser.set_defaults(dynamic_queries=True)

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=50, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_center', default=2, type=float,
                        help="center point coefficient in the matching cost")
    parser.add_argument('--set_cost_region', default=1, type=float,
                        help="organ-mask region prior coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--wise_iou_loss_coef', default=0, type=float)
    parser.add_argument('--iou_loss_type', default='giou', choices=('giou', 'diou', 'ciou', 'eiou'),
                        help='IoU-family loss used by loss_wise_iou; default keeps original GIoU behavior')
    parser.add_argument('--center_loss_coef', default=2, type=float)
    parser.add_argument('--heatmap_loss_coef', default=1, type=float)
    parser.add_argument('--region_loss_coef', default=1.0, type=float)
    parser.add_argument('--query_coverage_loss_coef', default=0.5, type=float)
    parser.add_argument('--slice_memory_loss_coef', default=0.2, type=float)
    parser.add_argument('--heatmap_sigma', default=2.0, type=float,
                        help='Gaussian sigma for small-fracture heatmap targets')
    parser.add_argument('--query_coverage_radius', default=0.08, type=float,
                        help='Normalized radius within which at least one query should cover each GT center')
    parser.add_argument('--consistency_loss_coef', default=0.5, type=float)
    parser.add_argument('--contrastive_loss_coef', default=0.1, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default=DEFAULT_COCO_PATH, type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--input_height', default=448, type=int)
    parser.add_argument('--input_width', default=224, type=int)
    parser.add_argument('--pixel_mean', default=[0.1325491989, 0.1326287777, 0.1327065940], nargs=3, type=float)
    parser.add_argument('--pixel_std', default=[0.1983262042, 0.1984989205, 0.1986687937], nargs=3, type=float)
    parser.add_argument('--rotation_prob', default=0.3, type=float)
    parser.add_argument('--rotation_degrees', default=5.0, type=float)
    parser.add_argument('--elastic_prob', default=0.15, type=float)
    parser.add_argument('--elastic_alpha', default=2.0, type=float)
    parser.add_argument('--elastic_sigma', default=6.0, type=float)
    parser.add_argument('--elastic_safe_mode', action='store_true', default=True)
    parser.add_argument('--roi_copy_paste_prob', default=0.15, type=float)
    parser.add_argument('--roi_copy_paste_iou_thresh', default=0.1, type=float)
    parser.add_argument('--roi_copy_paste_max_translate', default=0.08, type=float)
    parser.add_argument('--roi_copy_paste_min_organ_overlap', default=0.5, type=float,
                        help='Minimum target-region organ-mask overlap for fracture ROI copy-paste')
    parser.add_argument('--organ_mask_root', default='auto', type=str,
                        help='Root of 2.5D organ masks; auto uses coco/anatomy_masks when present')
    parser.add_argument('--organ_mask_suffixes', default=["", "_mask", "_seg", "-mask", "-seg"], nargs='*', type=str)
    parser.add_argument('--organ_mask_extensions', default=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy", ".pt", ".pth"], nargs='*', type=str)
    parser.add_argument('--organ_mask_missing_as_ones', dest='organ_mask_missing_as_ones', action='store_true')
    parser.add_argument('--organ_mask_missing_as_zeros', dest='organ_mask_missing_as_ones', action='store_false')
    parser.set_defaults(organ_mask_missing_as_ones=True)
    parser.add_argument('--strict_organ_masks', dest='strict_organ_masks', action='store_true',
                        help='Raise an error instead of using a fallback ROI when an organ mask is missing')
    parser.add_argument('--no_strict_organ_masks', dest='strict_organ_masks', action='store_false',
                        help='Allow all-ones/all-zeros fallback ROI masks when masks are missing')
    parser.set_defaults(strict_organ_masks=True)
    parser.add_argument('--organ_mask_missing_warning_limit', default=10, type=int,
                        help='Maximum missing-organ-mask fallback warnings printed per dataset worker')
    parser.add_argument('--organ_mask_validate_max_samples', default=2048, type=int,
                        help='Number of samples checked at dataset construction when strict organ masks are enabled; 0 checks all')
    parser.add_argument('--organ_mask_recursive', action='store_true',
                        help='Build a recursive stem index under organ_mask_root for non-standard mask layouts')
    parser.add_argument('--require_case_metadata', dest='require_case_metadata', action='store_true',
                        help='Fail fast when case_id or slice_index is missing from COCO image metadata')
    parser.add_argument('--no_require_case_metadata', dest='require_case_metadata', action='store_false')
    parser.set_defaults(require_case_metadata=True)
    parser.add_argument('--disable_mosaic', action='store_true', default=True)
    parser.add_argument('--disable_mixup', action='store_true', default=True)
    parser.add_argument('--disable_cutmix', action='store_true', default=True)
    parser.add_argument('--sample_type_ratios', default=[0.40, 0.40, 0.20], nargs=3, type=float,
                        help='P/PN/N sampling ratios for case-aware slice sampling')
    parser.add_argument('--pn_slice_window', default=5, type=int,
                        help='Negative slices within this distance from a positive slice in the same case are PN-slices')
    parser.add_argument('--sample_type_sampling', dest='sample_type_sampling', action='store_true')
    parser.add_argument('--no_sample_type_sampling', dest='sample_type_sampling', action='store_false')
    parser.set_defaults(sample_type_sampling=True)
    parser.add_argument('--case_aware_sampling', dest='case_aware_sampling', action='store_true')
    parser.add_argument('--no_case_aware_sampling', dest='case_aware_sampling', action='store_false')
    parser.set_defaults(case_aware_sampling=True)
    parser.add_argument('--case_aware_neighbor_window', default=5, type=int)
    parser.add_argument('--case_aware_same_case_fraction', default=0.5, type=float)
    parser.add_argument('--case_sequential_eval', dest='case_sequential_eval', action='store_true',
                        help='Evaluate/test with each mini-batch containing sorted slices from one case only')
    parser.add_argument('--no_case_sequential_eval', dest='case_sequential_eval', action='store_false')
    parser.set_defaults(case_sequential_eval=True)
    parser.add_argument('--stage1_epochs', default=TWO_STAGE_SCHEDULE_DEFAULT_STAGE1_EPOCHS, type=int,
                        help='Heatmap warm-up epochs before full detector training begins')
    parser.add_argument('--cosine_horizon', default=DEFAULT_TOTAL_EPOCHS, type=int,
                        help='Total epoch count used by the warmup+cosine LR schedule; '
                             'decoupled from --epochs so partial runs follow the reference curve')
    parser.add_argument('--region_prior_weight', default=1.0, type=float,
                        help='organ-mask region prior embedding weight in the transformer encoder')
    parser.add_argument('--slice_memory_alpha', default=1.0, type=float,
                        help='center-slice weight for adjacent-slice feature memory')
    parser.add_argument('--slice_memory_beta', default=0.5, type=float,
                        help='previous-slice weight for adjacent-slice feature memory')
    parser.add_argument('--slice_memory_gamma', default=0.5, type=float,
                        help='next-slice weight for adjacent-slice feature memory')
    parser.add_argument('--query_memory_alpha', dest='slice_memory_alpha', type=float,
                        help=argparse.SUPPRESS)
    parser.add_argument('--query_memory_beta', dest='slice_memory_beta', type=float,
                        help=argparse.SUPPRESS)
    parser.add_argument('--query_memory_gamma', dest='slice_memory_gamma', type=float,
                        help=argparse.SUPPRESS)
    parser.add_argument('--fracture_score_threshold', default=0.30, type=float,
                        help='Slice detection score threshold for fracture metrics and case aggregation')
    parser.add_argument('--case_association_iou', default=0.20, type=float,
                        help='IoU threshold for associating adjacent-slice boxes into one lesion track')
    parser.add_argument('--case_min_continuity', default=2, type=int,
                        help='Number of unique adjacent slices required to mark a lesion track as continuous')
    parser.add_argument('--case_topk_lesions', default=5, type=int,
                        help='Number of lesion tracks used for case-level score aggregation')
    parser.add_argument('--case_alpha', default=0.7, type=float)
    parser.add_argument('--case_beta', default=0.3, type=float)
    parser.add_argument('--case_continuity_norm', default=5.0, type=float,
                        help='Continuity length used to normalize log(1 + length) into [0, 1]')
    parser.add_argument('--case_threshold', default=0.50, type=float,
                        help='Threshold for case-level sensitivity/specificity and negative-case FP rate')
    parser.add_argument('--postprocess_organ_score_floor', default=0.05, type=float,
                        help='Minimum score multiplier for detections outside organ mask')
    parser.add_argument('--fracture_max_detections_per_slice', default=20, type=int,
                        help='Maximum thresholded fracture candidates kept per slice for lesion/FROC/case metrics; 0 keeps all')
    parser.add_argument('--froc_min_score', default=0.05, type=float,
                        help='Minimum raw detection score kept before FROC threshold scanning')
    parser.add_argument('--froc_max_detections_per_slice', default=20, type=int,
                        help='Maximum score-filtered detections kept per slice for FROC only; 0 keeps all')
    parser.add_argument('--max_froc_thresholds', default=256, type=int,
                        help='Maximum unique score thresholds swept for FROC after score filtering; 0 sweeps all thresholds')
    parser.add_argument('--compute_froc_metrics', dest='compute_froc_metrics', action='store_true',
                        help='Compute FROC threshold sweep in the current evaluation pass')
    parser.add_argument('--no_compute_froc_metrics', dest='compute_froc_metrics', action='store_false')
    parser.set_defaults(compute_froc_metrics=True)
    parser.add_argument('--verbose_eval_losses', action='store_true',
                        help='Log unscaled validation losses; disabled by default to reduce validation overhead/noise')
    parser.add_argument('--verbose_fracture_metrics', action='store_true',
                        help='Keep all fracture metrics in validation logs; disabled by default to keep only selection/core metrics')
    parser.add_argument('--compute_fracture_metrics', dest='compute_fracture_metrics', action='store_true')
    parser.add_argument('--no_compute_fracture_metrics', dest='compute_fracture_metrics', action='store_false')
    parser.set_defaults(compute_fracture_metrics=True)
    parser.add_argument('--compute_coco_metrics', dest='compute_coco_metrics', action='store_true',
                        help='Also compute COCO AP during validation; disabled by default for medical FROC selection')
    parser.add_argument('--no_compute_coco_metrics', dest='compute_coco_metrics', action='store_false')
    parser.set_defaults(compute_coco_metrics=False)
    parser.add_argument('--froc_primary_key', default='primary_lesion_track_froc@1', type=str,
                        help='FROC key logged and used as a tie-breaker for medical checkpoint selection')
    parser.add_argument('--checkpoint_fp_key', default='negative_case_primary_fp_per_case', type=str,
                        help='Negative-case FP metric constrained during medical checkpoint selection')
    parser.add_argument('--checkpoint_fp_neg_case_max', default=0.50, type=float,
                        help='Maximum allowed negative-case FP/case for medical checkpoint selection')
    parser.add_argument('--checkpoint_all_fp_key', default='negative_case_fp_per_case', type=str,
                        help='Secondary all-lesion FP/case constraint; set max <= 0 to disable')
    parser.add_argument('--checkpoint_all_fp_neg_case_max', default=2.0, type=float,
                        help='Maximum all-lesion FP/case allowed so isolated single-slice FP cannot dominate')
    parser.add_argument('--checkpoint_recall_key', default='primary_lesion_track_recall', type=str,
                        help='Recall metric maximized after the negative-case FP/case constraint is satisfied')
    parser.add_argument('--checkpoint_tie_eps', default=1e-6, type=float,
                        help='Tolerance for treating checkpoint recall values as tied before FP/FROC tie-breaks')
    parser.add_argument('--pretrained_model', default=DEFAULT_PRETRAINED_MODEL, type=str,
                        help='Optional checkpoint for shape-compatible warm start before training; unlike --resume it does not load optimizer state')
    parser.add_argument('--pretrained_include_keywords', default=['transformer'], type=str, nargs='+',
                        help='Only warm-start checkpoint keys containing one of these keywords')
    parser.add_argument('--allow_unsafe_checkpoint', action='store_true',
                        help='Allow unsafe pickle checkpoint loading only for trusted local checkpoints')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--validate_on_resume', dest='validate_on_resume', action='store_true',
                        help='Run a full validation pass immediately after loading --resume')
    parser.add_argument('--no_validate_on_resume', dest='validate_on_resume', action='store_false')
    parser.set_defaults(validate_on_resume=False)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        raise NotImplementedError(
            "--frozen_weights is the original segmentation-only DETR path and is not "
            "compatible with this fracture detector. Use --pretrained_model for warm start."
        )
    if args.masks:
        raise NotImplementedError(
            "--masks enables the original DETRsegm path. This project uses organ ROI masks "
            "through --organ_mask_root and does not support training a segmentation head."
        )
    if not args.eval:
        if not args.compute_fracture_metrics:
            raise ValueError(
                "Training requires --compute_fracture_metrics so every epoch can produce "
                "medical checkpoint-selection metrics."
            )
        if not args.compute_froc_metrics:
            raise ValueError(
                "Training requires --compute_froc_metrics because FROC is part of the "
                "medical checkpoint-selection tie-break."
            )
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    use_sample_type_sampler = (
        args.sample_type_sampling
        and hasattr(dataset_train, 'has_sample_types')
        and dataset_train.has_sample_types
    )
    if args.case_aware_sampling and use_sample_type_sampler:
        batch_sampler_train = samplers.CaseAwareSampleTypeBatchSampler(
            dataset_train,
            batch_size=args.batch_size,
            ratios=get_stage_sample_type_ratios('stage2_full', args),
            neighbor_window=args.case_aware_neighbor_window,
            same_case_fraction=args.case_aware_same_case_fraction,
            drop_last=True,
            num_replicas=utils.get_world_size() if args.distributed else 1,
            rank=utils.get_rank() if args.distributed else 0,
            base_sampler=sampler_train,
        )
    elif use_sample_type_sampler:
        batch_sampler_train = samplers.SampleTypeBatchSampler(
            dataset_train,
            batch_size=args.batch_size,
            ratios=get_stage_sample_type_ratios('stage2_full', args),
            drop_last=True,
            num_replicas=utils.get_world_size() if args.distributed else 1,
            rank=utils.get_rank() if args.distributed else 0,
            base_sampler=sampler_train,
        )
    else:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    if args.case_sequential_eval:
        batch_sampler_val = samplers.CaseSequentialBatchSampler(
            dataset_val,
            batch_size=args.batch_size,
            drop_last=False,
            num_replicas=utils.get_world_size() if args.distributed else 1,
            rank=utils.get_rank() if args.distributed else 0,
        )
        data_loader_val = DataLoader(dataset_val, batch_sampler=batch_sampler_val,
                                     collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                     pin_memory=True)
    else:
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                     pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = build_lr_scheduler(optimizer, args)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True,
        )
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    resume_selection_state = {}
    output_dir = Path(args.output_dir)
    if args.pretrained_model and not args.resume:
        load_shape_compatible_pretrained(
            model_without_ddp,
            args.pretrained_model,
            include_keywords=args.pretrained_include_keywords,
            allow_unsafe=args.allow_unsafe_checkpoint,
        )
    if args.resume:
        checkpoint = load_checkpoint(args.resume, allow_unsafe=args.allow_unsafe_checkpoint)
        resume_selection_state = checkpoint if isinstance(checkpoint, dict) else {}
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer_restored = False
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                optimizer_restored = True
            except ValueError as exc:
                print(f"Warning: could not restore optimizer state ({exc}); using fresh optimizer state.")
            if optimizer_restored:
                for pg, pg_old in zip(optimizer.param_groups, p_groups):
                    pg['lr'] = pg_old['lr']
                    pg['initial_lr'] = pg_old.get('initial_lr', pg_old['lr'])
                print(optimizer.param_groups)
                try:
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                except Exception as exc:
                    print(f"Warning: could not restore lr_scheduler state ({exc}); using current scheduler config.")
                if args.lr_scheduler == 'step':
                    print('Warning: step LR scheduler state was restored; overriding lr_drop with the current command-line value.')
                    lr_scheduler.step_size = args.lr_drop
                    lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
                lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        resume_stage_epoch = max(args.start_epoch, int(checkpoint.get('epoch', args.start_epoch)))
        configure_model_for_stage(
            model_without_ddp,
            criterion,
            optimizer,
            get_epoch_training_stage(resume_stage_epoch, args),
            args,
        )        # Optional resume validation is expensive because it runs COCO + medical
        # fracture metrics before the next training log can be written.
        if not args.eval and args.validate_on_resume:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args=args
            )
    
    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, args=args)
        if args.output_dir and coco_evaluator is not None:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    selection_passed_fp_gate = bool(resume_selection_state.get('selection_passed_fp_gate', False))
    best_medical_recall = float(resume_selection_state.get('best_medical_recall', -1.0))
    best_medical_fp = float(resume_selection_state.get('best_medical_fp', float('inf')))
    best_medical_froc = float(resume_selection_state.get('best_medical_froc', -1.0))
    if selection_passed_fp_gate:
        best_medical_recall = float(resume_selection_state.get('selection_score', best_medical_recall))
        best_medical_fp = float(resume_selection_state.get('selection_fp_per_negative_case', best_medical_fp))
        best_medical_froc = float(resume_selection_state.get('selection_froc_score', best_medical_froc))
    best_fallback_recall = float(resume_selection_state.get(
        'best_fallback_recall',
        resume_selection_state.get('selection_score', -1.0),
    ))
    best_fallback_fp = float(resume_selection_state.get(
        'best_fallback_fp',
        resume_selection_state.get('selection_fp_per_negative_case', float('inf')),
    ))
    best_fallback_froc = float(resume_selection_state.get(
        'best_fallback_froc',
        resume_selection_state.get('selection_froc_score', -1.0),
    ))
    current_stage = None
    stage_trainable_params = n_parameters
    for epoch in range(args.start_epoch, args.epochs):
        epoch_stage = get_epoch_training_stage(epoch, args)
        if epoch_stage != current_stage:
            current_stage = epoch_stage
            stage_trainable_params = configure_model_for_stage(model_without_ddp, criterion, optimizer, current_stage, args)
        if args.distributed and hasattr(sampler_train, "set_epoch"):
            sampler_train.set_epoch(epoch)
        if hasattr(batch_sampler_train, "set_epoch"):
            batch_sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'training_stage': current_stage,
                     'log_phase': 'train',
                     'validation_pending': True,
                     'stage_trainable_params': stage_trainable_params,
                     'n_parameters': n_parameters}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            is_step_drop = args.lr_scheduler == 'step' and (epoch + 1) % args.lr_drop == 0
            if is_step_drop or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(build_checkpoint_payload(
                    model_without_ddp,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    args,
                    best_medical_recall,
                    best_medical_fp,
                    best_medical_froc,
                    best_fallback_recall,
                    best_fallback_fp,
                    best_fallback_froc,
                ), checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args=args
        )
        if args.output_dir and args.compute_fracture_metrics:
            current_fp = float(test_stats.get(args.checkpoint_fp_key, float('inf')))
            current_all_fp = float(test_stats.get(args.checkpoint_all_fp_key, float('inf')))
            current_recall = float(test_stats.get(args.checkpoint_recall_key, -1.0))
            current_froc = float(test_stats.get(args.froc_primary_key, -1.0))
            passes_primary_fp_gate = current_fp <= args.checkpoint_fp_neg_case_max
            passes_all_fp_gate = (
                args.checkpoint_all_fp_neg_case_max <= 0
                or current_all_fp <= args.checkpoint_all_fp_neg_case_max
            )
            passes_fp_gate = passes_primary_fp_gate and passes_all_fp_gate
            improves_recall = current_recall > best_medical_recall + args.checkpoint_tie_eps
            tie_breaks = (
                abs(current_recall - best_medical_recall) <= args.checkpoint_tie_eps
                and (
                    current_fp < best_medical_fp - args.checkpoint_tie_eps
                    or (
                        abs(current_fp - best_medical_fp) <= args.checkpoint_tie_eps
                        and current_froc > best_medical_froc + args.checkpoint_tie_eps
                    )
                )
            )
            fallback_improves = (
                current_recall > best_fallback_recall + args.checkpoint_tie_eps
                or (
                    abs(current_recall - best_fallback_recall) <= args.checkpoint_tie_eps
                    and (
                        current_fp < best_fallback_fp - args.checkpoint_tie_eps
                        or (
                            abs(current_fp - best_fallback_fp) <= args.checkpoint_tie_eps
                            and current_froc > best_fallback_froc + args.checkpoint_tie_eps
                        )
                    )
                )
            )
            if fallback_improves:
                best_fallback_recall = current_recall
                best_fallback_fp = current_fp
                best_fallback_froc = current_froc
                utils.save_on_master(build_checkpoint_payload(
                    model_without_ddp,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    args,
                    best_medical_recall,
                    best_medical_fp,
                    best_medical_froc,
                    best_fallback_recall,
                    best_fallback_fp,
                    best_fallback_froc,
                    extra={
                        'selection_metric': args.checkpoint_recall_key,
                        'selection_score': best_fallback_recall,
                        'selection_fp_key': args.checkpoint_fp_key,
                        'selection_fp_per_negative_case': best_fallback_fp,
                        'selection_all_fp_key': args.checkpoint_all_fp_key,
                        'selection_all_fp_per_negative_case': current_all_fp,
                        'selection_froc_key': args.froc_primary_key,
                        'selection_froc_score': best_fallback_froc,
                        'selection_passed_fp_gate': False,
                        'selection_rule': (
                            f'fallback: maximize {args.checkpoint_recall_key} even if '
                            f'{args.checkpoint_fp_key}/{args.checkpoint_all_fp_key} gate is not met'
                        ),
                    },
                ), output_dir / 'checkpoint_best_medical_fallback.pth')
            if passes_fp_gate and (improves_recall or tie_breaks):
                best_medical_recall = current_recall
                best_medical_fp = current_fp
                best_medical_froc = current_froc
                utils.save_on_master(build_checkpoint_payload(
                    model_without_ddp,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    args,
                    best_medical_recall,
                    best_medical_fp,
                    best_medical_froc,
                    best_fallback_recall,
                    best_fallback_fp,
                    best_fallback_froc,
                    extra={
                        'selection_metric': args.checkpoint_recall_key,
                        'selection_score': best_medical_recall,
                        'selection_fp_key': args.checkpoint_fp_key,
                        'selection_fp_per_negative_case': best_medical_fp,
                        'selection_all_fp_key': args.checkpoint_all_fp_key,
                        'selection_all_fp_per_negative_case': current_all_fp,
                        'selection_froc_key': args.froc_primary_key,
                        'selection_froc_score': best_medical_froc,
                        'selection_passed_fp_gate': True,
                        'selection_rule': (
                            f'filter by {args.checkpoint_fp_key} <= {args.checkpoint_fp_neg_case_max} '
                            f'and {args.checkpoint_all_fp_key} <= {args.checkpoint_all_fp_neg_case_max}, '
                            f'then maximize {args.checkpoint_recall_key}; never select by val loss'
                        ),
                    },
                ), output_dir / 'checkpoint_best_medical.pth')

        if args.output_dir:
            utils.save_on_master(build_checkpoint_payload(
                model_without_ddp,
                optimizer,
                lr_scheduler,
                epoch,
                args,
                best_medical_recall,
                best_medical_fp,
                best_medical_froc,
                best_fallback_recall,
                best_fallback_fp,
                best_fallback_froc,
            ), output_dir / 'checkpoint.pth')

        log_stats.update({f'test_{k}': v for k, v in test_stats.items()})
        log_stats['log_phase'] = 'eval'
        log_stats['validation_pending'] = False

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
