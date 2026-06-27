# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
from .medical_modules import (AdjacentSliceFeatureMemory,
                              ChannelWiseSliceEmbedding, EdgeGuidedSpatialAttention,
                              FractureHeatmapBranch, OrganFeatureGating,
                              RegionPriorEmbedding, dilate_organ_mask,
                              normalize_organ_mask)
from util.fracture_ops import build_heatmap_targets, sample_map_at_points
import copy


STAGE_LOSSES = {
    'stage1_heatmap': ['heatmap'],
    'stage2_full': ['labels', 'boxes', 'centers', 'heatmap', 'region', 'query_coverage',
                    'slice_memory', 'consistency', 'contrastive', 'cardinality'],
    'stage3_hard_negative': ['labels', 'boxes', 'centers', 'heatmap', 'region', 'query_coverage',
                            'slice_memory', 'consistency', 'contrastive', 'hardneg', 'cardinality'],
    'stage4_calibration': ['temperature_calibration', 'calibration', 'cardinality'],
}


def get_stage_losses(stage, include_masks=False):
    losses = list(STAGE_LOSSES.get(stage, STAGE_LOSSES['stage2_full']))
    if include_masks:
        losses.append("masks")
    return losses


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False,
                 slice_embedding_scale=0.02, edge_attention_strength=0.25,
                 dynamic_queries=True,
                 slice_memory_alpha=1.0, slice_memory_beta=0.5, slice_memory_gamma=0.5,
                 organ_mask_input_strength=0.20,
                 organ_feature_gating_strength=1.0,
                 organ_feature_outside_scale=0.25,
                 organ_mask_dilation=5,
                 dynamic_query_fraction=0.70,
                 coverage_query_fraction=0.20,
                 query_region_floor=0.02,
                 use_slice_memory=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.center_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.uncertainty_embed = MLP(hidden_dim, hidden_dim, 1, 3)
        self.logit_temperature = nn.Parameter(torch.ones(1))
        self.num_feature_levels = num_feature_levels
        self.dynamic_queries = dynamic_queries and not two_stage
        self.organ_mask_input_strength = organ_mask_input_strength
        self.organ_mask_dilation = organ_mask_dilation
        self.dynamic_query_fraction = dynamic_query_fraction
        self.coverage_query_fraction = coverage_query_fraction
        self.query_region_floor = query_region_floor
        self.use_slice_memory = use_slice_memory
        self.slice_embedding = ChannelWiseSliceEmbedding(slice_embedding_scale)
        self.edge_attention = EdgeGuidedSpatialAttention(edge_attention_strength)
        self.organ_feature_gating = OrganFeatureGating(
            strength=organ_feature_gating_strength,
            outside_scale=organ_feature_outside_scale,
            dilation=organ_mask_dilation,
        )
        self.fracture_heatmap = FractureHeatmapBranch(hidden_dim)
        self.region_embedding = RegionPriorEmbedding(hidden_dim)
        self.dynamic_query_pos_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.dynamic_query_tgt_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.slice_memory = AdjacentSliceFeatureMemory(hidden_dim, alpha=slice_memory_alpha,
                                                       beta=slice_memory_beta, gamma=slice_memory_gamma)
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.center_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.center_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.uncertainty_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.uncertainty_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            self.center_embed = _get_clones(self.center_embed, num_pred)
            self.uncertainty_embed = _get_clones(self.uncertainty_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.center_embed = nn.ModuleList([self.center_embed for _ in range(num_pred)])
            self.uncertainty_embed = nn.ModuleList([self.uncertainty_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def _normalize_explicit_organ_mask(self, samples_tensor, organ_mask):
        bs, _, height, width = samples_tensor.shape
        if organ_mask is None:
            return None
        if organ_mask.ndim == 2:
            organ_mask = organ_mask.unsqueeze(0).unsqueeze(0)
        elif organ_mask.ndim == 3:
            if organ_mask.shape[0] == bs and organ_mask.shape[1:] == (height, width):
                pass
            elif organ_mask.shape[-2:] == (height, width) and organ_mask.shape[0] in (1, 3):
                organ_mask = organ_mask.unsqueeze(0)
            else:
                raise ValueError(
                    f"Unsupported organ_mask shape {tuple(organ_mask.shape)} for batch size {bs}"
                )
        elif organ_mask.ndim != 4:
            raise ValueError(f"Unsupported organ_mask ndim: {organ_mask.ndim}")
        if organ_mask.ndim == 4 and organ_mask.shape[0] == 1 and bs > 1:
            organ_mask = organ_mask.expand(bs, -1, -1, -1)
        elif organ_mask.ndim == 4 and organ_mask.shape[0] != bs:
            raise ValueError(
                f"Explicit organ_mask batch {organ_mask.shape[0]} does not match samples batch {bs}"
            )
        return normalize_organ_mask(
            organ_mask,
            bs,
            height,
            width,
            samples_tensor.device,
            samples_tensor.dtype,
            channels=3,
        )

    def _organ_mask_from_targets(self, samples_tensor, targets):
        bs, _, height, width = samples_tensor.shape
        if targets is None:
            return torch.ones(bs, 3, height, width, device=samples_tensor.device, dtype=samples_tensor.dtype)
        masks = []
        for target in targets:
            mask = target.get("organ_mask")
            if mask is None:
                mask = torch.ones(3, height, width, device=samples_tensor.device, dtype=samples_tensor.dtype)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = normalize_organ_mask(
                mask.unsqueeze(0),
                1,
                height,
                width,
                samples_tensor.device,
                samples_tensor.dtype,
                channels=3,
            )[0]
            masks.append(mask)
        return torch.stack(masks, dim=0)

    def _resolve_organ_mask(self, samples_tensor, targets=None, organ_mask=None):
        explicit_mask = self._normalize_explicit_organ_mask(samples_tensor, organ_mask)
        if explicit_mask is not None:
            return explicit_mask
        return self._organ_mask_from_targets(samples_tensor, targets)

    @staticmethod
    def _sample_features_at_points(src, ref_points):
        bs, _, _, _ = src.shape
        grid = ref_points.mul(2).sub(1).view(bs, ref_points.shape[1], 1, 2)
        sampled_features = F.grid_sample(src, grid, mode="bilinear",
                                         padding_mode="zeros", align_corners=False)
        return sampled_features[:, :, :, 0].transpose(1, 2)

    def _coverage_reference_points(self, region_prior, count, height, width):
        bs = region_prior.shape[0]
        if count <= 0:
            return region_prior.new_zeros((bs, 0, 2))
        flat = region_prior.flatten(2)[:, 0]
        refs = []
        for batch_idx in range(bs):
            valid = torch.nonzero(flat[batch_idx] > 0.05, as_tuple=False).flatten()
            if valid.numel() == 0:
                valid = torch.arange(height * width, device=region_prior.device)
            valid = valid[torch.argsort(valid)]
            if valid.numel() >= count:
                pick = torch.linspace(0, valid.numel() - 1, count, device=region_prior.device).round().long()
                indices = valid[pick]
            else:
                pad = valid[torch.arange(count - valid.numel(), device=region_prior.device) % valid.numel()]
                indices = torch.cat([valid, pad], dim=0)
            y = torch.div(indices, width, rounding_mode='floor')
            x = indices % width
            refs.append(torch.stack([
                (x.float() + 0.5) / float(width),
                (y.float() + 0.5) / float(height),
            ], dim=-1))
        return torch.stack(refs, dim=0)

    def _sample_dynamic_queries(self, src, heatmap_logits, samples_tensor,
                                region_prior=None, organ_mask=None, padding_mask=None):
        bs, hidden_dim, h, w = src.shape
        heatmap_score = heatmap_logits.sigmoid()
        if region_prior is not None:
            roi = F.interpolate(region_prior.to(device=src.device, dtype=src.dtype),
                                size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)
            heatmap_score = heatmap_score * (self.query_region_floor + (1.0 - self.query_region_floor) * roi)
        if padding_mask is not None:
            invalid = F.interpolate(
                padding_mask[:, None].float(), size=(h, w), mode="nearest"
            ).to(torch.bool)
            heatmap_score = heatmap_score.masked_fill(invalid, -1.0)

        dynamic_count = max(1, min(self.num_queries, int(round(self.num_queries * self.dynamic_query_fraction))))
        coverage_count = min(dynamic_count - 1, max(0, int(round(self.num_queries * self.coverage_query_fraction))))
        peak_count = max(dynamic_count - coverage_count, 1)
        peak_count = min(peak_count, h * w)
        peak_scores, peak_indices = torch.topk(heatmap_score.flatten(2), peak_count, dim=2)
        peak_indices = peak_indices[:, 0]
        y = torch.div(peak_indices, w, rounding_mode='floor')
        x = peak_indices % w
        peak_ref_points = torch.stack([
            (x.float() + 0.5) / float(w),
            (y.float() + 0.5) / float(h),
        ], dim=-1)
        if region_prior is not None:
            roi_small = F.interpolate(region_prior.to(device=src.device, dtype=src.dtype),
                                      size=(h, w), mode="nearest")
        else:
            roi_small = torch.ones(bs, 1, h, w, device=src.device, dtype=src.dtype)
        if padding_mask is not None:
            invalid = F.interpolate(
                padding_mask[:, None].float(), size=(h, w), mode="nearest"
            ).to(torch.bool)
            roi_small = roi_small.masked_fill(invalid, 0.0)
        coverage_ref_points = self._coverage_reference_points(roi_small, coverage_count, h, w)
        ref_points = torch.cat([peak_ref_points, coverage_ref_points], dim=1)
        sampled_features = self._sample_features_at_points(src, ref_points)

        if region_prior is not None and region_prior.numel() > 0:
            region_values = sample_map_at_points(region_prior, ref_points)
        else:
            region_values = sampled_features.new_zeros(ref_points.shape[:-1])
        query_base = sampled_features + self.region_embedding(region_values)
        query_embed = query_base
        memory_stats = {}
        if self.use_slice_memory:
            query_embed, memory_stats = self.slice_memory(
                query_embed, ref_points, samples_tensor,
                organ_mask=organ_mask, feature_size=(h, w)
            )

        if coverage_count > 0:
            coverage_scores = sample_map_at_points(roi_small, coverage_ref_points)
            peak_scores = torch.cat([peak_scores[:, 0], coverage_scores], dim=1)
        else:
            peak_scores = peak_scores[:, 0]

        query_pos = self.dynamic_query_pos_proj(query_base)
        query_tgt = self.dynamic_query_tgt_proj(query_embed)
        return query_pos, query_tgt, ref_points, peak_scores, memory_stats

    def forward(self, samples: NestedTensor, targets=None, organ_mask=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        raw_samples_tensor = samples.tensors
        if raw_samples_tensor.shape[1] != 3:
            raise ValueError(
                f"2.5D fracture detector expects explicit 3-channel t-1/t/t+1 input, "
                f"got shape {tuple(raw_samples_tensor.shape)}"
            )
        organ_mask = self._resolve_organ_mask(raw_samples_tensor, targets=targets, organ_mask=organ_mask)
        center_organ_mask = organ_mask[:, 1:2] if organ_mask.shape[1] >= 3 else organ_mask[:, :1]
        region_prior = dilate_organ_mask(center_organ_mask, self.organ_mask_dilation)
        gated_input = raw_samples_tensor * (1.0 + self.organ_mask_input_strength * organ_mask)
        samples = NestedTensor(self.slice_embedding(gated_input), samples.mask)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        srcs, region_level_priors = self.organ_feature_gating(srcs, region_prior)
        srcs, edge_prior = self.edge_attention(srcs, samples.tensors, organ_mask=organ_mask)

        heatmap_logits = self.fracture_heatmap(srcs[0])
        dynamic_query_embed = dynamic_tgt = dynamic_reference_points = None
        query_peak_scores = None
        slice_memory_stats = {}
        if self.dynamic_queries:
            dynamic_query_embed, dynamic_tgt, dynamic_reference_points, query_peak_scores, slice_memory_stats = \
                self._sample_dynamic_queries(
                    srcs[0], heatmap_logits, raw_samples_tensor,
                    region_prior=region_prior, organ_mask=organ_mask, padding_mask=masks[0]
                )

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                src, extra_priors = self.organ_feature_gating([src], region_prior)
                src = src[0]
                region_level_priors.extend(extra_priors)
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
            srcs, masks, pos, query_embeds,
            dynamic_query_embed=dynamic_query_embed,
            dynamic_tgt=dynamic_tgt,
            dynamic_reference_points=dynamic_reference_points,
            region_prior=region_prior)

        outputs_classes = []
        outputs_coords = []
        outputs_centers = []
        outputs_uncertainty = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            center = self.center_embed[lvl](hs[lvl])
            uncertainty = self.uncertainty_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                center += reference
            outputs_coord = tmp.sigmoid()
            outputs_center = center.sigmoid()
            temperature = self.logit_temperature.clamp(min=0.05, max=10.0).to(outputs_class.dtype)
            outputs_classes.append(outputs_class / temperature)
            outputs_coords.append(outputs_coord)
            outputs_centers.append(outputs_center)
            outputs_uncertainty.append(uncertainty)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_center = torch.stack(outputs_centers)
        outputs_uncertainty = torch.stack(outputs_uncertainty)

        pooled_slice_features = torch.stack([
            raw_samples_tensor[:, i].flatten(1).mean(dim=1) for i in range(raw_samples_tensor.shape[1])
        ], dim=1)
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'pred_centers': outputs_center[-1],
            'pred_uncertainty': outputs_uncertainty[-1],
            'fracture_heatmap': heatmap_logits,
            'edge_prior': edge_prior,
            'region_prior': region_prior,
            'organ_mask': organ_mask,
            'query_reference_points': init_reference,
            'query_peak_scores': query_peak_scores,
            'slice_features': pooled_slice_features,
            'query_features': hs[-1].mean(dim=1),
            'logit_temperature': self.logit_temperature.clamp(min=0.05, max=10.0),
        }
        if region_level_priors:
            out['region_level_priors'] = region_level_priors
        out.update(slice_memory_stats)
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord,
                                                    outputs_center, outputs_uncertainty)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_center, outputs_uncertainty):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_centers': c, 'pred_uncertainty': d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1],
                                      outputs_center[:-1], outputs_uncertainty[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25,
                 training_stage='stage2_full', iou_loss_type='giou',
                 heatmap_sigma=2.0, query_coverage_radius=0.08):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.training_stage = training_stage
        self.iou_loss_type = iou_loss_type
        self.heatmap_sigma = heatmap_sigma
        self.query_coverage_radius = query_coverage_radius

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            if target_classes_o.numel() == 0:
                losses['class_error'] = src_logits.sum() * 0.0
            else:
                losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Sigmoid-focal DETR has no explicit no-object class; count confident
        # query slots for logging instead of using the original softmax argmax.
        card_pred = (pred_logits.sigmoid().amax(dim=-1) > 0.5).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            zero = outputs['pred_boxes'].sum() * 0.0
            return {'loss_bbox': zero, 'loss_giou': zero, 'loss_wise_iou': zero}
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        if src_boxes.numel() == 0 or target_boxes.numel() == 0:
            zero = outputs['pred_boxes'].sum() * 0.0
            return {'loss_bbox': zero, 'loss_giou': zero, 'loss_wise_iou': zero}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        stable_iou_loss = loss_giou
        if self.iou_loss_type != 'giou':
            stable_iou_loss = box_ops.aligned_iou_loss(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
                loss_type=self.iou_loss_type,
            )
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['loss_wise_iou'] = stable_iou_loss.sum() / num_boxes
        return losses

    def loss_centers(self, outputs, targets, indices, num_boxes):
        assert 'pred_centers' in outputs
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            return {'loss_center': outputs['pred_centers'].sum() * 0.0}
        src_centers = outputs['pred_centers'][idx]
        target_centers = torch.cat([t['boxes'][i, :2] for t, (_, i) in zip(targets, indices)], dim=0)
        if src_centers.numel() == 0 or target_centers.numel() == 0:
            return {'loss_center': outputs['pred_centers'].sum() * 0.0}
        loss_center = F.l1_loss(src_centers, target_centers, reduction='none')
        losses = {'loss_center': loss_center.sum() / num_boxes}
        return losses

    def loss_heatmap(self, outputs, targets, indices, num_boxes):
        if 'fracture_heatmap' not in outputs:
            return {'loss_heatmap': torch.tensor(0.0, device=next(iter(outputs.values())).device)}
        heatmap_logits = outputs['fracture_heatmap']
        target_heatmap = build_heatmap_targets(
            targets, heatmap_logits.shape[-2:], heatmap_logits.device, sigma=self.heatmap_sigma
        ).to(dtype=heatmap_logits.dtype)
        weight = torch.ones_like(target_heatmap) * 0.25
        region_prior = outputs.get('region_prior')
        if region_prior is not None:
            region = F.interpolate(region_prior.to(device=heatmap_logits.device, dtype=heatmap_logits.dtype),
                                   size=heatmap_logits.shape[-2:], mode="bilinear", align_corners=False)
            weight = weight + 0.75 * region.clamp(0, 1)
        weight = torch.maximum(weight, target_heatmap)
        loss = F.binary_cross_entropy_with_logits(heatmap_logits, target_heatmap, reduction='none')
        return {'loss_heatmap': (loss * weight).sum() / weight.sum().clamp(min=1.0)}

    def loss_region(self, outputs, targets, indices, num_boxes):
        region_prior = outputs.get('region_prior')
        if region_prior is None or region_prior.numel() == 0:
            return {'loss_region': outputs['pred_logits'].sum() * 0.0}
        centers = outputs.get('pred_centers', outputs['pred_boxes'][..., :2]).clamp(0, 1)
        region_values = sample_map_at_points(
            region_prior.to(device=centers.device, dtype=centers.dtype), centers
        ).clamp(0, 1)
        probs = outputs['pred_logits'].sigmoid().amax(dim=-1)
        outside_penalty = probs * (1.0 - region_values).pow(2)

        idx = self._get_src_permutation_idx(indices)
        matched_bonus = outputs['pred_logits'].sum() * 0.0
        if idx[0].numel() > 0:
            matched_region = region_values[idx]
            matched_bonus = (1.0 - matched_region).mean()
        return {'loss_region': outside_penalty.mean() + matched_bonus}

    def loss_query_coverage(self, outputs, targets, indices, num_boxes):
        heatmap_logits = outputs.get('fracture_heatmap')
        heatmap_losses = []
        if heatmap_logits is not None and heatmap_logits.numel() > 0:
            heatmap_scores = heatmap_logits.sigmoid()
            for batch_idx, target in enumerate(targets):
                boxes = target.get("boxes")
                if boxes is None or boxes.numel() == 0:
                    continue
                target_centers = boxes[:, :2].to(heatmap_scores.device)
                sampled_scores = sample_map_at_points(
                    heatmap_scores[batch_idx:batch_idx + 1],
                    target_centers.unsqueeze(0),
                )[0]
                heatmap_losses.append((1.0 - sampled_scores.clamp(0, 1)).mean())

        reference_points = outputs.get('query_reference_points')
        distance_losses = []
        if reference_points is not None and reference_points.numel() > 0:
            reference_points = reference_points[..., :2]
            for batch_idx, target in enumerate(targets):
                boxes = target.get("boxes")
                if boxes is None or boxes.numel() == 0:
                    continue
                target_centers = boxes[:, :2].to(reference_points.device)
                distances = torch.cdist(target_centers, reference_points[batch_idx].clamp(0, 1), p=2)
                nearest = distances.min(dim=1).values
                distance_losses.append(F.relu(nearest - self.query_coverage_radius).mean())

        losses = []
        if heatmap_losses:
            losses.append(torch.stack(heatmap_losses).mean())
        if distance_losses:
            # Dynamic query indices come from top-k heatmap selection, so the
            # geometric distance term is mostly a coverage monitor. Keep it as
            # a small auxiliary term while the heatmap sampling term carries
            # the differentiable training signal.
            losses.append(0.1 * torch.stack(distance_losses).mean())
        if not losses:
            return {'loss_query_coverage': outputs['pred_logits'].sum() * 0.0}
        return {'loss_query_coverage': sum(losses)}

    def loss_slice_memory(self, outputs, targets, indices, num_boxes):
        consistency = outputs.get('slice_memory_consistency')
        if consistency is None:
            return {'loss_slice_memory': outputs['pred_logits'].sum() * 0.0}
        return {'loss_slice_memory': consistency}

    def loss_consistency(self, outputs, targets, indices, num_boxes):
        if 'pred_centers' not in outputs or 'pred_boxes' not in outputs:
            return {'loss_consistency': torch.tensor(0.0, device=outputs['pred_logits'].device)}
        idx = self._get_src_permutation_idx(indices)
        pred_centers = outputs['pred_centers'][idx]
        box_centers = outputs['pred_boxes'][idx][:, :2]
        if pred_centers.numel() == 0:
            return {'loss_consistency': outputs['pred_boxes'].sum() * 0.0}
        return {'loss_consistency': F.l1_loss(pred_centers, box_centers, reduction='mean')}

    def loss_contrastive(self, outputs, targets, indices, num_boxes):
        query_features = outputs.get('query_features')
        if query_features is None:
            return {'loss_contrastive': torch.tensor(0.0, device=outputs['pred_logits'].device)}
        sample_types = torch.stack([
            t.get('sample_type_id', torch.tensor(3, device=query_features.device)).to(query_features.device).view(())
            for t in targets
        ])
        # Case-aware contrastive objective: compare P/PN within the same case
        # so the model learns why neighboring same-anatomy slices are negative.
        case_ids = [str(t.get('case_id', f'batch-{idx}')) for idx, t in enumerate(targets)]
        losses = []
        embeddings = F.normalize(query_features, dim=1)
        for case_id in sorted(set(case_ids)):
            case_mask = torch.tensor([item == case_id for item in case_ids],
                                     device=query_features.device, dtype=torch.bool)
            pos_mask = case_mask & (sample_types == 0)
            neg_mask = case_mask & (sample_types == 1)
            if not pos_mask.any() or not neg_mask.any():
                continue
            pos = embeddings[pos_mask].mean(dim=0, keepdim=True)
            neg = embeddings[neg_mask].mean(dim=0, keepdim=True)
            margin = 0.5
            losses.append(F.relu(F.cosine_similarity(pos, neg) + margin).mean())
        if not losses:
            return {'loss_contrastive': query_features.sum() * 0.0}
        return {'loss_contrastive': torch.stack(losses).mean()}

    def loss_hardneg(self, outputs, targets, indices, num_boxes):
        sample_types = torch.stack([
            t.get('sample_type_id', torch.tensor(3, device=outputs['pred_logits'].device)).to(outputs['pred_logits'].device).view(())
            for t in targets
        ])
        hn_mask = sample_types == 2
        if not hn_mask.any():
            return {'loss_hardneg': outputs['pred_logits'].sum() * 0.0}
        probs_per_query = outputs['pred_logits'].sigmoid().amax(dim=2)
        region_prior = outputs.get('region_prior')
        if region_prior is not None:
            centers = outputs.get('pred_centers', outputs['pred_boxes'][..., :2]).clamp(0, 1)
            region_values = sample_map_at_points(
                region_prior.to(device=centers.device, dtype=centers.dtype), centers
            ).clamp(0, 1)
            probs_per_query = probs_per_query * region_values
        probs = probs_per_query.amax(dim=1)
        return {'loss_hardneg': probs[hn_mask].mean()}

    def loss_calibration(self, outputs, targets, indices, num_boxes):
        if 'pred_uncertainty' not in outputs:
            return {'loss_calibration': outputs['pred_logits'].sum() * 0.0}
        idx = self._get_src_permutation_idx(indices)
        pred_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        matched_uncertainty = outputs['pred_uncertainty'][idx]
        if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
            return {'loss_calibration': outputs['pred_uncertainty'].sum() * 0.0}
        giou = torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(pred_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        target_uncertainty = (1.0 - giou).unsqueeze(-1).clamp(0.0, 1.0).detach()
        return {'loss_calibration': F.smooth_l1_loss(matched_uncertainty.sigmoid(), target_uncertainty)}

    def loss_temperature_calibration(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        if idx[0].numel() > 0:
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes[idx] = target_classes_o

        target_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            device=src_logits.device,
        )
        target_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_onehot = target_onehot[:, :, :-1]
        weights = torch.full_like(target_onehot, 0.05)
        weights = torch.where(target_onehot > 0, torch.ones_like(weights), weights)
        nll = F.binary_cross_entropy_with_logits(src_logits, target_onehot, reduction='none')
        probs = src_logits.sigmoid()
        brier = (probs - target_onehot).pow(2)
        denom = weights.sum().clamp(min=1.0)
        return {
            'loss_temperature_nll': (nll * weights).sum() / denom,
            'loss_temperature_brier': (brier * weights).sum() / denom,
        }

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'centers': self.loss_centers,
            'heatmap': self.loss_heatmap,
            'consistency': self.loss_consistency,
            'contrastive': self.loss_contrastive,
            'hardneg': self.loss_hardneg,
            'calibration': self.loss_calibration,
            'temperature_calibration': self.loss_temperature_calibration,
            'region': self.loss_region,
            'query_coverage': self.loss_query_coverage,
            'slice_memory': self.loss_slice_memory,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def get_active_losses(self):
        return self.losses

    def set_training_stage(self, training_stage, include_masks=False):
        self.training_stage = training_stage
        self.losses = get_stage_losses(training_stage, include_masks=include_masks)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        matching_losses = {'labels', 'boxes', 'centers', 'masks', 'cardinality',
                           'calibration', 'temperature_calibration', 'region'}
        needs_matching = any(loss in matching_losses for loss in self.losses)
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets) if needs_matching else [
            (torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)) for _ in targets
        ]

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        active_losses = self.get_active_losses()
        for loss in active_losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if needs_matching and 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in active_losses:
                    if loss in {'masks', 'heatmap', 'contrastive', 'hardneg', 'region', 'query_coverage', 'slice_memory'}:
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if needs_matching and 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in active_losses:
                if loss in {'masks', 'heatmap', 'consistency', 'contrastive', 'hardneg',
                            'calibration', 'temperature_calibration', 'centers',
                            'region', 'query_coverage', 'slice_memory'}:
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        out_centers = outputs.get('pred_centers')
        out_uncertainty = outputs.get('pred_uncertainty')

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk = min(100, prob.shape[1] * prob.shape[2])
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), topk, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        centers = None
        uncertainty = None
        if out_centers is not None:
            centers = torch.gather(out_centers, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 2))
        if out_uncertainty is not None:
            uncertainty = torch.gather(out_uncertainty.sigmoid(), 1, topk_boxes.unsqueeze(-1))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        if centers is not None:
            centers = centers * torch.stack([img_w, img_h], dim=1)[:, None, :]

        results = []
        for idx, (s, l, b) in enumerate(zip(scores, labels, boxes)):
            result = {'scores': s, 'labels': l, 'boxes': b}
            if centers is not None:
                result['centers'] = centers[idx]
            if uncertainty is not None:
                result['uncertainty'] = uncertainty[idx, :, 0]
            results.append(result)

        return results


class PostProcessFracture(nn.Module):
    """Medical post-process with cross-slice lesion association metadata."""

    def __init__(
        self,
        score_threshold=0.3,
        iou_threshold=0.2,
        min_continuity=2,
        topk_lesions=5,
        case_alpha=0.7,
        case_beta=0.3,
        continuity_norm=5.0,
        organ_score_floor=0.05,
        uncertainty_score_floor=0.5,
        max_detections_per_slice=20,
    ):
        super().__init__()
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.min_continuity = min_continuity
        self.topk_lesions = topk_lesions
        self.case_alpha = case_alpha
        self.case_beta = case_beta
        self.continuity_norm = continuity_norm
        self.organ_score_floor = organ_score_floor
        self.uncertainty_score_floor = uncertainty_score_floor
        self.max_detections_per_slice = max(int(max_detections_per_slice), 0)
        self.base = PostProcess()

    def build_case_aggregator(self):
        from util.fracture_eval import CaseLevelAggregator

        return CaseLevelAggregator(
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold,
            min_continuity=self.min_continuity,
            topk_lesions=self.topk_lesions,
            case_alpha=self.case_alpha,
            case_beta=self.case_beta,
            continuity_norm=self.continuity_norm,
        )

    @staticmethod
    def _result_organ_mask(outputs, idx, target=None):
        organ_mask = outputs.get("organ_mask")
        if organ_mask is not None:
            if organ_mask.ndim == 4:
                organ_mask = organ_mask[idx]
            elif organ_mask.ndim != 3:
                organ_mask = None
        if organ_mask is None and target is not None:
            organ_mask = target.get("organ_mask")
        return organ_mask

    def _organ_scores(self, result, target_size, organ_mask=None, target=None):
        if organ_mask is None and target is not None:
            organ_mask = target.get("organ_mask")
        if organ_mask is None:
            return None
        if organ_mask.ndim == 2:
            organ_mask = organ_mask.unsqueeze(0)
        if organ_mask.shape[0] >= 3:
            organ_mask = organ_mask[1:2]
        else:
            organ_mask = organ_mask[:1]
        h, w = int(target_size[0].item()), int(target_size[1].item())
        organ_mask = F.interpolate(
            organ_mask[None].float().to(result["scores"].device),
            size=(h, w), mode="nearest"
        )
        if "centers" in result:
            centers = result["centers"]
        else:
            boxes = result["boxes"]
            centers = torch.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5,
                                   (boxes[:, 1] + boxes[:, 3]) * 0.5], dim=-1)
        norm = centers.clone()
        norm[:, 0] = norm[:, 0] / max(float(w), 1.0)
        norm[:, 1] = norm[:, 1] / max(float(h), 1.0)
        grid = norm.clamp(0, 1).mul(2).sub(1).view(1, -1, 1, 2)
        values = F.grid_sample(organ_mask, grid, mode="bilinear",
                               padding_mode="zeros", align_corners=False)
        return values[0, 0, :, 0].clamp(0, 1)

    def _adjust_result_scores(self, result, organ_scores=None):
        adjusted_scores = result["scores"]
        updated = {}
        if organ_scores is not None:
            updated["organ_scores"] = organ_scores
            adjusted_scores = adjusted_scores * (
                self.organ_score_floor + (1.0 - self.organ_score_floor) * organ_scores
            )
        uncertainty = result.get("uncertainty")
        if uncertainty is not None:
            certainty = (1.0 - uncertainty).clamp(0, 1)
            penalty = (uncertainty - 0.5).clamp(min=0, max=0.5) * 2.0
            updated["certainty"] = certainty
            adjusted_scores = adjusted_scores * (
                1.0 - (1.0 - self.uncertainty_score_floor) * penalty
            )
        if not updated:
            return result
        return {**result, **updated, "raw_scores": result["scores"], "scores": adjusted_scores}

    @torch.no_grad()
    def forward(self, outputs, target_sizes, targets=None):
        from util.fracture_eval import associate_cross_slice_lesions, aggregate_case_predictions

        base_results = self.base(outputs, target_sizes)
        fracture_results = []
        for idx, result in enumerate(base_results):
            target = targets[idx] if targets is not None else None
            organ_mask = self._result_organ_mask(outputs, idx, target=target)
            organ_scores = self._organ_scores(
                result,
                target_sizes[idx],
                organ_mask=organ_mask,
                target=target,
            )
            result = self._adjust_result_scores(result, organ_scores=organ_scores)
            slice_value = target.get("slice_index") if target is not None else None
            slice_index = int(slice_value.item()) if hasattr(slice_value, "item") else idx
            case_id = target.get("case_id", target.get("image_id", idx)) if target is not None else idx
            if isinstance(case_id, torch.Tensor):
                case_id = case_id.flatten()[0].item()
            all_detections = []
            for det_idx in range(result["scores"].shape[0]):
                detection = {
                    "score": result["scores"][det_idx].detach().cpu(),
                    "box": result["boxes"][det_idx].detach().cpu(),
                    "slice_index": slice_index,
                    "case_id": str(case_id),
                }
                if "centers" in result:
                    detection["center"] = result["centers"][det_idx].detach().cpu()
                if "organ_scores" in result:
                    detection["organ_score"] = result["organ_scores"][det_idx].detach().cpu()
                if "uncertainty" in result:
                    detection["uncertainty"] = result["uncertainty"][det_idx].detach().cpu()
                if "certainty" in result:
                    detection["certainty"] = result["certainty"][det_idx].detach().cpu()
                all_detections.append(detection)
            raw_detections = sorted(
                all_detections,
                key=lambda item: float(item["score"].view(-1)[0].item()),
                reverse=True,
            )
            if self.max_detections_per_slice > 0:
                raw_detections = raw_detections[:self.max_detections_per_slice]
            detections = [
                detection for detection in all_detections
                if float(detection["score"].view(-1)[0].item()) >= self.score_threshold
            ]
            detections.sort(key=lambda item: float(item["score"].view(-1)[0].item()), reverse=True)
            if self.max_detections_per_slice > 0:
                detections = detections[:self.max_detections_per_slice]
            lesions = associate_cross_slice_lesions(
                detections,
                iou_threshold=self.iou_threshold,
                min_continuity=self.min_continuity,
            )
            fracture_results.append({
                **result,
                "all_detections": all_detections,
                "raw_detections": raw_detections,
                "detections": detections,
                "lesions": lesions,
                "slice_local_case": aggregate_case_predictions(
                    detections,
                    alpha=self.case_alpha,
                    beta=self.case_beta,
                    topk=self.topk_lesions,
                    score_threshold=self.score_threshold,
                    iou_threshold=self.iou_threshold,
                    min_continuity=self.min_continuity,
                    continuity_norm=self.continuity_norm,
                ),
                "case": None,
            })
        if targets is not None:
            case_aggregator = self.build_case_aggregator()
            case_aggregator.update_many(fracture_results)
            case_summaries = case_aggregator.summarize_cases()
            for result in fracture_results:
                detections = result.get("detections", [])
                if not detections:
                    continue
                case_id = str(detections[0].get("case_id", "unknown"))
                result["case_batch_cross_slice"] = case_summaries.get(case_id)
        return fracture_results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    if getattr(args, "masks", False):
        raise NotImplementedError(
            "--masks uses the original DETRsegm path, which is not compatible with "
            "the current medical DeformableDETR forward. Organ ROI masks are loaded "
            "through --organ_mask_root and do not require --masks."
        )
    num_classes = 1 if getattr(args, "fracture_single_class", False) else (20 if args.dataset_file != 'coco' else 91)
    if args.dataset_file == "coco_panoptic" and not getattr(args, "fracture_single_class", False):
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        slice_embedding_scale=args.slice_embedding_scale,
        edge_attention_strength=args.edge_attention_strength,
        dynamic_queries=args.dynamic_queries,
        slice_memory_alpha=args.slice_memory_alpha,
        slice_memory_beta=args.slice_memory_beta,
        slice_memory_gamma=args.slice_memory_gamma,
        organ_mask_input_strength=args.organ_mask_input_strength,
        organ_feature_gating_strength=args.organ_feature_gating_strength,
        organ_feature_outside_scale=args.organ_feature_outside_scale,
        organ_mask_dilation=args.organ_mask_dilation,
        dynamic_query_fraction=args.dynamic_query_fraction,
        coverage_query_fraction=args.coverage_query_fraction,
        query_region_floor=args.query_region_floor,
        use_slice_memory=args.use_slice_memory,
    )
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_wise_iou'] = args.wise_iou_loss_coef
    weight_dict['loss_center'] = args.center_loss_coef
    weight_dict['loss_heatmap'] = args.heatmap_loss_coef
    weight_dict['loss_consistency'] = args.consistency_loss_coef
    weight_dict['loss_contrastive'] = args.contrastive_loss_coef
    weight_dict['loss_hardneg'] = args.hardneg_loss_coef
    weight_dict['loss_calibration'] = args.calibration_loss_coef
    weight_dict['loss_temperature_nll'] = args.calibration_loss_coef
    weight_dict['loss_temperature_brier'] = args.calibration_loss_coef
    weight_dict['loss_region'] = args.region_loss_coef
    weight_dict['loss_query_coverage'] = args.query_coverage_loss_coef
    weight_dict['loss_slice_memory'] = args.slice_memory_loss_coef
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    initial_stage = args.training_stage
    if initial_stage == 'auto_four_stage':
        initial_stage = 'stage1_heatmap'
    losses = get_stage_losses(initial_stage, include_masks=args.masks)
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses,
                             focal_alpha=args.focal_alpha, training_stage=initial_stage,
                             iou_loss_type=args.iou_loss_type,
                             heatmap_sigma=args.heatmap_sigma,
                             query_coverage_radius=args.query_coverage_radius)
    criterion.to(device)
    postprocessors = {
        'bbox': PostProcess(),
        'fracture': PostProcessFracture(
            score_threshold=args.fracture_score_threshold,
            iou_threshold=args.case_association_iou,
            min_continuity=args.case_min_continuity,
            topk_lesions=args.case_topk_lesions,
            case_alpha=args.case_alpha,
            case_beta=args.case_beta,
            continuity_norm=args.case_continuity_norm,
            organ_score_floor=args.postprocess_organ_score_floor,
            uncertainty_score_floor=args.postprocess_uncertainty_score_floor,
            max_detections_per_slice=args.fracture_max_detections_per_slice,
        ),
    }
    return model, criterion, postprocessors
