# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 cost_center: float = 0,
                 cost_region: float = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_center = cost_center
        self.cost_region = cost_region
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_center != 0 or cost_region != 0, "all costs cant be 0"

    @staticmethod
    def _sample_region_cost(prior, points, batch_size, num_queries):
        if prior is None or prior.numel() == 0:
            return None
        if prior.shape[0] == 1 and batch_size > 1:
            prior = prior.expand(batch_size, -1, -1, -1)
        grid = points.view(batch_size, num_queries, 1, 2).mul(2).sub(1)
        values = F.grid_sample(prior.to(device=points.device, dtype=points.dtype), grid,
                               mode="bilinear", padding_mode="zeros", align_corners=False)
        return 1.0 - values[:, 0, :, 0].flatten(0, 1)

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            if sum(len(v["boxes"]) for v in targets) == 0:
                empty = torch.empty(0, dtype=torch.int64)
                return [(empty, empty) for _ in range(bs)]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
            out_center = outputs.get("pred_centers", outputs["pred_boxes"][..., :2]).flatten(0, 1)

            # Also concat the target labels and boxes
            if sum(len(v["boxes"]) for v in targets) == 0:
                return [(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64))
                        for _ in range(bs)]
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_center = torch.cdist(out_center, tgt_bbox[:, :2], p=1)

            # Compute the giou cost betwen boxes
            if tgt_bbox.numel() == 0:
                cost_giou = cost_bbox.new_zeros(cost_bbox.shape)
            else:
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                                 box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = (self.cost_bbox * cost_bbox +
                 self.cost_class * cost_class +
                 self.cost_giou * cost_giou +
                 self.cost_center * cost_center)
            region_cost = self._sample_region_cost(
                outputs.get("region_prior"),
                out_center, bs, num_queries
            )
            if region_cost is not None and self.cost_region > 0:
                C = C + self.cost_region * region_cost[:, None]
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou,
                            cost_center=args.set_cost_center,
                            cost_region=args.set_cost_region)
