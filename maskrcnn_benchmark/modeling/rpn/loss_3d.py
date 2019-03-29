# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from ..utils import cat

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist3d_ops import boxlist_iou_3d, cat_boxlist_3d

DEBUG = True
SHOW_POS_ANCHOR_IOU = DEBUG and False
SHOW_POS_NEG_ANCHORS = DEBUG and False
SHOW_PRED_POS_ANCHORS = DEBUG and True

class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder

    def match_targets_to_anchors(self, anchor, target):
        if target.bbox3d.shape[0] == 0:
          matched_idxs = torch.ones([anchor.bbox3d.shape[0]], dtype=torch.int64, device=anchor.bbox3d.device) * (-1)
          matched_targets = anchor
        else:
          match_quality_matrix = boxlist_iou_3d(anchor, target)
          matched_idxs = self.proposal_matcher(match_quality_matrix)
          #anchor.show_together(target, 200)
          # RPN doesn't need any fields from target
          # for creating the labels, so clear them all
          target = target.copy_with_fields([])
          # get the targets corresponding GT for each anchor
          # NB: need to clamp the indices because we can have a single
          # GT in the image, and matched_idxs can be -2, which goes
          # out of bounds
          matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)

        if SHOW_POS_ANCHOR_IOU:
          num_gt = target.bbox3d.shape[0]
          for j in range(num_gt):
            sampled_pos_inds = torch.nonzero(matched_idxs==j).squeeze(1)
            #sampled_pos_inds = torch.nonzero(match_quality_matrix[j] > 0.3).squeeze(1)

            iou_j = match_quality_matrix[j][sampled_pos_inds]
            anchors_pos_j = anchor[sampled_pos_inds]
            print(f'{iou_j.shape[0]} anchor matched as positive. All anchor centroids are shown.')
            anchors_pos_j.show_together(target[j], points=anchor.bbox3d[:,0:3])

            for i in range(iou_j.shape[0]):
              print(f'{i}th iou: {iou_j[i]}')
              anchors_pos_j[i].show_together(target[j])
            #anchor.show_together(target[j],100)
            pass

        return matched_targets

    def prepare_targets(self, anchors, targets):
        '''
        labels: batch_size * []
        '''
        labels = []
        regression_targets = []
        batch_size = anchors.batch_size()
        assert batch_size == len(targets)
        for bi in range(batch_size):
            # merge anchors of all scales
            anchors_per_image = anchors.example(bi)
            targets_per_image = targets[bi]

            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)
            # discard anchors that go out of the boundaries of the image
            #labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox3d, anchors_per_image.bbox3d
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)


        return labels, regression_targets

    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (BoxList): box num: N
            objectness (list[Tensor]): len=scale_num
            box_regression (list[Tensor]): len=scale_num
            targets (list[BoxList]): len = batch size

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        labels, regression_targets = self.prepare_targets(anchors, targets)
        sampled_pos_inds0, sampled_neg_inds0 = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds0, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds0, dim=0)).squeeze(1)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        batch_size = anchors.batch_size()

        if SHOW_PRED_POS_ANCHORS:
            sampled_inds = sampled_pos_inds
            thres = 0.98
            self.show_pos_anchors_pred(thres, box_regression, anchors, objectness, targets, sampled_inds)


        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)


        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

    def show_pos_anchors_pred(self, thres, rpn_box_regression, anchors, objectness, targets, sampled_inds):
        pred_boxes_3d = self.box_coder.decode(rpn_box_regression, anchors.bbox3d)
        objectness_normed = objectness.sigmoid()
        pred_boxes = anchors.copy()
        pred_boxes.bbox3d = pred_boxes_3d
        pred_boxes.add_field('objectness', objectness_normed)

        anchor_flags = torch.zeros([len(anchors)])
        anchor_flags[sampled_inds] = 1
        pred_boxes.add_field('anchor_flags', anchor_flags)

        for bi,pdb in enumerate(pred_boxes.seperate_examples()):
          pdb.show_by_field('objectness',0.97, targets[bi])
          pdb.show_by_field('anchor_flags',0.5, targets[bi])
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass

def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = RPNLossComputation(matcher, fg_bg_sampler, box_coder)
    return loss_evaluator

