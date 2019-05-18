# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator

DEBUG = True
SHOW_ROI_INPUT = DEBUG and False

def rm_gt_from_proposals(class_logits, box_regression, proposals, detections_per_img, targets):
    class_logits = class_logits.clone().detach()
    box_regression = box_regression.clone().detach()

    batch_size = len(proposals)
    class_logits_ = []
    box_regression_ = []
    proposals_ = []
    s = 0
    for b in range(batch_size):
        #print(f's:{s}')
        real_proposal_num = len(proposals[b]) - len(targets[b])
        class_logits_.append( class_logits[s:s+real_proposal_num,:] )
        box_regression_.append( box_regression[s:s+real_proposal_num,:] )
        s += len(proposals[b])

        ids = range( real_proposal_num )
        proposals_.append(proposals[b][ids])
    class_logits_ = torch.cat(class_logits_, 0)
    box_regression_ = torch.cat(box_regression_, 0)
    return class_logits_, box_regression_, proposals_

class ROIBoxHead3D(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead3D, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        self.eval_in_train = cfg.DEBUG.eval_in_train
        self.add_gt_proposals = cfg.MODEL.RPN.ADD_GT_PROPOSALS
        self.detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        proposals = proposals.seperate_examples()
        if SHOW_ROI_INPUT and False:
          fgt = cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD
          bgt = cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD
          print(f"proposals over FG_IOU_THRESHOLD: {fgt}")
          proposals[0].show_by_objectness(fgt, targets[0])
          print(f"proposals below BG_IOU_THRESHOLD: {bgt}")
          proposals[0].show_by_objectness(bgt, targets[0], below=True)
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}
        if self.eval_in_train:
            if self.add_gt_proposals:
                class_logits_, box_regression_, proposals_ = rm_gt_from_proposals(
                    class_logits, box_regression, proposals,
                    self.detections_per_img, targets)
            proposals = self.post_processor((class_logits_, box_regression_), proposals_)

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression], targets
        )
        if DEBUG and False:
          print(f"\nloss_classifier_roi:{loss_classifier} \nloss_box_reg_roi: {loss_box_reg}")
          batch_size = len(proposals)
          proposals[0].show_by_objectness(0.5, targets[0])
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass
        return (
            x,
            proposals,
            {"loss_classifier_roi":loss_classifier, "loss_box_reg_roi":loss_box_reg},
        )


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead3D, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead3D(cfg)
