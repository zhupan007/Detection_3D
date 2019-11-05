# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder_3d import BoxCoder3D
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops_3d import boxlist_iou_3d
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.bounding_box_3d import cat_boxlist_3d
from maskrcnn_benchmark.modeling.seperate_classifier import SeperateClassifier

DEBUG = True
SHOW_ROI_CLASSFICATION = DEBUG and False
CHECK_IOU = False
CHECK_REGRESSION_TARGET_YAW = False


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder, yaw_loss_mode, add_gt_proposals, aug_thickness, seperate_classifier):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder3D)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.yaw_loss_mode = yaw_loss_mode

        self.high_threshold = proposal_matcher.high_threshold
        self.low_threshold = proposal_matcher.low_threshold
        self.add_gt_proposals = add_gt_proposals
        self.aug_thickness = aug_thickness
        self.seperate_classifier = seperate_classifier
        self.need_seperate = seperate_classifier.need_seperate

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou_3d(target, proposal, aug_thickness=self.aug_thickness, criterion=-1, flag='roi_label_generation')
        matched_idxs = self.proposal_matcher(match_quality_matrix, yaw_diff=None, flag='ROI')
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)

        if CHECK_IOU:
          num_gt = len(target)
          if not torch.all( matched_idxs[-num_gt:].cpu() == torch.arange(num_gt) ):
            ious = match_quality_matrix[:,-num_gt:].diag()
            err_inds = torch.nonzero(torch.abs(ious - 1) > 1e-5 ).view(-1) - len(ious)
            print( f"IOU error: \n{ious}")
            err_targets = target[err_inds]
            ious__ = boxlist_iou_3d(err_targets, err_targets, 0)
            print(err_targets.bbox3d)
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            assert False
            pass
        return matched_targets

    def prepare_targets(self, proposals, targets):
        '''
        proposals do not have object class info
        ROI is only performed on matched proposals.
        Generate class label and regression_targets for all matched proposals.
        '''
        labels = []
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            if len(targets_per_image) == 0:
              prop_num = len(proposals_per_image)
              # negative
              device = proposals[0].bbox3d.device
              labels.append(torch.zeros([prop_num],dtype=torch.int64).to(device))
              regression_targets.append(torch.zeros([prop_num,7],dtype=torch.float32).to(device))
              continue

            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")


            labels_per_image0 = matched_targets.get_field("labels")
            labels_per_image = labels_per_image0.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox3d, proposals_per_image.bbox3d
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        #if not labels[0].device == torch.device('cuda:0'):
        #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
        #  pass
        return labels, regression_targets


    def subsample(self, proposals, targets):
      if self.need_seperate:
        proposals = self.seperate_classifier.seperate_subsample(proposals, targets, self.subsample_standard)
        self._proposals = proposals
        return proposals
      else:
        return self.subsample_standard(proposals, targets)


    def subsample_standard(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        labels, regression_targets = self.prepare_targets(proposals, targets)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
            labels, regression_targets, proposals
        ):
            if labels_per_image.shape[0] != proposals_per_image.bbox3d.shape[0]:
              import pdb; pdb.set_trace()  # XXX BREAKPOINT
              pass
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        # rm ignored proposals
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, box_regression, targets=None):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])
            targets for debuging only

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        proposals = cat_boxlist_3d(proposals, per_example=True)
        labels = proposals.get_field("labels")
        regression_targets = proposals.get_field("regression_targets")
        pro_bbox3ds = proposals.bbox3d

        if not self.need_seperate:
          classification_loss = F.cross_entropy(class_logits, labels)
          box_loss = self.box_loss(labels, box_regression, regression_targets, pro_bbox3ds)
        else:
          classification_loss = self.seperate_classifier.roi_cross_entropy_seperated(class_logits, labels, proposals)
          box_loss = self.seperate_classifier.roi_box_loss_seperated(self.box_loss,
                              labels, box_regression, regression_targets, pro_bbox3ds)

        if SHOW_ROI_CLASSFICATION:
          self.show_roi_cls_regs(proposals, classification_loss, box_loss, class_logits,  targets, box_regression, regression_targets)

        return classification_loss, box_loss


    def box_loss(self, labels, box_regression, regression_targets, bbox3ds):
        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        device = box_regression.device
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        map_inds = 7 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3, 4, 5, 6], device=device)
        box_regression_pos = box_regression[sampled_pos_inds_subset[:, None], map_inds]
        regression_targets_pos = regression_targets[sampled_pos_inds_subset]

        if CHECK_REGRESSION_TARGET_YAW:
            roi_target_yaw = regression_targets_pos[:,-1]
            print(f'max_roi_target_yaw: {roi_target_yaw.max()}')
            print(f'min_roi_target_yaw: {roi_target_yaw.min()}')
            assert roi_target_yaw.max() < 1.5
            assert roi_target_yaw.min() > -1.5

        box_loss = smooth_l1_loss(
            box_regression_pos,
            regression_targets_pos,
            bbox3ds[sampled_pos_inds_subset],
            size_average=False,
            beta=1 / 5.,  # 1
            yaw_loss_mode = self.yaw_loss_mode
        )
        box_loss = box_loss / labels.numel()
        return box_loss

    def show_roi_cls_regs(self, proposals, classification_loss, box_loss,
              class_logits, targets,  box_regression, regression_targets):
          '''
          From rpn nms: FP, FN, TP
          ROI: (1)remove all FP (2) add all FN, (3) keep all TP
          '''
          assert proposals.batch_size() == 1
          targets = cat_boxlist_3d(targets, per_example=True)
          roi_class_pred = F.softmax(class_logits)
          pred_logits = torch.argmax(class_logits, 1)
          labels = proposals.get_field("labels")
          metric_inds, metric_evals = proposals.metric_4areas(self.low_threshold, self.high_threshold)
          gt_num = len(targets)
          device = class_logits.device
          num_classes = class_logits.shape[1]

          class_err = (labels != pred_logits).sum()

          print('\n-----------------------------------------\n roi classificatio\n')
          print(f"RPN_NMS: {metric_evals}")
          print(f"classification_loss:{classification_loss}, box_loss: {box_loss}")

          def show_one_type(eval_type):
              indices = metric_inds[eval_type]
              if eval_type == 'TP' and self.add_gt_proposals:
                indices = indices[0:-gt_num]
              n0 = indices.shape[0]
              pro_ = proposals[indices]
              objectness_ = pro_.get_field('objectness')
              logits_ = pred_logits[indices]
              labels_ = labels[indices]

              err_ = torch.abs(logits_ - labels_)
              err_num = err_.sum()
              print(f"\n * * * * * * * * \n{eval_type} :{n0} err num: {err_num}")
              print(f"objectness_:{objectness_}\n")
              if n0 > 0:
                roi_class_pred_ = roi_class_pred[indices[:,None], labels_[:,None]]

                #if eval_type != 'TP':
                print(f"roi_class_pred_:\n{roi_class_pred_}")

                if eval_type == 'FP':
                  pro_.show__together(targets)
                  pass

                if eval_type == 'FN' or eval_type == 'TP':
                  map_inds_ = 7 * labels_[:, None] + torch.tensor([0, 1, 2, 3, 4, 5, 6], device=device)
                  roi_box_regression_ = box_regression[indices[:,None], map_inds_]
                  roi_box = self.box_coder.decode(roi_box_regression_, pro_.bbox3d)
                  tar_reg = regression_targets[indices]
                  #roi_box = self.box_coder.decode(tar_reg, pro_.bbox3d)
                  print(f"target reg: \n{tar_reg[0:3]}")
                  print(f"roi_reg: \n{roi_box_regression_[0:3]}")

                  roi_box[:,0] += 15
                  roi_boxlist_ = pro_.copy()
                  roi_boxlist_.bbox3d = roi_box

                  targets_ = targets.copy()
                  targets_.bbox3d[:,0] += 15

                  bs_ = cat_boxlist_3d([pro_, roi_boxlist_], per_example = False)
                  tg_ = cat_boxlist_3d([targets, targets_], False)
                  bs_.show__together(tg_, twolabels=True)

                  pass
              pass

          show_one_type('FP')
          show_one_type('FN')
          show_one_type('TP')
          return


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder3D(is_corner_roi = cfg.MODEL.CORNER_ROI,  weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )
    yaw_loss_mode = cfg.MODEL.LOSS.YAW_MODE
    add_gt_proposals = cfg.MODEL.RPN.ADD_GT_PROPOSALS
    ay = cfg.MODEL.ROI_HEADS.LABEL_AUG_THICKNESS_Y_TAR_ANC
    az = cfg.MODEL.ROI_HEADS.LABEL_AUG_THICKNESS_Z_TAR_ANC
    aug_thickness = {'target_Y':ay[0], 'anchor_Y':ay[1], 'target_Z':az[0], 'anchor_Z':az[1]}
    in_classes = cfg.INPUT.CLASSES
    num_input_classes = len(in_classes)

    seperate_classes = cfg.MODEL.SEPARATE_CLASSES_ID
    seperate_classifier = SeperateClassifier( seperate_classes, num_input_classes )

    loss_evaluator = FastRCNNLossComputation(matcher, fg_bg_sampler, box_coder,
                    yaw_loss_mode, add_gt_proposals, aug_thickness, seperate_classifier)

    return loss_evaluator, seperate_classifier

