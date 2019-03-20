# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.box_coder_3d import BoxCoder3D
from .loss_3d import make_rpn_loss_evaluator
from .anchor_generator_sparse3d import make_anchor_generator
from .inference_3d import make_rpn_postprocessor
from maskrcnn_benchmark.structures.boxlist3d_ops import cat_boxlist_3d

DEBUG = False
SHOW_TARGETS_ANCHORS = DEBUG and False

def combine_anchor_scales(anchors):
    '''
     combine anchors of scales
     anchors: list(BoxList)
     anchors_new: BoxList
    '''
    scale_num = len(anchors)
    batch_size = anchors[0].batch_size()
    anchors_scales = []
    for s in range(scale_num):
      anchors_scales.append( anchors[s].seperate_examples() )
    examples = []
    for b in range(batch_size):
      examples.append( cat_boxlist_3d([a[b] for a in anchors_scales] ) )
    anchors_all_scales = cat_boxlist_3d(examples, per_example=True)
    return anchors_all_scales

@registry.RPN_HEADS.register("SingleConvRPNHead_Sparse3D")
class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=1, stride=1, padding=0  )
                #in_channels, in_channels, kernel_size=3, stride=1, padding=1  )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 7, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg):
        super(RPNModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator(cfg)

        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        head = rpn_head(
            cfg, in_channels, anchor_generator.num_anchors_per_location()[0]
        )

        rpn_box_coder = BoxCoder3D()

        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)

        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, points_sparse, features_sparse, targets=None):
        """
        Arguments:
            points_sparse (ImageList): points_sparse for which we want to compute the predictions
            features (list[Tensor]): features computed from the points_sparse that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        def reshape(f):
          return f.t().unsqueeze(0).unsqueeze(3)
        features = [fs.features for fs in features_sparse]
        features = [reshape(f) for f in features]
        #[print(f.shape) for f in features]
        objectness, rpn_box_regression = self.head(features)
        anchors = self.anchor_generator(points_sparse, features_sparse)
        anchors = combine_anchor_scales(anchors)

        if SHOW_TARGETS_ANCHORS:
            import numpy as np
            batch_size = len(targets)
            examples_scope = examples_bidx_2_sizes(points_sparse[0][:,-1])
            for bi in range(batch_size):
              se = examples_scope[bi]
              points = points_sparse[1][se[0]:se[1],0:3].cpu().data.numpy()
              print(f'\n targets')
              targets[bi].show(points=points)
              anchor_num = len(anchors)
              for i in np.random.choice(anchor_num, 5):
                anchor_i = anchors[int(i)].example(bi)
                print(f'\n anchor {i} / {anchor_num}')
                anchor_i.show_together(targets[bi], 200, points=points)
              import pdb; pdb.set_trace()  # XXX BREAKPOINT
              pass

        if self.training:
            return self._forward_train(anchors, objectness, rpn_box_regression, targets)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            with torch.no_grad():
                boxes = self.box_selector_train(
                    anchors, objectness, rpn_box_regression, targets
                )
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
            anchors, objectness, rpn_box_regression, targets
        )
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}


def build_rpn(cfg):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    return RPNModule(cfg)

def examples_bidx_2_sizes(examples_bidx):
  batch_size = examples_bidx[-1]+1
  s = torch.tensor(0)
  e = torch.tensor(0)
  examples_idxscope = []
  for bi in range(batch_size):
    e += torch.sum(examples_bidx==bi)
    examples_idxscope.append(torch.stack([s,e]).view(1,2))
    s = e.clone()
  examples_idxscope = torch.cat(examples_idxscope, 0)
  return examples_idxscope

