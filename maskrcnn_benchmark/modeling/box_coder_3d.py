# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch
from second.pytorch.core.box_torch_ops import second_box_encode, second_box_decode

class BoxCoder3D(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights=(1.0,)*7):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.smooth_dim = True
        weights = torch.tensor(weights).view(1,7)
        self.weights = weights
        if not self.smooth_dim:
          # Prevent sending too large values into torch.exp()
          bbox_xform_clip = math.log(1000. / 1)
        else:
          bbox_xform_clip = 10000. / 1
        self.bbox_xform_clip = bbox_xform_clip


    def encode(self, targets, anchors):
        box_encodings = second_box_encode(targets, anchors, smooth_dim=self.smooth_dim)
        box_encodings = box_encodings * self.weights.to(box_encodings.device)
        return box_encodings

    def decode(self, box_encodings, anchors):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        box_encodings = box_encodings / self.weights.to(box_encodings.device)
        box_encodings[:,3:6] = torch.clamp(box_encodings[:,3:6], max=self.bbox_xform_clip)
        return second_box_decode(box_encodings, anchors, smooth_dim=self.smooth_dim)
