# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch
from second.pytorch.core.box_torch_ops import second_box_encode, second_box_decode

class BoxCoder3D(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.smooth_dim = True


    def encode(self, proposals, reference_boxes):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes : anchors
            proposals (Tensor): boxes to be encoded
        """
        return second_box_encode(proposals, reference_boxes, smooth_dim=self.smooth_dim)

    def decode(self, box_encodings, anchors):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        return second_box_decode(box_encodings, anchors, smooth_dim=self.smooth_dim)
