# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

import _C


def convert_3d_to_2d(input):
  import sparseconvnet as scn
  spatial_size = input.spatial_size

  nPlane0 = input.features.shape[1]+1
  to_dense_layer =  scn.sparseToDense.SparseToDense(dimension=4, nPlanes=nPlane0)
  features_2d = to_dense_layer(input)
  features_2d = features_2d.squeeze(4) # [batch_size, channels_num, w,h]
  return features_2d


class _ROIAlign3D(Function):
    @staticmethod
    def forward(ctx, input_s3d, roi, output_size, spatial_scale, sampling_ratio):
        input_2d = convert_3d_to_2d(input_s3d)

        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input_2d.size()
        output = _C.roi_align_forward(
            input_2d, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_align_backward(
            grad_output,
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
            sampling_ratio,
        )
        return grad_input, None, None, None, None


roi_align_3d = _ROIAlign3D.apply


class ROIAlign3D(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(ROIAlign3D, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        return roi_align_3d(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr
