# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch, math
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from sparseconvnet.tools_3d_2d import sparse_3d_to_dense_2d
import _C


class _ROIAlignRotated3D(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        # input: [4, 256, 304, 200]
        # roi: [171, 5]
        # spatial_scale: 0.25
        # output_size: [7,7]
        # sampling_ratio: 2
        output = _C.roi_align_rotated_forward(
            input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio
        ) # [171, 256, 7, 7]
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_align_rotated_backward(
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


roi_align_rotated_3d = _ROIAlignRotated3D.apply


class ROIAlignRotated3D(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        '''
        output_size:[pooled_height, pooled_width]
        spatial_scale: size_of_map/size_of_original_image
        sampling_ratio: how many points to use for bilinear_interpolate
        '''
        super(ROIAlignRotated3D, self).__init__()
        self.output_size = output_size # (7,7)
        self.spatial_scale = spatial_scale # 0.25
        self.sampling_ratio = sampling_ratio # 2

    def forward(self, input_s3d, rois_3d):
        '''
        input0: sparse 3d tensor
        rois_3d: 3d box, xyz order is same as input0,
                yaw unit is rad, anti-clock wise is positive

        input: [batch_size, feature, h, w]
        rois: [n,5] [batch_ind, center_w, center_h, roi_width, roi_height, theta]
        theta unit: degree, anti-clock wise is positive

        Note: the order of w and h inside of input and rois is different.
        '''
        input_d3d = sparse_3d_to_dense_2d(input_s3d)
        batch_size, channel0, xs,ys,zs = input_d3d.shape
        input_d3d = input_d3d.permute(0, 1, 4, 2, 3)
        input_d3d = input_d3d.reshape(batch_size, channel0*zs, xs, ys)
        rois_2d = rois_3d[:,[0, 2,1, 5,4,7]] # reverse the order of x and y
        rois_2d[:,-1]  *= 180.0/math.pi
        output = roi_align_rotated_3d(
            input_d3d, rois_2d, self.output_size, self.spatial_scale, self.sampling_ratio
        )
        pro_n,c,xo,yo = output.shape
        output = output.reshape(pro_n, channel0, zs, xo,yo).permute(0,1,3,4,2)
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        return output

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr

