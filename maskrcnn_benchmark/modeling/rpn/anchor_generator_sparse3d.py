# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import numpy as np
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box_3d import BoxList3D
from data3d.data import locations_to_position
from utils3d.geometric_torch import OBJ_DEF

DEBUG = False
SHOW_ANCHOR_EACH_SCALE = DEBUG and False
if DEBUG:
  from utils3d.bbox3d_ops import Bbox3D

class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(
        self,
        voxel_scale=20,
        sizes_3d=[[0.2,1,3], [0.5,2,3], [1,3,3]],
        yaws=(0, -1.57),
        anchor_strides=[[8,8,729], [16,16,729], [32,32,729]],
        straddle_thresh=0,
    ):
        super(AnchorGenerator, self).__init__()

        sizes_3d = np.array(sizes_3d, dtype=np.float32)
        assert sizes_3d[0,0] >= sizes_3d[-1,0], "should be from the last scale"
        assert sizes_3d.shape[1] == 3
        anchor_strides = np.array(anchor_strides, dtype=np.float32)
        assert anchor_strides.shape[1] == 3
        assert sizes_3d.shape[0] == anchor_strides.shape[0]
        yaws = np.array(yaws, dtype=np.float32).reshape([-1,1])

        cell_anchors = [ generate_anchors_3d(size, yaws).float()
                        for size in sizes_3d]
        [OBJ_DEF.check_bboxes(ca, yx_zb=True) for ca in cell_anchors]
        for anchors in cell_anchors:
          anchors[:,2] += anchors[:,5]*0.5
        self.yaws = yaws
        self.anchor_num_per_loc = len(yaws) # only one size per loc
        self.voxel_scale = voxel_scale
        self.strides = torch.from_numpy(anchor_strides)
        self.cell_anchors = BufferList(cell_anchors)
        self.straddle_thresh = straddle_thresh
        self.anchor_mode = 'yx_zb'

    def num_anchors_per_location(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, locations):
        anchors = []
        assert len(self.cell_anchors) == len(locations), "scales num not right"
        for base_anchors, location, stride in zip(
            self.cell_anchors, locations, self.strides
        ):
            anchor_centroids = (location[:,0:3].float()+0) / self.voxel_scale * stride.view(1,3)
            anchor_centroids = torch.cat([anchor_centroids, torch.zeros(anchor_centroids.shape[0],4)], 1)
            anchor_centroids = anchor_centroids.view(-1,1,7)
            base_anchors = base_anchors.view(1,-1,7)

            device = base_anchors.device
            # got standard anchors
            #flatten order [sparse_location_num, yaws_num, 7]
            anchors_scale = anchor_centroids.to(device) + base_anchors
            anchors_scale = anchors_scale.reshape(-1,7)
            anchors.append( anchors_scale )

        if DEBUG and False:
          Bbox3D.draw_bboxes(anchors[0].cpu().numpy(), 'Z', False)
        return anchors

    def add_visibility_to(self, boxlist):
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox
        if self.straddle_thresh >= 0:
            inds_inside = (
                (anchors[..., 0] >= -self.straddle_thresh)
                & (anchors[..., 1] >= -self.straddle_thresh)
                & (anchors[..., 2] < image_width + self.straddle_thresh)
                & (anchors[..., 3] < image_height + self.straddle_thresh)
            )
        else:
            device = anchors.device
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.uint8, device=device)
        boxlist.add_field("visibility", inds_inside)

    def forward(self, points_sparse, feature_maps_sparse):
        '''
          Note: all the batches are concatenated together
        '''
        #grid_sizes = [feature_map.spatial_size for feature_map in feature_maps_sparse]
        locations = [feature_map.get_spatial_locations() for feature_map in feature_maps_sparse]
        anchors_over_all_feature_maps_sparse = self.grid_anchors(locations)
        examples_idxscope = [examples_bidx_2_sizes(f.get_spatial_locations()[:,-1]) * self.anchor_num_per_loc
                              for f in feature_maps_sparse]
        size3d = sparse_points_scope(points_sparse)
        anchors = [BoxList3D(a, size3d, "yx_zb", ei) \
                      for a,ei in zip(anchors_over_all_feature_maps_sparse, examples_idxscope)]
        anchors = [a.convert(self.anchor_mode) for a in anchors]

        if SHOW_ANCHOR_EACH_SCALE:
            scale_num = len(anchors)
            for scale_i in range(scale_num):
              print(f'scale {scale_i} {len(anchors[scale_i])} enchors')
              min_xyz = anchors[scale_i].bbox3d[:,0:3].min(0)
              mean_xyz = anchors[scale_i].bbox3d[:,0:3].mean(0)
              max_xyz = anchors[scale_i].bbox3d[:,0:3].max(0)
              print(f'anchor ctr min: {min_xyz}')
              print(f'anchor ctr mean: {mean_xyz}')
              print(f'anchor ctr max: {max_xyz}')
              points = points_sparse[1][:,0:3].cpu().data.numpy()
              print(f'points ctr min: {points.min(0)}')
              print(f'points ctr mean: {points.mean(0)}')
              print(f'points ctr max: {points.max(0)}')

              anchors[scale_i].show(20, points, with_centroids=True)
              #anchors[scale_i].show_centroids(-1, points)
              import pdb; pdb.set_trace()  # XXX BREAKPOINT
        return anchors


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


def sparse_points_scope(points_sparse):
  batch_idx = points_sparse[0][:,-1]
  points = points_sparse[1][:,0:3]
  batch_size = batch_idx[-1] + 1
  s = torch.tensor(0)
  e = torch.tensor(0)
  size3d = []
  for bi in range(batch_size):
    e += torch.sum(batch_idx==bi)
    xyz = points[s:e]
    s = e.clone()
    xyz_min = xyz.min(0)[0]
    xyz_max = xyz.max(0)[0]
    size3d.append( torch.cat([xyz_min, xyz_max], 0).view(1,6) )
  size3d = torch.cat(size3d, 0)
  return size3d


def make_anchor_generator(config):
    anchor_sizes_3d = config.MODEL.RPN.ANCHOR_SIZES_3D
    yaws = config.MODEL.RPN.YAWS
    anchor_stride = config.MODEL.RPN.ANCHOR_STRIDE
    straddle_thresh = config.MODEL.RPN.STRADDLE_THRESH
    voxel_scale = config.SPARSE3D.VOXEL_SCALE
    voxel_full_scale = config.SPARSE3D.VOXEL_FULL_SCALE

    if config.MODEL.RPN.USE_FPN:
        assert len(anchor_stride) == len(
            anchor_sizes_3d
        ), "FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)"
    else:
        assert len(anchor_stride) == 1, "Non-FPN should have a single ANCHOR_STRIDE"
    anchor_generator = AnchorGenerator(
        voxel_scale, anchor_sizes_3d, yaws, anchor_stride, straddle_thresh
    )
    return anchor_generator


# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#        [-175.,  -87.,  192.,  104.],
#        [-359., -183.,  376.,  200.],
#        [ -55.,  -55.,   72.,   72.],
#        [-119., -119.,  136.,  136.],
#        [-247., -247.,  264.,  264.],
#        [ -35.,  -79.,   52.,   96.],
#        [ -79., -167.,   96.,  184.],
#        [-167., -343.,  184.,  360.]])



def generate_anchors_3d( size, yaws, centroids=np.array([[0,0,0]])):
    anchors = []
    for j in range(yaws.shape[0]):
        for k in range(centroids.shape[0]):
          anchor = np.concatenate([centroids[k], size, yaws[j]]).reshape([1,-1])
          anchors.append(anchor)
    anchors = np.concatenate(anchors, 0)
    return torch.from_numpy( anchors )

def generate_anchors(
    sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_anchors(
        np.array(sizes, dtype=np.float),
        np.array(aspect_ratios, dtype=np.float),
    )


def _generate_anchors(base_size, aspect_ratios):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([-1,-1,-1, 1, 1, 1], dtype=np.float)*0.5*base_size
    anchors = _ratio_enum(anchor, aspect_ratios)
    return torch.from_numpy(anchors)


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    X = anchor[3] - anchor[0]
    Y = anchor[4] - anchor[1]
    Z = anchor[5] - anchor[2]
    x_ctr = anchor[0] + 0.5 * (X)
    y_ctr = anchor[1] + 0.5 * (Y)
    z_ctr = anchor[2] + 0.5 * (Z)
    return X, Y, Z, x_ctr, y_ctr, z_ctr


def _mkanchors(Xs, Ys, Zs, x_ctr, y_ctr, z_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    Xs = Xs[:, np.newaxis]
    Ys = Ys[:, np.newaxis]
    Zs = Zs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (Xs),
            y_ctr - 0.5 * (Ys),
            z_ctr - 0.5 * (Zs),
            x_ctr + 0.5 * (Xs),
            y_ctr + 0.5 * (Ys),
            z_ctr + 0.5 * (Zs),
        )
    )
    return anchors


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor.
      Keep Z unchanged"""
    X, Y, Z, x_ctr, y_ctr, z_ctr = _whctrs(anchor)
    size = X * Y
    size_ratios = size / ratios
    Xs = np.sqrt(size_ratios)
    Ys = Xs * ratios
    Zs = np.tile(np.expand_dims(Z,0), ratios.shape[0])
    anchors = _mkanchors(Xs, Ys, Zs, x_ctr, y_ctr, z_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    X, Y, Z, x_ctr, y_ctr, z_ctr = _whctrs(anchor)
    Xs = X * scales
    Ys = Y * scales
    Zs = Z * scales
    anchors = _mkanchors(Xs, Ys, Zs, x_ctr, y_ctr, z_ctr)
    return anchors
