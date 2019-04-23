# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
from maskrcnn_benchmark.structures.bounding_box_3d import BoxList3D, cat_boxlist_3d
from .suncg_utils.suncg_meta import SUNCG_META
from utils3d.bbox3d_ops import Bbox3D
from .suncg_utils.suncg_dataset import SUNCGDataset

DEBUG = True


def bbox_dic_to_BoxList3D(bbox_dic, size3d):
  bboxes = []
  labels = []
  for obj in bbox_dic:
    bboxes.append(bbox_dic[obj])
    label_i = SUNCG_META.class_2_label[obj]
    assert np.all(label_i>0), "label >1, 0 is for negative, -1 is ignore"
    labels.append(np.array([label_i]*bbox_dic[obj].shape[0]))
  bboxes = np.concatenate(bboxes, 0)
  labels = np.concatenate(labels, 0)

  examples_idxscope = torch.tensor([0, bboxes.shape[0]]).view(1,2)
  bboxlist3d = BoxList3D(bboxes, size3d=size3d, mode='yx_zb',
                        examples_idxscope=examples_idxscope, constants={})
  bboxlist3d.add_field('labels', labels)
  return bboxlist3d

DATASET = 'SUNCG'
def make_data_loader(cfg, is_train, is_distributed=False, start_iter=0):
  batch_size = cfg.SOLVER.IMS_PER_BATCH if is_train else cfg.TEST.IMS_PER_BATCH

  split = 'train' if is_train else 'val'
  dataset_ = SUNCGDataset(split, cfg)


  def trainMerge(data_ls):
    locs = torch.cat( [data['x'][0] for data in data_ls], 0 )
    pns = [data['x'][0].shape[0] for data in data_ls]
    batch_size = len(data_ls)
    batch_ids = torch.cat([torch.LongTensor(pns[i],1).fill_(i) for i in range(batch_size)], 0)
    locs = torch.cat([locs, batch_ids], 1)

    feats = torch.cat( [data['x'][1] for data in data_ls], 0 )
    labels = [data['y'] for data in data_ls]
    ids = [data['id'] for data in data_ls]
    data = {'x': [locs,feats], 'y': labels, 'id': ids}
    return data

  train_data_loader = torch.utils.data.DataLoader(
      dataset_, batch_size=batch_size, collate_fn=trainMerge, num_workers=20*(1-DEBUG), shuffle=is_train)


  return train_data_loader


#def locations_to_position(locations, voxel_scale):
#  return [location_to_position(loc, voxel_scale) for loc in locations]
#
#def location_to_position(location, voxel_scale):
#  assert location.shape[1] == 4
#  return location[:,0:3].float() / voxel_scale
#
