# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
from .suncg_utils.suncg_dataset import SUNCGDataset
import logging

DEBUG = False

def make_data_loader(cfg, is_train, is_distributed=False, start_iter=0):
  batch_size = cfg.SOLVER.IMS_PER_BATCH if is_train else cfg.TEST.IMS_PER_BATCH

  split = 'train' if is_train else 'val'
  dataset_ = SUNCGDataset(split, cfg)
  cfg.INPUT['Example_num']=len(dataset_)
  logger = logging.getLogger("maskrcnn_benchmark.input")
  logger.info(f'\n\nexample num: {len(dataset_)}\n')

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
      dataset_, batch_size=batch_size, collate_fn=trainMerge, num_workers=10*(1-DEBUG), shuffle=is_train)


  return train_data_loader

