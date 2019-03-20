# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
from maskrcnn_benchmark.structures.bounding_box_3d import BoxList3D
from .suncg_meta import SUNCG_META
from utils3d.bbox3d_ops import Bbox3D

DEBUG = False


def bbox_dic_to_BoxList3D(bbox_dic, size3d):
  bboxes = []
  labels = []
  for obj in bbox_dic:
    bboxes.append(bbox_dic[obj])
    label_i = SUNCG_META.class_2_label[obj]
    labels.append(np.array([label_i]*bbox_dic[obj].shape[0]))
  bboxes = np.concatenate(bboxes, 0)
  labels = np.concatenate(labels, 0)

  examples_idxscope = torch.tensor([0, bboxes.shape[0]]).view(1,2)
  bboxlist3d = BoxList3D(bboxes, size3d=size3d, mode='yx_zb',
                        examples_idxscope=examples_idxscope)
  bboxlist3d.add_field('label', labels)
  return bboxlist3d

DATASET = 'SUNCG'
def make_data_loader(cfg, is_train, is_distributed=False, start_iter=0):
  scale = cfg.SPARSE3D.VOXEL_SCALE
  full_scale=cfg.SPARSE3D.VOXEL_FULL_SCALE
  val_reps = cfg.SPARSE3D.VAL_REPS
  batch_size = cfg.SOLVER.IMS_PER_BATCH
  objects_to_detect = cfg.INPUT.OBJECTS
  dimension=3

  full_scale = np.array(full_scale)
  assert full_scale.shape == (3,)

  # VALID_CLAS_IDS have been mapped to the range {0,1,...,19}
  #VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

  def get_files_Scannet(split):
    import os
    cur_path = os.path.dirname(os.path.abspath(__file__))
    dset_path = f'{cur_path}/ScanNetTorch'
    with open(f'{cur_path}/Benchmark_Small/scannetv1_{split}.txt') as f:
      scene_names = [l.strip() for l in f.readlines()]
    files = [f'{dset_path}/{scene}/{scene}_vh_clean_2.pth' for scene in scene_names]
    return files

  def get_files_Suncg(split):
    import os
    cur_path = os.path.dirname(os.path.abspath(__file__))
    dset_path = f'{cur_path}/SuncgTorch'
    with open(f'{dset_path}/train_test_splited/{split}.txt') as f:
      scene_names = [l.strip() for l in f.readlines()]
    files = []
    for scene in scene_names:
      files += glob.glob(f'{dset_path}/houses/{scene}/*.pth')
    return files

  if DATASET == 'SCANNET':
    get_files = get_files_Scannet
  elif DATASET == 'SUNCG':
    get_files = get_files_Suncg

  train,val = [],[]
  for x in torch.utils.data.DataLoader(
        get_files('train'),
          collate_fn=lambda x: torch.load(x[0]), num_workers=mp.cpu_count()):
      train.append(x)
  for x in torch.utils.data.DataLoader(
        get_files('val'),
          collate_fn=lambda x: torch.load(x[0]), num_workers=mp.cpu_count()):
      val.append(x)
  print('Training examples:', len(train))
  print('Validation examples:', len(val))

  #Elastic distortion
  blur0=np.ones((3,1,1)).astype('float32')/3
  blur1=np.ones((1,3,1)).astype('float32')/3
  blur2=np.ones((1,1,3)).astype('float32')/3
  def elastic(x,gran,mag):
      bb=np.abs(x).max(0).astype(np.int32)//gran+3
      noise=[np.random.randn(bb[0],bb[1],bb[2]).astype('float32') for _ in range(3)]
      noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
      noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
      noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
      noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
      noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
      noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
      ax=[np.linspace(-(b-1)*gran,(b-1)*gran,b) for b in bb]
      interp=[scipy.interpolate.RegularGridInterpolator(ax,n,bounds_error=0,fill_value=0) for n in noise]
      def g(x_):
          return np.hstack([i(x_)[:,None] for i in interp])
      return x+g(x)*mag

  def trainMerge(tbl):
      zoom_rate = 0.1*0
      flip_x = False
      random_rotate = False
      distortion = False
      origin_offset = False
      feature_with_xyz = True
      norm_noise = 0.01

      locs=[]
      feats=[]
      labels=[]
      for idx,i in enumerate(tbl):
          pcl_i, bboxes_dic_i_0 = train[i]
          a = pcl_i[:,0:3].copy()
          b = pcl_i
          bboxes_dic_i = {}
          for obj in objects_to_detect:
            assert obj in bboxes_dic_i_0 or obj=='all'
          for obj in bboxes_dic_i_0:
            if ('all' in objects_to_detect) or (obj in objects_to_detect):
              bboxes_dic_i[obj] = Bbox3D.convert_to_yx_zb_boxes(bboxes_dic_i_0[obj])
          if DEBUG:
            show_pcl_boxdic(pcl_i, bboxes_dic_i)

          #---------------------------------------------------------------------
          # augmentation of xyz
          m=np.eye(3)+np.random.randn(3,3)*zoom_rate # aug: zoom
          if flip_x:
            m[0][0]*=np.random.randint(0,2)*2-1  # aug: x flip
          m*=scale
          if random_rotate:
            theta=np.random.rand()*2*math.pi # rotation aug
            m=np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
          a=np.matmul(a,m)
          if distortion:
            a=elastic(a,6*scale//50,40*scale/50)
            a=elastic(a,20*scale//50,160*scale/50)
          m=a.min(0)
          M=a.max(0)
          q=M-m
          # aug: the centroid between [0,full_scale]
          offset = -m
          if origin_offset:
            offset += np.clip(full_scale-M+m-0.001, 0, None) * np.random.rand(3)+np.clip(full_scale-M+m+0.001,None,0)*np.random.rand(3)
          a+=offset

          xyz_min = a.min(0) / scale
          xyz_max = a.max(0) / scale
          size3d = np.expand_dims(np.concatenate([xyz_min, xyz_max], 0), 0).astype(np.float32)
          size3d = torch.from_numpy(size3d)
          #---------------------------------------------------------------------
          # augmentation of feature
          # aug norm
          b[:,3:6] += np.random.randn(3)*norm_noise
          if feature_with_xyz:
            # import augmentation of xyz to feature
            b[:,0:3] = a / scale
          else:
            assert b.shape[1] > 3
            b = b[:,3:]

          #---------------------------------------------------------------------
          # augment gt boxes
          for obj in bboxes_dic_i:
            #print(bboxes_dic_i[obj][:,0:3])
            bboxes_dic_i[obj][:,0:3] += np.expand_dims(offset,0)/scale
            #print(bboxes_dic_i[obj][:,0:3])
            pass

          #---------------------------------------------------------------------
          up_check = np.all(a < full_scale[np.newaxis,:], 1)
          idxs = (a.min(1)>=0)*(up_check)
          assert np.all(idxs), "some points are missed in train"
          a=a[idxs]
          b=b[idxs]
          #c=c[idxs]
          a=torch.from_numpy(a).long()
          locs.append(torch.cat([a,torch.LongTensor(a.shape[0],1).fill_(idx)],1))
          feats.append(torch.from_numpy(b))

          #---------------------------------------------------------------------
          bboxlist3d = bbox_dic_to_BoxList3D(bboxes_dic_i, size3d)
          labels.append(bboxlist3d)
      locs=torch.cat(locs,0)
      feats=torch.cat(feats,0)

      #batch_scopes(locs, scale)
      return {'x': [locs,feats], 'y': labels, 'id': tbl}
  train_data_loader = torch.utils.data.DataLoader(
      list(range(len(train))),batch_size=batch_size, collate_fn=trainMerge, num_workers=20*(1-DEBUG), shuffle=True)


  return train_data_loader

  valOffsets=[0]
  valLabels=[]
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  for idx,x in enumerate(val):
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      valOffsets.append(valOffsets[-1]+x[2].size)
      valLabels.append(x[2].astype(np.int32))
  valLabels=np.hstack(valLabels)

  def valMerge(tbl):
      locs=[]
      feats=[]
      labels=[]
      point_ids=[]
      for idx,i in enumerate(tbl):
          a,b,c=val[i]
          m=np.eye(3)
          m[0][0]*=np.random.randint(0,2)*2-1
          m*=scale
          theta=np.random.rand()*2*math.pi
          m=np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
          a=np.matmul(a,m)+full_scale/2+np.random.uniform(-2,2,3)
          m=a.min(0)
          M=a.max(0)
          q=M-m
          offset=-m+np.clip(full_scale-M+m-0.001,0,None)*np.random.rand(3)+np.clip(full_scale-M+m+0.001,None,0)*np.random.rand(3)
          a+=offset
          idxs=(a.min(1)>=0)*(a.max(1)<full_scale)
          assert np.all(idxs), "some points are missed in val"
          a=a[idxs]
          b=b[idxs]
          c=c[idxs]
          a=torch.from_numpy(a).long()
          locs.append(torch.cat([a,torch.LongTensor(a.shape[0],1).fill_(idx)],1))
          feats.append(torch.from_numpy(b))
          labels.append(torch.from_numpy(c))
          point_ids.append(torch.from_numpy(np.nonzero(idxs)[0]+valOffsets[i]))
      locs=torch.cat(locs,0)
      feats=torch.cat(feats,0)
      labels=torch.cat(labels,0)
      point_ids=torch.cat(point_ids,0)
      return {'x': [locs,feats], 'y': labels.long(), 'id': tbl, 'point_ids': point_ids}
  val_data_loader = torch.utils.data.DataLoader(
      list(range(len(val))),batch_size=batch_size, collate_fn=valMerge, num_workers=20,shuffle=True)

  if is_train:
    return train_data_loader
  else:
    return val_data_loader


def locations_to_position(locations, voxel_scale):
  return [location_to_position(loc, voxel_scale) for loc in locations]

def location_to_position(location, voxel_scale):
  assert location.shape[1] == 4
  return location[:,0:3].float() / voxel_scale

def batch_scopes(location, voxel_scale):
  batch_size = torch.max(location[:,3])+1
  s = 0
  e = 0
  scopes = []
  for i in range(batch_size):
    e += torch.sum(location[:,3]==i)
    xyz = location[s:e,0:3].float() / voxel_scale
    s = e.clone()
    xyz_max = xyz.max(0)[0]
    xyz_min = xyz.min(0)[0]
    xyz_scope = xyz_max - xyz_min
    print(f"min:{xyz_min}  max:{xyz_max} scope:{xyz_scope}")
    scopes.append(xyz_scope)
  scopes = torch.cat(scopes, 0)
  return scopes


def show_pcl_boxdic(pcl, bboxes_dic):
  from utils3d.bbox3d_ops import Bbox3D
  boxes = []
  for obj in bboxes_dic:
    boxes.append(bboxes_dic[obj])
  boxes = np.concatenate(boxes, 0)
  Bbox3D.draw_points_bboxes(pcl[:,0:3], boxes, 'Z', is_yx_zb=True)
  import pdb; pdb.set_trace()  # XXX BREAKPOINT

