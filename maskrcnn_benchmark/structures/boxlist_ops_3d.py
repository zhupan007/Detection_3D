# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np

from maskrcnn_benchmark.structures.bounding_box_3d import BoxList3D, cat_boxlist_3d

from maskrcnn_benchmark.layers import nms as _box_nms
from second.pytorch.core.box_torch_ops import rotate_nms, \
                              multiclass_nms

from second.core.non_max_suppression.nms_gpu import rotate_iou_gpu_eval

DEBUG = False

def boxlist_nms_3d(boxlist, nms_thresh, max_proposals=-1, score_field="score"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maxium suppression
        score_field (str)
    """
    if max_proposals<0:
      max_proposals = 500
    bbox2d = boxlist.bbox3d[:,[0,1,3,4,6]]
    objectness = boxlist.get_field(score_field)
    keep = rotate_nms(
              bbox2d,
              objectness,
              pre_max_size=2000,
              post_max_size=max_proposals,
              iou_threshold=nms_thresh, # 0.1
               )
    boxlist = boxlist[keep]
    return boxlist

def remove_small_boxes3d(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]


def iou_one_dim(targets_z, anchors_z):
    '''
    For ceiling, and floor: z size of target is small, augment to 1
    '''
    targets_z[:,1] = torch.clamp(targets_z[:,1], min=0.8)
    anchors_z[:,1] = torch.clamp(anchors_z[:,1], min=0.8) # aug proposal for ROI input as well

    anchors_z[:,1] = anchors_z[:,0] + anchors_z[:,1]
    targets_z[:,1] = targets_z[:,0] + targets_z[:,1]
    targets_z = targets_z.unsqueeze(1)
    anchors_z = anchors_z.unsqueeze(0)
    overlap = torch.min(anchors_z[:,:,1], targets_z[:,:,1]) - torch.max(anchors_z[:,:,0], targets_z[:,:,0])
    common = torch.max(anchors_z[:,:,1], targets_z[:,:,1]) - torch.min(anchors_z[:,:,0], targets_z[:,:,0])
    iou_z = overlap / common
    return iou_z

def boxlist_iou_3d(targets, anchors, aug_thickness, criterion, only_xy=False, flag=''):
  '''
  about criterion check:
    /home/z/Research/Detection_3D/second/core/non_max_suppression/nms_gpu.py devRotateIoUEval
  '''
  assert targets.mode == 'yx_zb'
  assert anchors.mode == 'yx_zb'

  assert isinstance(aug_thickness, dict)
  assert 'target' in aug_thickness
  assert 'anchor' in aug_thickness

  iouz = iou_one_dim(targets.bbox3d[:,[2,5]].clone(), anchors.bbox3d[:,[2,5]].clone())

  cuda_index = targets.bbox3d.device.index
  anchors_2d = anchors.bbox3d[:,[0,1,3,4,6]].cpu().data.numpy()
  targets_2d = targets.bbox3d[:,[0,1,3,4,6]].cpu().data.numpy()

  #print(f"targets yaw : {targets_2d[:,-1].min()} , {targets_2d[:,-1].max()}")
  #print(f"anchors yaw : {anchors_2d[:,-1].min()} , {anchors_2d[:,-1].max()}")

  # aug thickness. When thickness==0, iou is wrong
  targets_2d[:,2] = np.clip(targets_2d[:,2], a_min=aug_thickness['target'], a_max=None)
  anchors_2d[:,2] = np.clip(anchors_2d[:,2], a_min=aug_thickness['anchor'], a_max=None)

  #aug_th_mask = (targets_2d[:,2] < 0.3).astype(np.float32)
  #targets_2d[:,2] += aug_thickness['target'] * aug_th_mask  # 0.25
  #aug_th_mask = (anchors_2d[:,2] < 0.3).astype(np.float32)
  #anchors_2d[:,2] += aug_thickness['anchor'] * aug_th_mask

  # criterion=1: use targets_2d as ref
  iou2d = rotate_iou_gpu_eval(targets_2d, anchors_2d, criterion=criterion, device_id=cuda_index)
  iou2d = torch.from_numpy(iou2d)
  iou2d = iou2d.to(targets.bbox3d.device)

  if only_xy:
      iou3d = iou2d
  else:
      iou3d = iou2d * iouz

  if DEBUG and flag=='eval':
    if iou3d.max() < 1:
        return iou3d

    mask = iou3d == iou3d.max()
    t_i, a_i = torch.nonzero(mask)[0]
    t = targets[t_i]
    a = anchors[a_i]
    print(f"max iou: {iou3d.max()}")
    a.show_together(t)

    if iou3d.max() > 1:
        torch.set_printoptions(precision=16)
        print(a.bbox3d)
        print(t.bbox3d)

        #np.set_printoptions(precision=10)
        #print(a.bbox3d.cpu().data.numpy())
        #print(t.bbox3d.cpu().data.numpy())

        areas = rotate_iou_gpu_eval(targets_2d, anchors_2d, criterion=3, device_id=cuda_index)
        ious0 = rotate_iou_gpu_eval(targets_2d, anchors_2d, criterion=0, device_id=cuda_index)
        ious1 = rotate_iou_gpu_eval(targets_2d, anchors_2d, criterion=1, device_id=cuda_index)
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        areas_max = areas[t_i, a_i]
        import pdb; pdb.set_trace()  # XXX BREAKPOINT


    iou_preds = iou3d.max(0)[0]
    mask = iou3d == iou_preds.min()
    t_i, a_i = torch.nonzero(mask)[0]
    t = targets[t_i]
    a = anchors[a_i]
    print(f"min pred iou: {iou_preds.min()}")
    a.show_together(t)
    anchors.show_highlight([a_i])
    targets.show_highlight([t_i])
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass

  return iou3d

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications



def test_iou_3d(bbox3d0, bbox3d1, mode):
  '''
  bbox3d: [N,7]
  '''
  boxlist3d0 = BoxList3D(bbox3d0, size3d = None, mode=mode, examples_idxscope = None, constants={})
  boxlist3d1 = BoxList3D(bbox3d1, size3d = None, mode=mode, examples_idxscope = None, constants={})
  boxlist3d0 = boxlist3d0.convert("yx_zb")
  boxlist3d1 = boxlist3d1.convert("yx_zb")

  aug_thickness = {'target':0, 'anchor':0}
  ious = boxlist_iou_3d(boxlist3d0, boxlist3d1, aug_thickness, -1)
  ious_diag = ious.diag()

  print(f"ious_diag: {ious_diag}")

  err_mask = torch.abs(ious_diag - 1) > 0.01
  err_inds = torch.nonzero(err_mask).view(-1)
  #print(f"ious:{ious}")
  print(f"err_inds: {err_inds}")
  #print(err_boxlist.bbox3d)
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass


def main_test_iou_3d():
  '''
  small yaw, small thickness
  '''
  device = torch.device('cuda:0')
  #[ 43.9440, -40.0217,   0.0000,   0.0947,   2.4079,   2.7350,  -1.5549],
  #[ 43.9400, -45.1191,   0.0000,   0.0947,   2.4011,   2.7350,  -1.5550],

  bbox3d0 = torch.tensor([
   [0,0,   0.0000,   0.001,   2.,   2.,  0],
   [0,0,   0.0000,   0.01,   2.,   2.,  0],

   [0,0,   0.0000,   0.001,   2.,   2.,  np.pi/2],
   [0,0,   0.0000,   0.01,   2.,   2.,  np.pi/2],


   [0,0,   0.0000,   0.1,   2.,   2.,  np.pi/2],
   [0,0,   0.0000,   1,   2.,   2.,  np.pi/2],
      [ 2.3569,  7.0700, -0.0300,  0.0947,  1.8593,  2.7350,  0.0000],
   ],
   dtype=torch.float32
  )

  bbox3d1 = torch.tensor([
        #[ 3.9165,  2.2180, -0.0500,  0.0947,  5.0945,  2.7350, -1.5708],
        [ 7.2792,  0.2153, -0.0500,  0.0947,  1.4407,  2.7350, -1.5708],
        [ 6.5114,  0.1272, -0.0500,  0.0947,  0.2544,  2.7350,  0.0000]
    ],
   dtype=torch.float32
  )

  bbox3d1 = torch.tensor([
      [ 2.3569,  7.0700, -0.0300,  0.0947,  1.8593,  2.7350,  0.0000],
        [ 1.1548,  6.1797, -0.0300,  0.0947,  2.3096,  2.7350, -1.5708]
    ],
   dtype=torch.float32
  )

  bbox3d0 = bbox3d0.to(device)
  bbox3d1 = bbox3d1.to(device)


  test_iou_3d(bbox3d1, bbox3d1, 'yx_zb')

def main1_test_iou_3d():
  device = torch.device('cuda:0')

  bbox3d0 = torch.tensor([
      [1.2175720215e+01, 7.8515229225e+00, 5.2835583687e-02, 9.6419714391e-02,
         3.1705775261e+00, 2.7384383678e+00, 7.3978723958e-04]
    ],
   dtype=torch.float32
  )

  bbox3d1 = torch.tensor([
      [12.1804752350,  7.8437194824,  0.0490041152,  0.0947349519,
          3.1549880505,  2.7349998951,  0.0000000000]
    ],
   dtype=torch.float32
  )

  bbox3d0 = bbox3d0.to(device)
  bbox3d1 = bbox3d1.to(device)


  test_iou_3d(bbox3d1, bbox3d1, 'yx_zb')

if __name__ == '__main__':
  main1_test_iou_3d()


