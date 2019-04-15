# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np

from bounding_box_3d import BoxList3D, cat_boxlist_3d

from maskrcnn_benchmark.layers import nms as _box_nms
from second.pytorch.core.box_torch_ops import rotate_nms, \
                              multiclass_nms

from second.core.non_max_suppression.nms_gpu import rotate_iou_gpu_eval

DEBUG = True

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
    bbox2d = boxlist.bbox3d[:,[0,1,3,4,6]]
    objectness = boxlist.get_field(score_field)
    keep = rotate_nms(
              bbox2d,
              objectness,
              pre_max_size=2000,
              post_max_size=100,
              iou_threshold=nms_thresh, # 0.1
               )
    #bbox2d = bbox2d.unsqueeze(1)
    #keep = multiclass_nms(
    #          nms_func = rotate_nms,
    #          boxes = bbox2d,
    #          scores = objectness,
    #          num_class = 2,
    #          pre_max_size=2000,
    #          post_max_size=100,
    #          iou_threshold=nms_thresh, # 0.1
    #          )
    boxlist = boxlist[keep]
    return boxlist

    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


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


def boxlist_iou_3d(targets, anchors, aug_wall_target_thickness):
  '''
  about criterion check:
    second.core.non_max_suppression.nms_gpu/devRotateIoUEval
  '''
  targets.mode == 'yx_zb'
  anchors.mode == 'yx_zb'
  cuda_index = targets.bbox3d.device.index
  anchors_2d = anchors.bbox3d[:,[0,1,3,4,6]].cpu().data.numpy()
  targets_2d = targets.bbox3d[:,[0,1,3,4,6]].cpu().data.numpy()

  #print(f"targets yaw : {targets_2d[:,-1].min()} , {targets_2d[:,-1].max()}")
  #print(f"anchors yaw : {anchors_2d[:,-1].min()} , {anchors_2d[:,-1].max()}")

  # aug thickness. When thickness==0, iou is wrong
  targets_2d[:,2] += aug_wall_target_thickness # 0.25
  # criterion=1: use targets_2d as ref
  iou = rotate_iou_gpu_eval(targets_2d, anchors_2d, criterion=2, device_id=cuda_index)

  area_inter = rotate_iou_gpu_eval(targets_2d, anchors_2d, criterion=-2, device_id=cuda_index)
  area_1 = rotate_iou_gpu_eval(targets_2d, anchors_2d, criterion=-3, device_id=cuda_index)
  area_2 = rotate_iou_gpu_eval(targets_2d, anchors_2d, criterion=-4, device_id=cuda_index)


  iou = torch.from_numpy(iou)
  iou = iou.to(targets.bbox3d.device)

  if DEBUG:
    mask = iou == iou.max()
    t_i, a_i = torch.nonzero(mask)[0]
    t = targets[t_i]
    a = anchors[a_i]
    print(f"iou: {iou.max()}")
    a.show_together(t)
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass
  return iou

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications


def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def test_iou_3d(bbox3d0, bbox3d1, mode):
  '''
  bbox3d: [N,7]
  '''
  boxlist3d0 = BoxList3D(bbox3d0, size3d = None, mode=mode, examples_idxscope = None, constants={})
  boxlist3d1 = BoxList3D(bbox3d1, size3d = None, mode=mode, examples_idxscope = None, constants={})
  boxlist3d0 = boxlist3d0.convert("yx_zb")
  boxlist3d1 = boxlist3d1.convert("yx_zb")

  ious = boxlist_iou_3d(boxlist3d0, boxlist3d1, 0)
  ious_diag = ious.diag()
  err_mask = torch.abs(ious_diag - 1) > 0.01
  err_inds = torch.nonzero(err_mask).view(-1)
  err_boxlist = boxlist3d[err_inds]
  #print(f"ious:{ious}")
  print(f"ious_diag: {ious_diag}")
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
   #[0,0,   0.0000,   0.001,   2.,   2.,  0],
   #[0,0,   0.0000,   0.01,   2.,   2.,  0],

   #[0,0,   0.0000,   0.001,   2.,   2.,  np.pi/2],
   #[0,0,   0.0000,   0.01,   2.,   2.,  np.pi/2],


   #[0,0,   0.0000,   0.1,   2.,   2.,  np.pi/2],
   [0,0,   0.0000,   1,   2.,   2.,  np.pi/2],
   ],
   dtype=torch.float32
  )

  bbox3d1 = torch.tensor([
   [0.00001,0,   0.0000,   1,   2.,   2.,  np.pi/2],
   ],
   dtype=torch.float32
  )

  bbox3d0 = bbox3d0.to(device)
  bbox3d1 = bbox3d1.to(device)


  test_iou_3d(bbox3d0, bbox3d1, 'yx_zb')

if __name__ == '__main__':
  main_test_iou_3d()


