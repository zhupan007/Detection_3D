# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .bounding_box_3d import BoxList3D, cat_boxlist_3d

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
  cuda_index = targets.bbox3d.device.index
  anchors_2d = anchors.bbox3d[:,[0,1,3,4,6]].cpu().data.numpy()
  targets_2d = targets.bbox3d[:,[0,1,3,4,6]].cpu().data.numpy()

  #print(f"targets yaw : {targets_2d[:,-1].min()} , {targets_2d[:,-1].max()}")
  #print(f"anchors yaw : {anchors_2d[:,-1].min()} , {anchors_2d[:,-1].max()}")

  # aug thickness. When thickness==0, iou is wrong
  targets_2d[:,2] += aug_wall_target_thickness # 0.25
  # criterion=1: use targets_2d as ref
  iou = rotate_iou_gpu_eval(targets_2d, anchors_2d, criterion=2, device_id=cuda_index)
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




