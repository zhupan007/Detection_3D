# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .bounding_box_3d import BoxList3D

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


def boxlist_iou_3d(anchors, targets):
    cuda_index = targets.bbox3d.device.index
    anchors_2d = anchors.bbox3d[:,[0,1,3,4,6]].cpu().data.numpy()
    targets_2d = targets.bbox3d[:,[0,1,3,4,6]].cpu().data.numpy()
    # aug thickness. When thickness==0, iou is wrong
    targets_2d[:,2] += 0.25
    # criterion=1: use targets_2d as ref
    iou = rotate_iou_gpu_eval(targets_2d, anchors_2d, criterion=-1, device_id=cuda_index)
    iou = torch.from_numpy(iou)
    iou = iou.to(targets.bbox3d.device)
    if DEBUG:
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


# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist_3d(bboxes, per_example=False):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
        per_example: if True, each element in bboxes is an example, combine to a batch
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList3D) for bbox in bboxes)

    if not per_example:
      size3d = bboxes[0].size3d
      for bbox3d in bboxes:
        #is_size_close =  torch.abs(bbox3d.size3d - size3d).max() < 0.01
        #if not is_size_close:
        if not torch.isclose( bbox3d.size3d, size3d ).all():
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass
    else:
      size3d = torch.cat([b.size3d for b in bboxes])

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    batch_size0 = bboxes[0].batch_size()
    for bbox in bboxes:
      assert bbox.batch_size() == batch_size0

    bbox3d_cat = _cat([bbox3d.bbox3d for bbox3d in bboxes], dim=0)
    if not per_example:
      examples_idxscope = torch.tensor([[0, bbox3d_cat.shape[0]]], dtype=torch.int32)
      batch_size = batch_size0
      assert batch_size0 == 1, "check if >1 if need to"
    else:
      assert batch_size0 == 1, "check if >1 if need to"
      batch_size = len(bboxes)
      examples_idxscope = torch.cat([b.examples_idxscope for b in bboxes])
      for b in range(1,batch_size):
        examples_idxscope[b,:] += examples_idxscope[b-1,1]
    cat_boxes = BoxList3D(bbox3d_cat, size3d, mode, examples_idxscope)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes
