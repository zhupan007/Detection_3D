# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box_3d import BoxList3D
from maskrcnn_benchmark.structures.boxlist_ops_3d import boxlist_nms_3d
from maskrcnn_benchmark.structures.boxlist_ops_3d import cat_boxlist_3d
from maskrcnn_benchmark.modeling.box_coder_3d import BoxCoder3D
from utils3d.bbox3d_ops_torch import Box3D_Torch

DEBUG = False
SHOW_FILTER = DEBUG and 0

MERGE_BY_CORNER = False

class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self, score_thresh=0.05, nms=0.5, nms_aug_thickness=None, detections_per_img=100, box_coder=None, class_specific=True
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder3D)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder3D(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.nms_aug_thickness = nms_aug_thickness
        self.class_specific = class_specific

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList3D]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList3D]): one BoxList3D for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x # [100*batch_size,num_class] [100*batch_size, num_classes*7]
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        size3ds = [box.size3d for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox3d for a in boxes], dim=0)

        proposals = self.box_coder.decode(
          box_regression, concat_boxes
          )

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, size3d in zip(
            class_prob, proposals, size3ds
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, size3d)
            #boxlist = boxlist.clip_to_pcl(remove_empty=False)
            #if SHOW_FILTER:
            #  show_before_filter(boxlist, 'before filter')
            boxlist = self.filter_results(boxlist, num_classes)
            if MERGE_BY_CORNER:
              boxlist = self.merge_by_corners(boxlist)
            if SHOW_FILTER:
              show_before_filter(boxlist, 'after filter')
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, size3d):
        """
        Returns BoxList3D from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        if not self.class_specific:
          class_num = scores.shape[1]
          boxes = boxes.unsqueeze(1).repeat(1,class_num,1)
        boxes = boxes.reshape(-1, 7)
        scores = scores.reshape(-1)
        boxlist = BoxList3D(boxes, size3d, mode="yx_zb", examples_idxscope=None,
          constants={'prediction': True})
        boxlist.add_field("scores", scores)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox3d.reshape(-1, num_classes * 7)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 7 : (j + 1) * 7]
            boxlist_for_class = BoxList3D(boxes_j, boxlist.size3d, mode="yx_zb",
              examples_idxscope=None, constants={'prediction':True})
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms_3d(
              boxlist_for_class, nms_thresh=self.nms,
              nms_aug_thickness=self.nms_aug_thickness, score_field="scores", flag='roi_post'
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

            # debuging
            if DEBUG and False:
                inds_small_scrore = (1-inds_all[:, j]).nonzero().squeeze(1)
                scores_small_j = scores[inds_small_scrore,j]
                max_score_abandoned = scores_small_j.max()
                print(f'max_score_abandoned: {max_score_abandoned}')

        result = cat_boxlist_3d(result, per_example=False)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result

    def merge_by_corners(self, boxlist, threshold=0.1):
      #show_before_filter(boxlist, 'before merging corners')
      top_2corners0, boxes_2corners0 = boxlist.get_2top_corners_offseted()
      boxes_2corners = boxes_2corners0.clone()
      top_2corners = top_2corners0.clone().view([-1,3])
      n = top_2corners.shape[0]
      dis = top_2corners.view([-1,1,3]) - top_2corners.view([1,-1,3])
      dis = dis.norm(dim=2)
      mask = dis < threshold
      device = top_2corners.device
      mask = mask - torch.eye(n, dtype=torch.uint8, device=device)
      mask_merged = torch.zeros([n], dtype=torch.int32, device=device)
      for i in range(n):
        dif_i = top_2corners[i:i+1] - top_2corners
        dis_i = dif_i.norm(dim=1)
        mask_i = dis_i < threshold
        j = i + 2 * (i%2==0) - 1
        mask_i[j] = 0
        ids_i = torch.nonzero(mask_i).squeeze(1)

        # check if the close ids include one whole object
        ids_j = ids_i + 2*(ids_i%2==0).to(torch.int64) - 1
        any_same_obj = ids_i.view(-1,1) == ids_j.view(1,-1)
        if any_same_obj.sum() > 0:
          continue

        #print(ids_i)
        if ids_i.shape[0] > 1:
          ave_i = top_2corners[ids_i].mean(dim=0).view(1,3)
          top_2corners[ids_i] = ave_i
          mask_merged[ids_i] = 1
          #if DEBUG:
          #  print(f'ids: {ids_i}')
          #  print(f'org: {top_2corners[ids_i]}')
          #  print(f'ave: {ave_i}')
          pass

        if DEBUG and False:
          corners_close = top_2corners0.view([-1,3])[ids_i]
          boxlist.show(points = corners_close)

          cor_tmp = top_2corners.view(-1,2,3)
          tmp = (cor_tmp[:,0] - cor_tmp[:,1]).norm(dim=1)
          if tmp.min()==0:
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass

      ids_merged = torch.nonzero(mask_merged).squeeze(1)
      corners_merged = top_2corners[ids_merged]
      top_2corners = top_2corners.view([-1,2,3])


      # offset the corners to the end by half thickness
      centroids = top_2corners.mean(dim=1, keepdim=True)
      offset = top_2corners - centroids
      offset = offset / offset.norm(dim=2, keepdim=True) * boxes_2corners[:,-1].view(-1,1,1) * 0.5
      top_2corners = top_2corners + offset

      boxes_2corners[:,0:2] = top_2corners[:,0,0:2]
      boxes_2corners[:,2:4] = top_2corners[:,1,0:2]
      boxes_2corners[:,5] = top_2corners[:,:,2].mean(dim=1)
      boxes_2corners[:,4] = boxes_2corners[:,4].mean()


      boxlist.bbox3d = Box3D_Torch.from_2corners_to_yxzb(boxes_2corners)

      #boxlist.show(points=corners_merged)
      #show_before_filter(boxlist, 'after merging corners')
      return boxlist

def show_before_filter(boxlist, msg):
  print(msg)
  boxlist.show_with_corners()
  pass

def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder3D(is_corner_roi=cfg.MODEL.CORNER_ROI,  weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    nms_aug_thickness = cfg.MODEL.ROI_HEADS.NMS_AUG_THICKNESS_Y_Z
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    class_specific = cfg.MODEL.CLASS_SPECIFIC

    postprocessor = PostProcessor(
        score_thresh, nms_thresh,
      nms_aug_thickness=nms_aug_thickness,
      detections_per_img=detections_per_img,
      box_coder=box_coder,
      class_specific=class_specific
    )
    return postprocessor
