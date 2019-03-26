# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from maskrcnn_benchmark.modeling.box_coder_3d import BoxCoder3D
from maskrcnn_benchmark.structures.bounding_box_3d import BoxList3D
from maskrcnn_benchmark.structures.boxlist3d_ops import cat_boxlist_3d
from maskrcnn_benchmark.structures.boxlist3d_ops import boxlist_nms_3d
from maskrcnn_benchmark.structures.boxlist3d_ops import remove_small_boxes3d

from ..utils import cat


class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(
        self,
        batch_size,
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        min_size,
        box_coder=None,
        fpn_post_nms_top_n=None,
    ):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
            fpn_post_nms_top_n (int)
        """
        super(RPNPostProcessor, self).__init__()
        self.batch_size = batch_size
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        if box_coder is None:
            box_coder = BoxCoder()
        self.box_coder = box_coder

        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        device = proposals[0].bbox3d.device

        gt_boxes = [target.copy_with_fields([]) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))

        proposals = [
            cat_boxlist_3d((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def forward_for_single_feature_map(self, anchors, objectness, box_regression):
        """
        Arguments:
            anchors: BoxList -> all examples within same batch are concated together
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 7, H, W
            N==1, W==1, A is anchor num per loc
            H is sum of anchors of all examples in the same batch
        """
        device = objectness.device
        assert objectness.shape[0] == 1 == objectness.shape[3]

        examples_idxscope = anchors.examples_idxscope
        batch_size = anchors.batch_size()
        anchors_per_loc = objectness.shape[1]
        result = []
        for bi in range(batch_size):
          # split examples in the batch
          s,e = examples_idxscope[bi]/ anchors_per_loc
          objectness_i = objectness[:,:,s:e,:]
          box_regression_i = box_regression[:,:,s:e,:]

          N, A, H, W = objectness_i.shape

          # put in the same format as anchors
          objectness_i = objectness_i.permute(0, 2, 3, 1).reshape(N, -1)
          objectness_i = objectness_i.sigmoid()
          box_regression_i = box_regression_i.view(N, -1, 7, H, W).permute(0, 3, 4, 1, 2)
          box_regression_i = box_regression_i.reshape(N, -1, 7)

          # only choose top 2000 proposals for nms
          num_anchors = A * H * W
          pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
          objectness_i, topk_idx = objectness_i.topk(pre_nms_top_n, dim=1, sorted=True)

          batch_idx = torch.arange(N, device=device)[:, None]
          box_regression_i = box_regression_i[batch_idx, topk_idx]

          pcl_size3d = anchors.size3d[bi:bi+1]
          s,e = examples_idxscope[bi]
          concat_anchors_i = anchors.bbox3d[s:e,:]
          concat_anchors_i = concat_anchors_i.reshape(N, -1, 7)[batch_idx, topk_idx]

          # decode box_regression to get proposals
          proposals_i = self.box_coder.decode(
              box_regression_i.view(-1, 7), concat_anchors_i.view(-1, 7)
          )
          #proposals_i = proposals_i.view(N, -1, 7)


          #*********************************************************************
          # apply nms
          examples_idxscope_new = torch.tensor([[0, proposals_i.shape[0]]])
          boxlist = BoxList3D(proposals_i, pcl_size3d, mode="yx_zb",
                              examples_idxscope= examples_idxscope_new)
          boxlist.add_field("objectness", objectness_i.squeeze(0))
          #boxlist = boxlist.clip_to_pcl(remove_empty=False)
          #boxlist = remove_small_boxes3d(boxlist, self.min_size)
          boxlist = boxlist_nms_3d(
              boxlist,
              self.nms_thresh,
              max_proposals=self.post_nms_top_n,
              score_field="objectness",
          )
          result.append(boxlist)
        return result

    def forward(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: BoxList
            objectness: list[tensor]
            box_regression: list[tensor]
            batch_size = anchors.batch_size()
            scale_num = num_levels = len(objectness) == len(box_regression)

            anchors: list[list[BoxList]
            objectness: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        print(anchors.batch_size())
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        sampled_boxes = []
        num_levels = len(objectness)
        for a, o, b in zip(anchors, objectness, box_regression):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist_3d(boxlist) for boxlist in boxlists]

        if num_levels > 1:
            boxlists = self.select_over_all_levels(boxlists)

        # append ground-truth bboxes to proposals
        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)

        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        # different behavior during training and during testing:
        # during training, post_nms_top_n is over *all* the proposals combined, while
        # during testing, it is over the proposals for each image
        # TODO resolve this difference and make it consistent. It should be per image,
        # and not per batch
        if self.training:
            objectness = torch.cat(
                [boxlist.get_field("objectness") for boxlist in boxlists], dim=0
            )
            box_sizes = [len(boxlist) for boxlist in boxlists]
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.uint8)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i]]
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field("objectness")
                post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(
                    objectness, post_nms_top_n, dim=0, sorted=True
                )
                boxlists[i] = boxlists[i][inds_sorted]
        return boxlists



def make_rpn_postprocessor(config, rpn_box_coder, is_train):
    fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN
    if not is_train:
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST

    pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TRAIN
    if not is_train:
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    min_size = config.MODEL.RPN.MIN_SIZE
    batch_size = config.SOLVER.IMS_PER_BATCH
    box_selector = RPNPostProcessor(
        batch_size = batch_size,
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
    )
    return box_selector
