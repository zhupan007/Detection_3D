# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_suncg.py)
from __future__ import division

import os
from collections import defaultdict
import numpy as np
from maskrcnn_benchmark.structures.bounding_box_3d import BoxList3D
from maskrcnn_benchmark.structures.boxlist_ops_3d import boxlist_iou_3d

DEBUG = True
SHOW_IOU = DEBUG and False
SHOW_GOOD_PRED = DEBUG and True

def do_suncg_evaluation(dataset, predictions, output_folder, logger):
    # TODO need to make the use_07_metric format available
    # for the user to choose
    pred_boxlists = predictions
    gt_boxlists = []
    image_ids = []
    for i, prediction in enumerate(predictions):
        image_id = prediction.constants['data_id']
        image_ids.append(image_id)
        img_info = dataset.get_img_info(image_id)
        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_boxlists.append(gt_boxlist)

    gt_num_totally = sum([len(g) for g in gt_boxlists])
    if gt_num_totally == 0:
        print(f'gt_num_totally=0, abort evalution')
        return

    result = eval_detection_suncg(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=0.5,
        use_07_metric=True,
    )
    result_str = "mAP: {:.4f}\n".format(result["map"])
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        result_str += "{:<16}: {:.4f}\n".format(
            dataset.map_class_id_to_class_name(i), ap
        )
    logger.info(result_str)
    if output_folder:
        with open(os.path.join(output_folder, "result.txt"), "w") as fid:
            fid.write(result_str)

    if SHOW_GOOD_PRED:
        print('SHOW_GOOD_PRED')
        ap = result['ap'][1:]
        if np.isnan(ap).any():
            return result
        for i in range(len(pred_boxlists)):
            pcl_i = dataset[image_ids[i]]['x'][1][:,0:6]
            tops = pred_boxlists[i].remove_low('scores', 0.5)
            tops.show_together(gt_boxlists[i], points=pcl_i, offset_x=10)

            #pred_boxlists[i].show_by_objectness(0.5, gt_boxlists[i])
    return result


def eval_detection_suncg(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
    """Evaluate on suncg dataset.
    Args:
        pred_boxlists(list[BoxList3D]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList3D]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
    prec, rec = calc_detection_suncg_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh
    )
    ap = calc_detection_suncg_ap(prec, rec, use_07_metric=use_07_metric)
    return {"ap": ap, "map": np.nanmean(ap)}


def calc_detection_suncg_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        pred_bbox = pred_boxlist.bbox3d.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox3d.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        gt_difficult = gt_boxlist.get_field("difficult").numpy()

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            pred_bbox_l = pred_bbox_l.copy()
            gt_bbox_l = gt_bbox_l.copy()
            iou = boxlist_iou_3d(
                BoxList3D(pred_bbox_l, pred_boxlist.size3d, pred_boxlist.mode, None, pred_boxlist.constants),
                BoxList3D(gt_bbox_l, gt_boxlist.size3d, gt_boxlist.mode, None, gt_boxlist.constants),
                aug_thickness = { 'target':0, 'anchor':0},
                criterion = -1,
            ).numpy()

            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1

            if SHOW_IOU:
              show_ious(pred_boxlist, gt_boxlist, iou, gt_index)
              import pdb; pdb.set_trace()  # XXX BREAKPOINT
              pass
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def show_ious(pred_boxlist, gt_boxlist, iou, gt_index):
  print(f"iou: {iou.max(1)}")
  print(f"gt_index: {gt_index}")
  pred_boxlist.show_together(gt_boxlist)

def calc_detection_suncg_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
