# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_suncg.py)
from __future__ import division

import os
from collections import defaultdict
import numpy as np
from maskrcnn_benchmark.structures.bounding_box_3d import BoxList3D
from maskrcnn_benchmark.structures.boxlist_ops_3d import boxlist_iou_3d
from data3d.suncg_utils.suncg_meta import SUNCG_META
import matplotlib.pyplot as plt

DEBUG = True
SHOW_RES_IOU = DEBUG and False
SHOW_GOOD_PRED = DEBUG and False
DRAW_RECALL_PRECISION = DEBUG and False
DRAW_REGRESSION = DEBUG and True


def get_obj_nums(gt_boxlists):
    batch_size = len(gt_boxlists)
    obj_gt_nums = defaultdict(list)
    for bi in range(batch_size):
        labels = gt_boxlists[bi].get_field('labels').cpu().data.numpy()
        bins = set(labels)
        for label in set(labels):
            obj = SUNCG_META.label_2_class[int(label)]
            obj_gt_nums[obj].append( sum(labels==label) )
    return obj_gt_nums

def do_suncg_evaluation(dataset, predictions, output_folder, logger):
    # TODO need to make the use_07_metric format available
    # for the user to choose
    pred_boxlists = predictions
    gt_boxlists = []
    image_ids = []
    fns = []
    for i, prediction in enumerate(predictions):
        image_id = prediction.constants['data_id']
        fns.append( dataset.files[image_id] )
        image_ids.append(image_id)
        img_info = dataset.get_img_info(image_id)
        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_boxlists.append(gt_boxlist)

    print(f'\n{fns}')
    gt_nums = [len(g) for g in gt_boxlists]
    gt_num_totally = sum(gt_nums)
    if gt_num_totally == 0:
        print(f'gt_num_totally=0, abort evalution')
        return

    result = eval_detection_suncg(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=0.5,
        use_07_metric=True,
    )

    obj_gt_nums = get_obj_nums(gt_boxlists)
    parse_pred_for_each_gt(result['pred_for_each_gt'], obj_gt_nums, logger)

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
        if np.isnan(ap).all():
            return result
        for i in range(len(pred_boxlists)):
            pcl_i = dataset[image_ids[i]]['x'][1][:,0:6]
            tops = pred_boxlists[i].remove_low('scores', 0.5)
            gt_nums = get_obejct_numbers(gt_boxlists[i])
            pred_nums = get_obejct_numbers(tops)
            print(f'pred: {pred_nums}\ngt: {gt_nums}')
            xyz_max = pcl_i[:,0:3].max(0).values.data.numpy()
            xyz_min = pcl_i[:,0:3].min(0).values.data.numpy()
            xyz_size = xyz_max - xyz_min
            print(f'xyz_size:{xyz_size}')
            tops.show_together(gt_boxlists[i], points=pcl_i, offset_x=xyz_size[0]+0.2)

            #pred_boxlists[i].show_by_objectness(0.5, gt_boxlists[i])
    if DRAW_RECALL_PRECISION:
        draw_recall_precision(result['recall_precision'])
    return result

def parse_pred_for_each_gt(pred_for_each_gt, obj_gt_nums, logger):
    misses_gt_ids = defaultdict(list)
    multi_preds_gt_ids = defaultdict(list)
    ious = defaultdict(list)
    scores = defaultdict(list)
    ious_flat = {}
    scores_flat = {}
    regression_res = {}

    for obj in pred_for_each_gt.keys():
        batch_size = len(pred_for_each_gt[obj])
        for bi in range(batch_size):
            peg = pred_for_each_gt[obj][bi]

            # get scores and iou
            ious_bi = []
            scores_bi = []
            for pi in peg:
                # if a gt matches multiple preds, the max score one is positive
                # here the first one actually has the max score
                peg_max_score = peg[pi][0]
                scores_bi.append( peg_max_score['score'] )
                ious_bi.append( peg_max_score['iou'] )
            ious_bi = np.array(ious_bi)
            scores_bi = np.array(scores_bi)
            ious[obj].append(ious_bi)
            scores[obj].append(scores_bi)

            # get missed_gt_ids  and multi_preds_gt_ids
            gt_ids = np.array([k for k in peg.keys()])
            pred_num_each_gt = np.histogram(gt_ids, bins=range(obj_gt_nums[obj][bi]+1))[0]
            pred_num_hist = np.histogram(pred_num_each_gt, bins=[0,1,2,3,4])[0]
            print(f'{pred_num_hist[0]} gt boxes are missed \n{pred_num_hist[1]} t Boxes got one prediction')
            print(f'{pred_num_hist[2]} gt boxes got 2 predictions')
            missed_gt_ids_bi = np.where(pred_num_each_gt==0)[0]
            multi_preds_gt_ids_bi = np.where(pred_num_each_gt>1)[0]

            misses_gt_ids[obj].append(missed_gt_ids_bi)
            multi_preds_gt_ids[obj].append(multi_preds_gt_ids_bi)

            pass

        ious_flat[obj] = np.concatenate(ious[obj], 0)
        scores_flat[obj] = np.concatenate(scores[obj], 0)
        ave_iou = np.mean(ious_flat[obj])
        max_iou = np.max(ious_flat[obj])
        min_iou = np.min(ious_flat[obj])
        ave_score = np.mean(scores_flat[obj])
        max_score =  np.max(scores_flat[obj])
        min_score =  np.min(scores_flat[obj])
        regression_res[obj] = {}
        regression_res[obj]['min_ave_max_iou'] = [min_iou, ave_iou, max_iou]
        regression_res[obj]['min_ave_max_score'] = [min_score, ave_score, max_score]

        missed_gt_num = sum([len(gti) for gti in misses_gt_ids[obj] ])
        multi_gt_num = sum([len(gti) for gti in multi_preds_gt_ids[obj] ])
        gt_num_sum = sum(obj_gt_nums[obj])
        regression_res[obj]['missed_multi_sum_gtnum'] = [missed_gt_num, multi_gt_num, gt_num_sum]
        missed_rate = 1.0*missed_gt_num / gt_num_sum
        multi_rate = 1.0*multi_gt_num / gt_num_sum
        matched_rate = 1 - missed_rate - multi_rate
        regression_res[obj]['missed_multi_rate'] = [matched_rate, missed_rate, multi_rate]

    logger.info(f'regression_res: {regression_res}')

    if DRAW_REGRESSION:
        for obj in ious_flat:
            plt.figure(1)
            plt.plot(ious_flat[obj])
            plt.title(f'iou of {obj} prediction')
        plt.show()
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass

def draw_recall_precision(recall_precision):
    num_classes = len(recall_precision)
    for i in range(1,num_classes):
        obj = SUNCG_META.label_2_class[i]
        rp = recall_precision[i]
        print(f'\n{obj}\n{rp}')
        plt.figure(i)
        plt.plot(rp[:,0], rp[:,1])
        plt.ylabel('precision')
        plt.xlabel('recall')
        plt.title(obj)
    plt.show()

def get_obejct_numbers(boxlist):
    labels = boxlist.get_field('labels').data.numpy()
    lset = list(set(labels))
    obj_nums = {}
    for l in lset:
        obj_nums[SUNCG_META.label_2_class[l]] = sum(labels==l)
    return obj_nums

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
    prec, rec, pred_for_each_gt = calc_detection_suncg_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh
    )
    ap, recall_precision = calc_detection_suncg_ap(prec, rec, use_07_metric=use_07_metric)
    return {"ap": ap, "map": np.nanmean(ap), "recall_precision":recall_precision, "pred_for_each_gt":pred_for_each_gt}


def calc_detection_suncg_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    # The pred having maximum iou with a gt is matched with the gt.
    # If multiple preds share same maximum iou gt, the one with highest score is
    # selected. NOTICE HERE, not the one with highest iou! Because, in test,
    # only score is available.
    match = defaultdict(list)  # 1:true, 0:false, -1:ignore

    pred_for_each_gt = defaultdict(list)
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        pred_bbox = pred_boxlist.bbox3d.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox3d.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        gt_difficult = gt_boxlist.get_field("difficult").numpy()

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            obj_name = SUNCG_META.label_2_class[l]
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

            gt_index = iou.argmax(axis=1) # the gt index for each predicion
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            iou_pred = iou.max(1)
            pred_for_each_gt_l = defaultdict(list)
            for pi in range(gt_index.shape[0]):
                pis = {'pred_idx': pi, 'iou':iou[pi, gt_index[pi]], 'score':score[l][pi]}
                pred_for_each_gt_l[gt_index[pi]].append(pis)
            pred_for_each_gt[obj_name].append(pred_for_each_gt_l)

            if SHOW_RES_IOU:
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
                            # gt_index is already sorted by scores,
                            # thus the first pred match a gt box is set 1
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)
            pass


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

    return prec, rec, pred_for_each_gt


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
    recall_precision = np.empty([n_fg_class, 11, 2])
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            recall_precision[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            rp = []
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
                rp.append([t, p])
            recall_precision[l] = np.array(rp)
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

    return ap, recall_precision
