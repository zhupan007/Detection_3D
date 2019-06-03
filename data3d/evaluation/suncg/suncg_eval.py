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
SHOW_PRED = DEBUG and False
DRAW_RECALL_PRECISION = DEBUG and False
SHOW_FILE_NAMES = DEBUG and False

DRAW_REGRESSION_IOU = DEBUG and False

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

def do_suncg_evaluation(dataset, predictions, iou_thresh_eval, output_folder, logger):
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

    if SHOW_FILE_NAMES:
        print(f'\n{fns}')
    gt_nums = [len(g) for g in gt_boxlists]
    pred_nums = [len(p) for p in pred_boxlists]
    gt_num_totally = sum(gt_nums)
    if gt_num_totally == 0:
        print(f'gt_num_totally=0, abort evalution')
        return

    result = eval_detection_suncg(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=iou_thresh_eval,
        use_07_metric=True,
    )

    obj_gt_nums = get_obj_nums(gt_boxlists)
    regression_res, missed_gt_ids, multi_preds_gt_ids, good_pred_ids, small_iou_preds = \
        parse_pred_for_each_gt(result['pred_for_each_gt'], obj_gt_nums, logger)

    recall_precision_10steps = result["recall_precision_10steps"]
    result_str = performance_str(result, dataset, regression_res)
    logger.info(result_str)

    if output_folder:
        with open(os.path.join(output_folder, "result.txt"), "w") as fid:
            fid.write(result_str)

    if SHOW_PRED:
        SHOW_SMALL_IOU = False
        print('SHOW_PRED')
        ap = result['ap'][1:]
        if np.isnan(ap).all():
            return result
        gt_boxlists_ = modify_gt_labels(gt_boxlists, missed_gt_ids, multi_preds_gt_ids, gt_nums, obj_gt_nums)
        pred_boxlists_ = modify_pred_labels(pred_boxlists, good_pred_ids, pred_nums)
        for i in range(len(pred_boxlists)):
            pcl_i = dataset[image_ids[i]]['x'][1][:,0:6]
            #preds = pred_boxlists[i].remove_low('scores', 0.5)
            preds = pred_boxlists_[i] # already post processed in:
            # ~/Research/Detection_3D/maskrcnn_benchmark/modeling/roi_heads/box_head_3d/inference.py
            # cfg.MODEL.ROI_HEADS.SCORE_THRESH
            xyz_max = pcl_i[:,0:3].max(0).values.data.numpy()
            xyz_min = pcl_i[:,0:3].min(0).values.data.numpy()
            xyz_size = xyz_max - xyz_min
            print(f'xyz_size:{xyz_size}')

            preds.show_together(gt_boxlists_[i], points=pcl_i, offset_x=xyz_size[0]+0.3, twolabels=False)
            #preds.show_together(gt_boxlists_[i], points=pcl_i, offset_x=0, twolabels=True)


            if SHOW_SMALL_IOU:
                small_iou_pred_ids = [p['pred_idx'] for p in  small_iou_preds[i]]
                small_ious = [p['iou'] for p in  small_iou_preds[i]]
                print(f'small iou preds: {small_iou_preds[i]}')
                if len(small_iou_pred_ids)>0:
                    preds.show_highlight(small_iou_pred_ids, points=pcl_i)

            #pred_boxlists[i].show_by_objectness(0.5, gt_boxlists[i])
    if DRAW_RECALL_PRECISION:
        draw_recall_precision_10steps(result['recall_precision_10steps'], '10steps')
        draw_recall_precision_10steps(result['rec_prec_org'], 'original')
    return result


def performance_str(result, dataset, regression_res):
    result_str = "\nmAP: {:.4f}\n".format(result["map"])
    ap = result['ap']
    recall_precision_10steps = result["recall_precision_10steps"]

    class_num = len(ap)
    class_names = []
    rec7_precision = []
    rec9_precision = []
    ious_mean = []
    ious_std = []
    ious_min = []
    scores_mean = []
    scores_std = []
    scores_min = []
    for i in range(class_num):
        clsn = dataset.map_class_id_to_class_name(i)
        if i==0:
            clsn = 'mean'
        class_names.append(clsn )
        rec7_precision.append( recall_precision_10steps[i][7][1] )
        rec9_precision.append( recall_precision_10steps[i][9][1] )
        if i==0:
            ious_mean.append( np.nan )
            ious_std.append( np.nan)
            ious_min.append( np.nan )
            scores_mean.append( np.nan )
            scores_std.append( np.nan)
            scores_min.append( np.nan )
        else:
            ious_mean.append( regression_res[clsn]['ave_std_iou'][0] )
            ious_std.append( regression_res[clsn]['ave_std_iou'][1] )
            ious_min.append( regression_res[clsn]['min_max_iou'][0] )
            scores_mean.append( regression_res[clsn]['ave_std_score'][0] )
            scores_std.append( regression_res[clsn] ['ave_std_score'][1] )
            scores_min.append( regression_res[clsn] ['min_max_score'][0] )


    ious_mean[0] = np.mean(ious_mean[1:])
    ious_std[0] = np.mean(ious_std[1:])
    ious_min[0] = np.min(ious_std[1:])

    scores_mean[0] = np.mean(scores_mean[1:])
    scores_std[0] =  np.mean(scores_std[1:])
    scores_min[0] =  np.min( scores_std[1:])

    result_str += f'{"class:":12}' + ' '.join([f'{c:<10}' for c in  class_names]) + '\n'
    result_str += f'{"ap:":12}' + ' '.join([f'{p:<10.4f}' for p in ap]) + '\n'
    result_str += f'{"r7p:":12}' + ' '.join([f'{p:<10.4f}' for p in rec7_precision]) + '\n'
    result_str += f'{"r9p:":12}' + ' '.join([f'{p:<10.4f}' for p in rec9_precision]) + '\n'
    result_str += f'{"iou mean:":12}' + ' '.join([f'{p:<10.4f}' for p in ious_mean]) + '\n'
    result_str += f'{"iou std:":12}' + ' '.join([f'{p:<10.4f}' for p in ious_std]) + '\n'
    result_str += f'{"iou min:":12}' + ' '.join([f'{p:<10.4f}' for p in ious_min]) + '\n'
    result_str += f'{"score mean:":12}' + ' '.join([f'{p:<10.4f}' for p in scores_mean]) + '\n'
    result_str += f'{"score std:":12}' + ' '.join([f'{p:<10.4f}' for p in  scores_std]) + '\n'
    result_str += f'{"score min:":12}' + ' '.join([f'{p:<10.4f}' for p in  scores_min]) + '\n'
    result_str += '\n'
    #print(result_str)
    return result_str

def performance_str0(result):
    result_str = "\nmAP: {:.4f}\n".format(result["map"])
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            class_name = 'ave'
        else:
            class_name = dataset.map_class_id_to_class_name(i)
        prec_rec7 = recall_precision_10steps[i][7][1]
        prec_rec9 = recall_precision_10steps[i][9][1]
        result_str += "{:<16}: {:.4f}, \t(rec,prec): (0.7,{:.4f}), (0.9,{:.4f})\n".format(
            class_name, ap, prec_rec7, prec_rec9
        )

def modify_pred_labels(pred_boxlists, good_pred_ids, pred_nums):
    batch_size = len(pred_nums)
    pred_labels = []
    new_pred_boxlists = []
    for bi in range(batch_size):
        # incorrect pred: 0,  others: class label + 1
        #labels_i = pred_boxlists[bi].get_field('labels') + 1
        labels_i = np.zeros([pred_nums[bi]], dtype=np.int32)
        for obj in good_pred_ids:
            l = SUNCG_META.class_2_label[obj]
            labels_i[good_pred_ids[obj][bi]] = l + 1
        pred_labels.append(labels_i)

        pred = pred_boxlists[bi].copy()
        pred.add_field('labels', labels_i)
        new_pred_boxlists.append(pred)
    return new_pred_boxlists

def modify_gt_labels(gt_boxlists, missed_gt_ids, multi_preds_gt_ids, gt_nums, obj_gt_nums):
    batch_size = len(gt_nums)
    gt_labels = []
    new_gt_boxlists = []
    for bi in range(batch_size):
        # missed:0, multi: 1, matched: class label + 1
        #labels_i = np.zeros([gt_nums[bi]], dtype=np.int32)
        labels_i = gt_boxlists[bi].get_field('labels') + 1
        #labels_i = np.random.choice(gt_nums[bi], gt_nums[bi], replace=False)+2
        start = 0 # the gt_ids is only of one class (TAG: GT_MASK)
        for obj in missed_gt_ids:
            #gt_label_i = SUNCG_META.class_2_label[obj]
            labels_i[ missed_gt_ids[obj][bi] + start ] = 0
            labels_i[ multi_preds_gt_ids[obj][bi] + start ] = 1
            start += obj_gt_nums[obj][bi]
        gt_labels.append(labels_i)

        boxlist = gt_boxlists[bi].copy()
        boxlist.add_field('labels', labels_i)
        new_gt_boxlists.append(boxlist)

    return new_gt_boxlists

def parse_pred_for_each_gt(pred_for_each_gt, obj_gt_nums, logger):
    missed_gt_ids = defaultdict(list)
    multi_preds_gt_ids = defaultdict(list)
    ious = defaultdict(list)
    scores = defaultdict(list)
    good_pred_ids = defaultdict(list)
    ious_flat = {}
    scores_flat = {}
    regression_res = {}
    batch_size = len(pred_for_each_gt['wall'])

    small_iou_preds = []
    for bi in range(batch_size):
        small_iou_preds.append([])
    small_iou_threshold = 0.5

    for obj in pred_for_each_gt.keys():
        for bi in range(batch_size):
            if len(pred_for_each_gt[obj]) == 0:
                continue
            try:
                peg = pred_for_each_gt[obj][bi]
            except:
                import pdb; pdb.set_trace()  # XXX BREAKPOINT
                pass

            #-------------------------------
            # get scores and iou
            ious_bi = []
            scores_bi = []
            good_pred_ids_bi = []
            for pi in peg:
                # if a gt matches multiple preds, the max score one is positive
                # here the first one actually has the max score
                peg_max_score = peg[pi][0]
                scores_bi.append( peg_max_score['score'] )
                ious_bi.append( peg_max_score['iou'] )
                good_pred_ids_bi.append( peg_max_score['pred_idx'] )
            ious_bi = np.array(ious_bi)
            scores_bi = np.array(scores_bi)
            ious[obj].append(ious_bi)
            scores[obj].append(scores_bi)
            good_pred_ids[obj].append(np.array(good_pred_ids_bi))

            #-------------------------------
            # small iou
            small_iou_preds_bi = []
            for pi in peg:
                for peg_ in peg[pi]:
                    if peg_['iou'] < small_iou_threshold:
                        bad_p = {}
                        bad_p['pred_idx'] = peg_['pred_idx']
                        bad_p['iou'] = peg_['iou']
                        bad_p['score'] = peg_['score']
                        bad_p['gt_idx'] = pi
                        bad_p['class'] = obj
                        small_iou_preds_bi.append(bad_p)
            small_iou_preds[bi] += small_iou_preds_bi

            #-------------------------------
            # get missed_gt_ids  and multi_preds_gt_ids
            gt_ids = np.array([k for k in peg.keys()])
            #gt_ids is only the index inside of one single class gts: (TAG: GT_MASK)
            pred_num_each_gt = np.histogram(gt_ids, bins=range(obj_gt_nums[obj][bi]+1))[0]
            pred_num_hist = np.histogram(pred_num_each_gt, bins=[0,1,2,3,4])[0]
            #print(f'{pred_num_hist[0]} gt boxes are missed \n{pred_num_hist[1]} t Boxes got one prediction')
            #print(f'{pred_num_hist[2]} gt boxes got 2 predictions')
            missed_gt_ids_bi = np.where(pred_num_each_gt==0)[0]
            multi_preds_gt_ids_bi = np.where(pred_num_each_gt>1)[0]

            missed_gt_ids[obj].append(missed_gt_ids_bi)
            multi_preds_gt_ids[obj].append(multi_preds_gt_ids_bi)

            pass

        ious_flat[obj] = np.concatenate(ious[obj], 0)
        scores_flat[obj] = np.concatenate(scores[obj], 0)

        ave_iou = np.mean(ious_flat[obj])
        std_iou = np.std(ious_flat[obj])
        max_iou = np.max(ious_flat[obj])
        min_iou = np.min(ious_flat[obj])
        ave_score = np.mean(scores_flat[obj])
        std_score = np.std(scores_flat[obj])
        max_score =  np.max(scores_flat[obj])
        min_score =  np.min(scores_flat[obj])
        regression_res[obj] = {}
        regression_res[obj]['min_max_iou'] = [min_iou, max_iou]
        regression_res[obj]['ave_std_iou'] = [ave_iou, std_iou]
        regression_res[obj]['min_max_score'] = [min_score, max_score]
        regression_res[obj]['ave_std_score'] = [ave_score, std_score]

        missed_gt_num = sum([len(gti) for gti in missed_gt_ids[obj] ])
        multi_gt_num = sum([len(gti) for gti in multi_preds_gt_ids[obj] ])
        gt_num_sum = sum(obj_gt_nums[obj])
        regression_res[obj]['missed_multi_sum_gtnum'] = [missed_gt_num, multi_gt_num, gt_num_sum]
        missed_rate = 1.0*missed_gt_num / gt_num_sum
        multi_rate = 1.0*multi_gt_num / gt_num_sum
        matched_rate = 1 - missed_rate - multi_rate
        regression_res[obj]['missed_multi_rate'] = [matched_rate, missed_rate, multi_rate]
        pass

    reg_str = regression_res_str(regression_res)
    logger.info(reg_str)

    if DRAW_REGRESSION_IOU:
        for obj in ious_flat:
            fig = plt.figure(1)
            plt.plot(ious_flat[obj],'o')
            plt.xlabel(f'{obj} index')
            plt.ylabel('iou')
            title = f'iou of {obj} prediction'
            #plt.title(title)
            plt.show()
            fname = f'iou_{obj}.png'
            fig.savefig(fname)
            print(fname)
            pass

    return regression_res, missed_gt_ids, multi_preds_gt_ids, good_pred_ids, small_iou_preds

def regression_res_str(regression_res):
    reg_str = '\n\nregression result\n'
    for key in regression_res:
        value = regression_res[key]
        reg_str += f'{key}:\n{value}\n'
    return reg_str

def draw_recall_precision_10steps(recall_precision_10steps, flag):
    num_classes = len(recall_precision_10steps)
    for i in range(num_classes):
        obj = SUNCG_META.label_2_class[i]
        if i==0:
            if flag == '10steps':
                continue
            obj = 'ave'
        rp = recall_precision_10steps[i]
        print(f'\n{obj}\n{rp}')
        fig = plt.figure(i)
        plt.plot(rp[:,0], rp[:,1])
        plt.ylabel('precision')
        plt.xlabel('recall')
        title = flag+' '+obj+' recall-precision'
        plt.title(title)
        fig.savefig(title+'.png')
        print('save: '+title+'.png')
    plt.show()

def get_obejct_numbers(boxlist):
    labels = boxlist.get_field('labels').data.numpy()
    lset = list(set(labels))
    obj_nums = {}
    for l in lset:
        obj_nums[SUNCG_META.label_2_class[l]] = sum(labels==l)
    return obj_nums

def eval_detection_suncg(pred_boxlists, gt_boxlists, iou_thresh, use_07_metric=False):
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
    rec_prec_org = [np.concatenate([np.array(r).reshape([-1,1]), np.array(p).reshape([-1,1])],1) for r,p in zip(rec, prec)]
    ap, recall_precision_10steps = calc_detection_suncg_ap(prec, rec, use_07_metric=use_07_metric)
    return {"ap": ap, "map": np.nanmean(ap), "rec_prec_org":rec_prec_org, "recall_precision_10steps":recall_precision_10steps, "pred_for_each_gt":pred_for_each_gt}


def calc_detection_suncg_prec_rec(gt_boxlists, pred_boxlists, iou_thresh):
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
    batch_size = len(gt_boxlists)

    bi = -1
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        bi += 1
        pred_bbox = pred_boxlist.bbox3d.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox3d.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        gt_difficult = gt_boxlist.get_field("difficult").numpy()

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            obj_name = SUNCG_META.label_2_class[l]
            pred_mask_l = pred_label == l
            pred_ids_l = np.where(pred_mask_l)[0]
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            # Extract gt only of current class, thus gt_index is the index
            # inside of one signle class gts, not of all gts
            # TAG: GT_MASK
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
                BoxList3D(gt_bbox_l, gt_boxlist.size3d, gt_boxlist.mode, None, gt_boxlist.constants),
                BoxList3D(pred_bbox_l, pred_boxlist.size3d, pred_boxlist.mode, None, pred_boxlist.constants),
                aug_thickness = { 'target':0, 'anchor':0},
                criterion = -1,
                flag='eval'
            ).numpy()

            gt_index = iou.argmax(axis=0) # the gt index for each predicion
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=0) < iou_thresh] = -1
            iou_pred = iou.max(0)
            pred_for_each_gt_l = defaultdict(list)
            neg_count =  0
            for pi in range(gt_index.shape[0]):
                pis = {'pred_idx': pred_ids_l[pi], 'iou':iou[gt_index[pi], pi], 'score':score[l][pi]}
                gt_idx = gt_index[pi]
                if gt_idx<0:
                    neg_count += 1
                    gt_idx -= (gt_idx==-1) * neg_count
                pred_for_each_gt_l[gt_idx].append(pis)

            if obj_name not in pred_for_each_gt:
                for iii in range(batch_size):
                    pred_for_each_gt[obj_name].append(defaultdict(list))
            pred_for_each_gt[obj_name][bi] = pred_for_each_gt_l

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
    recall_precision_10steps = np.empty([n_fg_class, 11, 2])
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            recall_precision_10steps[l] = np.nan
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
            recall_precision_10steps[l] = np.array(rp)
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

    # set the first as average
    recall_precision_10steps[0] = recall_precision_10steps[1:].mean(0)
    ap[0] = ap[1:].mean()
    return ap, recall_precision_10steps
