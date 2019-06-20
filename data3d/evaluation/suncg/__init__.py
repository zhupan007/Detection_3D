import logging

from .suncg_eval import do_suncg_evaluation


def suncg_evaluation(dataset, predictions, iou_thresh_eval, output_folder, box_only, epoch=None, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("evaluation with box_only / RPN_Only")
    logger.info("performing suncg evaluation")
    return do_suncg_evaluation(
        dataset=dataset,
        predictions=predictions,
        iou_thresh_eval=iou_thresh_eval,
        output_folder=output_folder,
        logger=logger,
        epoch=epoch
    )
