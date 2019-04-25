import logging

from .suncg_eval import do_suncg_evaluation


def suncg_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("evaluation with box_only / RPN_Only")
    logger.info("performing suncg evaluation")
    return do_suncg_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )
