# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import numpy as np

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference_3d import inference_3d
from maskrcnn_benchmark.engine.trainer_sparse3d import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from data3d.data import make_data_loader

def train(cfg, local_rank, distributed, loop):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(cfg, is_train=True, is_distributed=distributed,
                  start_iter=arguments["iteration"])

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    epochs_between_test = cfg.SOLVER.EPOCHS_BETWEEN_TEST
    for e in range(epochs_between_test):
      do_train(
          model,
          data_loader,
          optimizer,
          scheduler,
          checkpointer,
          device,
          checkpoint_period,
          arguments,
          e + loop * epochs_between_test
      )

    return model


def test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_3d", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = [ make_data_loader(cfg, is_train=False, is_distributed=distributed) ]
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference_3d(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    intact_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    for loop in range(cfg.SOLVER.EPOCHS // cfg.SOLVER.EPOCHS_BETWEEN_TEST):
      model = train(cfg, args.local_rank, args.distributed, loop)

      if not args.skip_test:
          test(cfg, model, args.distributed)


def intact_cfg(cfg):
  fpn_scalse = cfg.MODEL.FPN_SCALES
  strides = cfg.SPARSE3D.STRIDE
  nPlanesFront = cfg.SPARSE3D.nPlanesFront
  scale_num = len(nPlanesFront)
  assert scale_num == len(strides) + 1
  ANCHOR_STRIDE = [np.array([1,1,1])]
  for s in range(scale_num-1):
    anchor_stride = ANCHOR_STRIDE[-1] * np.array(strides[s])
    ANCHOR_STRIDE.append(anchor_stride)
  # fpn scales set from 0 to 1..., but used from large to small
  cfg.MODEL.RPN.ANCHOR_STRIDE = list(reversed([ANCHOR_STRIDE[i] for i in fpn_scalse]))
  cfg.MODEL.RPN.ANCHOR_SIZES_3D = list(reversed( cfg.MODEL.RPN.ANCHOR_SIZES_3D ))
  tmp = cfg.MODEL.RPN.ANCHOR_SIZES_3D
  if len(tmp)>1:
    assert tmp[0][0] > tmp[1][0], "ANCHOR_SIZES_3D should set from small to large"

if __name__ == "__main__":
    main()
