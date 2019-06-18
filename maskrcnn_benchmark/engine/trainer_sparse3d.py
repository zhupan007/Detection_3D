# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime, os
import logging
import time

import torch
from torch import autograd
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from data3d.evaluation import evaluate

SHOW_FN = True
CHECK_NAN = True

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    epoch_id,
    eval_in_train,
    eval_out_dir,
    eval_in_train_per_iter,
    iou_thresh_eval,
    min_loss
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info(f"Start training {epoch_id}")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    predictions_all = []
    losses_last = 100
    for iteration, batch in enumerate(data_loader, start_iter):
        fn = [os.path.basename(os.path.dirname(nm)) for nm in batch['fn']]
        if SHOW_FN:
          print(f'\t\t{fn}')

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        batch['x'][1] = batch['x'][1].to(device)
        batch['y'] = [b.to(device) for b in batch['y']]

        loss_dict, predictions_i = model(batch['x'], batch['y'])


        if CHECK_NAN:
          any_nan = sum(torch.isnan(v.data) for v in loss_dict.values())
          if any_nan:
            print(f'\nGot nan loss:\n{fn}\n')
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            continue

        losses = sum(loss for loss in loss_dict.values())

        if eval_in_train>0 and epoch_id % eval_in_train == 0:
          data_id = batch['id']
          for k in range(len(data_id)):
            predictions_i[k].constants['data_id'] = data_id[k]

          predictions_i = [p.to(torch.device('cpu')) for p in predictions_i]
          [p.detach() for p in predictions_i]
          predictions_all += predictions_i

          if eval_in_train_per_iter>0 and epoch_id % eval_in_train_per_iter == 0:
            logger.info(f'\nepoch {epoch_id}, data_id:{data_id}\n')
            eval_res_i = evaluate(dataset=data_loader.dataset, predictions=predictions_i,
                                 iou_thresh_eval=iou_thresh_eval,  output_folder=eval_out_dir, box_only=False)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        with autograd.detect_anomaly():
          optimizer.zero_grad()
          losses.backward()
          optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 1 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        avg_loss = meters.loss.avg
        if iteration % 10 == 0 and avg_loss < min_loss:
            checkpointer.save("model_min_loss", **arguments)
            logger.info(f'\nmin loss: {avg_loss} at {iteration}\n')
            min_loss = avg_loss

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)\n".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    if eval_in_train>0 and epoch_id % eval_in_train == 0:
      logger.info(f'\nepoch {epoch_id}\n')
      eval_res = evaluate(dataset=data_loader.dataset, predictions=predictions_all,
                          iou_thresh_eval=iou_thresh_eval, output_folder=eval_out_dir, box_only=False)
      pass
    return min_loss


