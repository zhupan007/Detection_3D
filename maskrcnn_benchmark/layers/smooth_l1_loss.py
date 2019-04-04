# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch, math
from utils3d.geometric_torch import limit_period


def get_yaw_loss(input, target, anchor):
    """
    Note: target[:,-1] is the offset truth, not the yaw truth
    """
    dif_loss = torch.abs(input[:,-1]-target[:,-1])
    sin_loss = torch.sin(dif_loss)

    pred_yaw = input[:,-1] + anchor.bbox3d[:,-1]
    yaw_scope_mask = torch.abs(pred_yaw) <= math.pi/2
    yaw_loss = torch.where(yaw_scope_mask, sin_loss, dif_loss)
    return yaw_loss

def smooth_l1_loss(input, target, anchor, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    dif = torch.abs(input - target)

    dif[:,-1] = get_yaw_loss(input, target, anchor)

    cond = dif < beta
    loss = torch.where(cond, 0.5 * dif ** 2 / beta, dif - 0.5 * beta)

    if size_average:
        return loss.mean()
    return loss.sum()
