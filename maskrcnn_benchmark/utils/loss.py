# https://github.com/marvis/pytorch-yolo2/blob/master/FocalLoss.py

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]

def focal_loss( x, y, size_average=True):
        '''Focal loss.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2
        num_classes = x.shape[1]

        y = y.to(torch.int64)
        t = one_hot_embedding(y.data.cpu(), 1+num_classes)  # [N,21]
        t = t[:,1:]  # exclude background
        t = Variable(t).cuda()  # [N,20]

        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        weight = w * (1-pt).pow(gamma)
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        loss = F.binary_cross_entropy_with_logits(x, t, weight, size_average=size_average)
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        return loss

def focal_loss_alt(x, y, size_average=True):
        '''Focal loss alternative.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        num_classes = x.shape[1]
        y = y.to(torch.int64)

        t = one_hot_embedding(y.data.cpu(), 1+num_classes)
        t = t[:,1:]
        t = Variable(t).cuda()

        xt = x*(2*t-1)  # xt = x if t > 0 else -x
        pt = (2*xt+1).sigmoid()

        w = alpha*t + (1-alpha)*(1-t)
        loss = -w*pt.log() / 2
        if size_average:
          loss_ = loss.mean()
        else:
          loss_ = loss.sum()
        return loss_

