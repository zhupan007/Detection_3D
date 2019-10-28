# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
import torch
from utils3d.bbox3d_ops_torch import Box3D_Torch

class FastRCNNPredictor(nn.Module):
    def __init__(self, config, pretrained=None):
        super(FastRCNNPredictor, self).__init__()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = config.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = res2_out_channels * stage2_relative_factor

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.bbox_pred = nn.Linear(num_inputs, num_classes * 7)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


class FPNPredictor(nn.Module):
    def __init__(self, cfg):
        super(FPNPredictor, self).__init__()
        classes_names = cfg.INPUT.CLASSES
        num_classes = len(classes_names)
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        separate_classes = cfg.MODEL.SEPARATE_CLASSES
        if len(separate_classes) > 0:
          num_classes += len(separate_classes)
        self.num_classes = num_classes

        self.cls_score_body = nn.Linear(representation_size, num_classes)
        self.bbox_pred_body = nn.Linear(representation_size,  3) # z0, z1, thickness
        #self.bbox_pred_body = nn.Linear(representation_size, num_classes * 3) # z0, z1, thickness
        self.bbox_connect_cor = nn.Linear(representation_size, 16) # connection along four directions: binary class at each directon
        self.bbox_pred_cor = nn.Linear(representation_size, 2) # x,y
        self.score_pred_cor = nn.Linear(representation_size, 1)

        nn.init.normal_(self.cls_score_body.weight, std=0.01)
        nn.init.normal_(self.bbox_pred_body.weight, std=0.001)
        nn.init.normal_(self.bbox_pred_cor.weight, std=0.001)
        for l in [self.cls_score_body, self.bbox_pred_body, self.bbox_pred_cor, self.score_pred_cor]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x_cor0, x_cor1, x_body = x
        scores = self.cls_score_body(x_body)
        bbox_body = self.bbox_pred_body(x_body)
        bbox_cor0 = self.bbox_pred_cor(x_cor0)
        bbox_cor1 = self.bbox_pred_cor(x_cor1)

        bbox_corners = torch.cat([bbox_cor0, bbox_cor1, bbox_body], 1)
        bbox_centroids = Box3D_Torch.corner_box_to_yxzb(bbox_corners)

        cor0_score = self.score_pred_cor(x_cor0)
        cor1_score = self.score_pred_cor(x_cor1)
        corner_scores = torch.cat([cor0_score, cor1_score], 1)

        bbox_con_cor0 = self.bbox_connect_cor(x_cor0)
        bbox_con_cor1 = self.bbox_connect_cor(x_cor1)
        bbox_connects = torch.cat([bbox_con_cor0, bbox_con_cor1],1)


        return scores, bbox_centroids, bbox_connects


_ROI_BOX_PREDICTOR = {
    "FastRCNNPredictor": FastRCNNPredictor,
    "FPNPredictor": FPNPredictor,
}


def make_roi_box_predictor(cfg):
    func = _ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg)
