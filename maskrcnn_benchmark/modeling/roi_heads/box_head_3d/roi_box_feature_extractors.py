# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers_3d import Pooler
import math

DEBUG = False

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
        )

        self.pooler = pooler
        self.head = head

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES_SPATIAL
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        canonical_size = cfg.MODEL.ROI_BOX_HEAD.CANONICAL_SIZE
        voxel_scale = cfg.SPARSE3D.VOXEL_SCALE

        pooler = Pooler(
            output_size=(resolution[0], resolution[1], resolution[2]),
            scales=scales,
            sampling_ratio=sampling_ratio,
            canonical_size=canonical_size,
            canonical_level=None
        )
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS * resolution[0] * resolution[1] * resolution[2]
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.pooler = pooler
        self.voxel_scale = voxel_scale

        pooler_z = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION[2]
        conv3d_ = nn.Conv3d(cfg.MODEL.BACKBONE.OUT_CHANNELS, representation_size,
                               kernel_size=[1,1,pooler_z], stride=[1,1,1])
        bn = nn.BatchNorm3d(representation_size, track_running_stats=cfg.SOLVER.TRACK_RUNNING_STATS)
        relu = nn.ReLU(inplace=True)
        self.conv3d = nn.Sequential(conv3d_, bn, relu)

        self.fc6 = nn.Linear(input_size, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

        for l in [self.fc6, self.fc7]:
            # Caffe2 implementation uses XavierFill, which in fact
            # corresponds to kaiming_uniform_ in PyTorch
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

    def convert_metric_to_pixel(self, proposals0):
      #print(proposals0[0].bbox3d[:,0])
      proposals = [p.copy() for p in proposals0]
      for prop in proposals:
        prop.bbox3d[:,0:6] *= self.voxel_scale
      #print(proposals0[0].bbox3d[:,0])
      return proposals

    def forward(self, x0, proposals):
        proposals = self.convert_metric_to_pixel(proposals)
        x1_ = self.pooler(x0, proposals)
        x1 = self.conv3d(x1_)

        #x2 = F.relu(self.conv3d(x1))
        x2 = x1.view(x1.size(0), -1)

        x3 = F.relu(self.fc6(x2))
        x4 = F.relu(self.fc7(x3))

        if DEBUG:
          print('\nFPN2MLPFeatureExtractorN:\n')
          scale_num = len(x0)
          print(f"scale_num: {scale_num}")
          for s in range(scale_num):
            print(f"x0[{s}]: {x0[s].features.shape}, {x0[s].spatial_size}")
          print(f'x1: {x1.shape}')
          print(f'x2: {x2.shape}')
          print(f'x3: {x3.shape}')
          print(f'x4: {x4.shape}')
        return x4


def make_roi_box_feature_extractor(cfg):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg)
