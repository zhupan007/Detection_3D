# xyz


# on going process
- rethink how to improve acc for long wall
- add window
- crop gt box with anchor
- multi scale: feature concate
- rpn acc
- 3d roi

# Theory
- yaw: [-pi/2, pi/2]


# Installation

## Envirionment 1 tested
- Ubuntu 18.04.2 LTS
- 1080TI
- conda 4.6.8
- Python 3.7.2
- NVIDIA-SMI 390.116
- Cuda V9.0.176
- gcc 5.5.0
- cmake version 3.13.3

## general
- conda install -c open3d-admin open3d

## maskrcnn
A gcc error occured while builing with this project
Build with original project https://github.com/facebookresearch/maskrcnn-benchmark
- https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md
Then copy the \_C.cpython-37m-x86_64-linux-gnu.so to current prokect

## second
ref: https://github.com/traveller59/second.pytorch  
pip install numba  
Setup cuda for numba: add following to ~/.bashrc: 
``` bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so 
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so 
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice 
```

## SparseConvCnn
ref: https://github.com/facebookresearch/SparseConvNet  
Build with original project and copy  
``` bash
conda install google-sparsehash -c bioconda
conda install -c anaconda pillow
bash develop.sh from  https://github.com/facebookresearch/SparseConvNet
copy SCN.cpython-37m-x86_64-linux-gnu.so
```

## SpConv        
ref: https://github.com/traveller59/spconv  
later, SpConv and SparseConvCnn should only need to install one

## Optinal
- Pymesh: https://pymesh.readthedocs.io/en/latest/installation.html
- pip install plyfile


# Data generation

# Data generation steps for as-built BIM

-  data3d/suncg.py/parse_house()
-  data3d/suncg.py/gen_train_eval_split()
-  data3d/suncg.py/gen_house_names_1level()

-  data3d/indoor_data_util.py/creat_splited_pcl_box()
-  data3d/indoor_data_util.py/creat_indoor_info_file()

* crop_bbox_by_points=False in bbox3d_ops.py/Bbox3D.crop_bbox_by_points
* keep_unseen_intersection=False in indoor_data_util.py/IndoorData.split_bbox

# run
- run.sh

# Debug

 - modeling/rpn/rpn_sparse3d.py RPNModule/forward  
        The targets and anchors for training  

 - modeling/rpn/loss_3d.py  RPNLossComputation/\__call\__  
        SHOW_POS_NEG_ANCHORS: Positive and negative anchors  
        SHOW_PRED_GT: show prediction and ground truth  

 - modeling/rpn/loss_3d.py  RPNLossComputation/match_targets_to_anchors  
        SHOW_POS_ANCHOR_IOU: Show the process of finding positive anchors by iou with ground truth target  

 - rpn/anchor_generator_sparse3d.py  AnchorGenerator/forward:  
        SHOW_ANCHOR_EACH_SCALE:

# Basic code structure
- maskrcnn_benchmark/structures/bounding_box_3d.py/BoxList3D
        Box class used for training
## configurations:
- maskrcnn_benchmark/config/defaults.py 
- configs/sparse_faster_rcnn.yaml

  ### Learning rate
        - maskrcnn_benchmark/solver/lr_scheduler.py

## MODEL
1. tools/train_net_sparse3d.py:main -> :train & test
2. modeling/detector/detectors.py: 
```
build_detection_model -> sparse_rcnn.py:SparseRCNN  
In SparseRCNN:  
features = self.backbone(points)  
proposals, proposal_losses = self.rpn(points, features, targets)  
x, result, detector_losses = self.roi_heads(features, proposals, targets)  
```
3. modeling/backbone/backbone.py:
```
build_backbone -> :build_sparse_resnet_fpn_backbone -> sparseconvnet.FPN_Net  
```
4. modeling/rpn_sparse3d.py: 
```
build_rpn ->  RPNModule -> inference_3d/make_rpn_postprocessor -> loss_3d/make_rpn_loss_evaluator  
```
4.1 RPNModule
```
objectness, rpn_box_regression = self.head(features)  
anchors = self.anchor_generator(points_sparse, features_sparse)  
-> rpn/anchor_generator_sparse3d.py/AnchorGenerator.forward()
```
4.2 modeling/rpn/loss_3d.py:
```
make_rpn_loss_evaluator -> RPNLossComputation  
objectness_loss = torch.nn.functional.binary_cross_entropy_with_logits(...)  
box_loss = smooth_l1_loss(...)  
```
4.3 modeling/rpn/inference_3d.py:
```
make_rpn_postprocessor -> RPNPostProcessor -> structures.boxlist3d_ops.boxlist_nms_3d  
-> second.pytorch.core.box_torch_ops.rotate_nms & multiclass_nms + second.core.non_max_suppression.nms_gpu/rotate_iou_gpu_eval
```

## Data feeding
1. data.py/make_data_loader/

## Evaluation
- maskrcnn_benchmark/data/datasets/evaluation/suncg/suncg_eval.py/do_suncg_evaluation

## Anchor
- rpn/anchor_generator_sparse3d.py/AnchorGenerator.forward()
* ANCHOR_SIZES_3D: [[0.5,1,3], [1,4,3]]
* YAWS: (0, -1.57, -0.785, 0.785)
- BG_IOU_THRESHOLD: 0.1
- FG_IOU_THRESHOLD: 0.3

- flatten order
```
1. anchor_generator_sparse3d.py/AnchorGenerator.grid_anchors:   
        flatten order: [sparse_location_num, yaws_num, 7]     
2. rpn_sparse3d.py/RPNModule/forward ->  bounding_box_3d.py/ cat_scales_anchor:   
        final flatten order: [batch_size, scale_num, sparse_location_num, yaws_num]
3. loss_3d.py/RPNLossComputation.prepare_targets:  
        labels same as anchors   
4. objectness and rpn_box_regression  
        rpn_sparse3d.py/RPNHead.forward: [sparse_location_num, yaws_num]
                reg_shape_method = 'box_toghter' or 'yaws_toghter'  
        rpn_sparse3d.py/cat_scales_obj_reg:         
                flatten order same as anchor
```

### Positive policy **Very Important**
-1:ignore, 0: negative, 1:positive  
Positive anchor: 1. this anchor location is the closest to the target centroid. 2. the feature view field contains the target at most.
- modeling/rpn/loss_3d.py/RPNLossComputation/match_targets_to_anchors:  
        match_quality_matrix = boxlist_iou_3d(anchor, target)  
        matched_idxs = self.proposal_matcher(match_quality_matrix)  
- second.core.non_max_suppression.nms_gpu/rotate_iou_gpu_eval &  devRotateIoUEval:   
        criterion == 2:  area_inter / (area2 + max(0,area1*0.5 - area_inter)), area2 is target  
- modeling/matcher.py/Matcher/__call__

###  model classes
```
- SparseRCNN:  maskrcnn_benchmark/modeling/detector/sparse_rcnn.py
- RPNModule: maskrcnn_benchmark/modeling/rpn/rpn_sparse3d.py
- RPNPostProcessor: maskrcnn_benchmark/modeling/rpn/rpn_sparse3d.py
```

### maskrcnn_benchmark call second
```
- maskrcnn_benchmark/structures/boxlist3d_ops.py:
        from second.core.non_max_suppression.nms_gpu import rotate_iou_gpu_eval
        from second.pytorch.core.box_torch_ops import rotate_nms

- maskrcnn_benchmark/modeling/box_coder_3d.py
        from second.pytorch.core.box_torch_ops import second_box_encode, second_box_decode
```

### maskrcnn_benchmark call sparse_faster_rcnn
```
- modeling/backbone/backbone.py/build_sparse_resnet_fpn_backbone:
        fpn = scn.FPN_Net(full_scale, dimension, raw_elements, block_reps, nPlanesF,...)
```

# Ideas for the future
- 3D object detection by keypoint
- 3D object detection with deformable convolution
- BIM detection aided by constrain of connection relationship
- Indoor navigation with 3d mapping of BIM 
