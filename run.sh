ipython tools/train_net_sparse3d.py -- --config-file "configs/sparse_faster_rcnn.yaml"  --skip-test
#CUDA_LAUNCH_BLOCKING=1 ipython tools/train_net_sparse3d.py -- --config-file "configs/sparse_faster_rcnn.yaml"  --skip-test



#ipython tools/train_net.py -- --config-file "configs/e2e_faster_rcnn_R_50_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 72 SOLVER.STEPS "(48, 64)" TEST.IMS_PER_BATCH 1
