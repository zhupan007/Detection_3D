export CUDA_VISIBLE_DEVICES=0
#export CUDA_LAUNCH_BLOCKING=1 
#ipython tools/train_net_sparse3d.py -- --config-file "configs/sparse_faster_rcnn_NMS_bs1.yaml"  --skip-test
ipython tools/train_net_sparse3d.py -- --config-file "configs/sparse_faster_rcnn_NMS_bs1_difyawl.yaml"  --skip-test
#ipython tools/train_net_sparse3d.py -- --config-file "configs/sparse_faster_rcnn_NMS_bs1.yaml"  --only-test
#ipython tools/train_net_sparse3d.py -- --config-file "configs/sparse_faster_rcnn_NMS_bs18.yaml"  --skip-test
#ipython tools/train_net_sparse3d.py -- --config-file "configs/sparse_faster_rcnn_RPN_ONLY_bs1.yaml"  --skip-test
#ipython tools/train_net_sparse3d.py -- --config-file "configs/sparse_faster_rcnn_RPN_ONLY_bs1.yaml"  --only-test

