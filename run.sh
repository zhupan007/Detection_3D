export CUDA_VISIBLE_DEVICES=0
#ipython tools/train_net_sparse3d.py -- --config-file "configs/sparse_faster_rcnn_RPN_ONLY.yaml"  --only-test
ipython tools/train_net_sparse3d.py -- --config-file "configs/sparse_faster_rcnn_RPN_ONLY_bs1.yaml"  --skip-test

