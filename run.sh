export CUDA_VISIBLE_DEVICES=1
#ipython tools/train_net_sparse3d.py -- --config-file "configs/sparse_faster_rcnn_RPN_ONLY.yaml"  --only-test
ipython tools/train_net_sparse3d.py -- --config-file "configs/sparse_faster_rcnn_RPN_ONLY.yaml"  --skip-test

