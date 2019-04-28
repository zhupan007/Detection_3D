export PYTHONPATH=$PWD
#export CUDA_LAUNCH_BLOCKING=1 

TEST='--skip-test'
#TEST='--only-test'

CONFIG_FILE='proj2d_4c.yaml'
CONFIG_FILE='sparse_3d_4c.yaml'

#export CUDA_VISIBLE_DEVICES=1
#CONFIG_FILE='proj2d_2c.yaml'

ipython tools/train_net_sparse3d.py -- --config-file "configs/$CONFIG_FILE"  $TEST

