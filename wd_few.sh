export PYTHONPATH=$PWD
#export CUDA_LAUNCH_BLOCKING=1 
#export CUDA_VISIBLE_DEVICES=1

TEST='--skip-test'
TEST='--only-test'

CONFIG_FILE='fpn432_3d1_bs1_wd_lr10_2S.yaml'
#CONFIG_FILE='fpn432_3d1_bs1_wall_lr10_2S.yaml'

ipython tools/train_net_sparse3d.py -- --config-file "configs/$CONFIG_FILE"  $TEST

