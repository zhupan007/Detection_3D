export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=1
#export CUDA_LAUNCH_BLOCKING=1 
TEST='--skip-test'
#TEST='--only-test'
#CONFIG_FILE='nms_bs1_difyaw.yaml'
CONFIG_FILE='nms_bs1_difyaw_dlr.yaml'
#CONFIG_FILE='nms_bs1_sinyaw.yaml'

ipython tools/train_net_sparse3d.py -- --config-file "configs/$CONFIG_FILE"  $TEST

