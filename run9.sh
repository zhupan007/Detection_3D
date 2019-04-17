export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=0
#export CUDA_LAUNCH_BLOCKING=1 
TEST='--skip-test'
#TEST='--only-test'
CONFIG_FILE='dlr_bs9.yaml'

ipython tools/train_net_sparse3d.py -- --config-file "configs/$CONFIG_FILE"  $TEST

