export PYTHONPATH=$PWD
#export CUDA_LAUNCH_BLOCKING=1 
#export CUDA_VISIBLE_DEVICES=1

TEST='--skip-test'
#TEST='--only-test'

CONFIG_FILE='fpn432_2d_bs1_wall.yaml'
#CONFIG_FILE='fpn432_2d_bs1_4c_noaugroi_wall.yaml'

ipython tools/train_net_sparse3d.py -- --config-file "configs/$CONFIG_FILE"  $TEST

