export PYTHONPATH=$PWD
#export CUDA_LAUNCH_BLOCKING=1 
#export CUDA_VISIBLE_DEVICES=1

#TEST='--skip-test'
#TEST='--only-test' 



CONFIG_FILE='Wall/wall_Fpn432_BEV_bs1_lr20_SD.yaml'
CONFIG_FILE='Wall/wall_Fpn432_BEV_bs1_lr20.yaml'

ipython tools/train_net_sparse3d.py -- --config-file "configs/$CONFIG_FILE"  $TEST

