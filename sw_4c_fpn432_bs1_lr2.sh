export PYTHONPATH=$PWD
#export CUDA_LAUNCH_BLOCKING=1 
export CUDA_VISIBLE_DEVICES=1

#TEST='--skip-test'
#TEST='--only-test' 

#CONFIG_FILE='SeW4c/SeW4c_Fpn432_bs1_lr2.yaml'
CONFIG_FILE='SeW4c/SeW4c_Fpn432_bs1_lr5_SD.yaml'

ipython tools/train_net_sparse3d.py -- --config-file "configs/$CONFIG_FILE"  $TEST

