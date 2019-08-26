export PYTHONPATH=$PWD
#export CUDA_LAUNCH_BLOCKING=1 
#export CUDA_VISIBLE_DEVICES=1

#TEST='--skip-test'
#TEST='--only-test' 

#CONFIG_FILE='CF/Cf_Fpn21_bs1_lr5_SD.yaml'
#CONFIG_FILE='CF/Cf_Fpn21_bs1_lr2_SD.yaml'
CONFIG_FILE='CF/Cf_Fpn21_bs1_lr10.yaml'
#CONFIG_FILE='CF/Cf_Fpn21_bs1_lr2.yaml'

ipython tools/train_net_sparse3d.py -- --config-file "configs/$CONFIG_FILE"  $TEST

