export PYTHONPATH=$PWD
#export CUDA_LAUNCH_BLOCKING=1 
##export CUDA_VISIBLE_DEVICES=1

#TEST='--skip-test'
#TEST='--only-test' 



#CONFIG_FILE='walls/wall_Fpn4321_bs1_lr5_rpn.yaml'
#CONFIG_FILE='walls/wall_Fpn4321_bs1_lr5_corsem.yaml'
#
#CONFIG_FILE='walls/wall_Fpn4321_bs1_lr20_SD_corsem.yaml'
CONFIG_FILE='walls/wall_Fpn4321_bs1_lr20_SD_rpn.yaml'

ipython tools/train_net_sparse3d.py -- --config-file "configs/$CONFIG_FILE"  $TEST

