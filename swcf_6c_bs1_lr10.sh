export PYTHONPATH=$PWD
#export CUDA_LAUNCH_BLOCKING=1 
#export CUDA_VISIBLE_DEVICES=1

#TEST='--skip-test'
#TEST='--only-test' 

#CONFIG_FILE='t1_sw6c/sw6c_fpn4321_bs1_lr10.yaml'
CONFIG_FILE='t1_sw6c/swcf6c_fpn4321_bs1_lr10.yaml'

ipython tools/train_net_sparse3d.py -- --config-file "configs/$CONFIG_FILE"  $TEST

