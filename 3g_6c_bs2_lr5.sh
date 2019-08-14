export PYTHONPATH=$PWD
#export CUDA_LAUNCH_BLOCKING=1 
export CUDA_VISIBLE_DEVICES=1

#TEST='--skip-test'
#TEST='--only-test' 

#CONFIG_FILE='t1_sw6c/3groups_6c_fpn4321_bs1_lr10.yaml'

CONFIG_FILE='sw6c/3g6c_fpn4321_bs2_lr5.yaml'

ipython tools/train_net_sparse3d.py -- --config-file "configs/$CONFIG_FILE"  $TEST

