export PYTHONPATH=$PWD
#export CUDA_LAUNCH_BLOCKING=1 

TEST='--skip-test'
#TEST='--only-test'

CONFIG_FILE='3d1_bs1.yaml'
#CONFIG_FILE='3d1_bs10.yaml'


ipython tools/train_net_sparse3d.py -- --config-file "configs/$CONFIG_FILE"  $TEST

