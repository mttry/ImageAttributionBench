#!/bin/bash  

export CUDA_VISIBLE_DEVICES=0  

config_path="config/model/patchcraft.yaml"  

log_dir="logs"
n_epoch=15  
n=2000
num_workers=4


# log_dir="logs_try"
# n_epoch=5
# n=20

mkdir -p $log_dir  

echo "Starting standard training..."  
python train.py -n $n --log_dir $log_dir --config $config_path  --n_epoch $n_epoch --num_workers $num_workers

echo "Starting semantic split training for task_id 1,2,3..."  
for task_id in 2; do  
    echo "Training with task_id=${task_id} ..."  
    python train.py -n $n --log_dir $log_dir --use_semantic_split --task_id $task_id --config $config_path  --n_epoch $n_epoch --num_workers $num_workers