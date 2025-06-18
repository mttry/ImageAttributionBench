#!/bin/bash  

export CUDA_VISIBLE_DEVICES=2

config_path="config/model/repmix.yaml"  

log_dir="logs_debug"
n_epoch=10  
n=2000

# log_dir="logs_try"
# n_epoch=5
# n=20

mkdir -p $log_dir  

echo "Starting standard training..."  
python train.py -n $n --log_dir $log_dir --config $config_path  --n_epoch $n_epoch --resume_checkpoint $resume_ckpt

echo "Starting semantic split training for task_id 1,2,3..."  
for task_id in 1 2 3; do  
    echo "Training with task_id=${task_id} ..."  
    python train.py -n $n --log_dir $log_dir --use_semantic_split --task_id $task_id --config $config_path  --n_epoch $n_epoch 
done  