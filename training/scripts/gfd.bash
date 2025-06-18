#!/bin/bash  

export CUDA_VISIBLE_DEVICES=3

config_path="config/model/gfd.yaml"  

log_dir="logs_debug"
n_epoch=15  
n=2000
batch_size=16

# log_dir="logs_debug"
# n_epoch=5
# n=300

mkdir -p $log_dir  


echo "Starting standard training..."  
python train.py -n $n --log_dir $log_dir --config $config_path  --n_epoch $n_epoch --batch_size $batch_size --num_workers 4

echo "Starting semantic split training for task_id 1,2,3..."  
for task_id in 2; do  
    echo "Training with task_id=${task_id} ..."  
    python train.py -n $n --log_dir $log_dir --use_semantic_split --task_id $task_id --config $config_path  --n_epoch $n_epoch --batch_size $batch_size
done  