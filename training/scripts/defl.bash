#!/bin/bash  

export CUDA_VISIBLE_DEVICES=4

config_path="config/model/defl.yaml"  

log_dir="logs"
n_epoch=10  
n=2000
batch_size=8
num_workers=2
# log_dir="logs_try"
# n_epoch=1
# n=2
# batch_size=8

mkdir -p $log_dir  
echo "Starting standard training..."  
python train.py -n $n --log_dir $log_dir --config $config_path  --n_epoch $n_epoch --batch_size $batch_size  --num_workers $num_workers

echo "Starting semantic split training for task_id 1,2,3..."  
for task_id in 1 2 3; do  
    echo "Training with task_id=${task_id} ..."  
    python train.py -n $n --log_dir $log_dir --use_semantic_split --task_id $task_id --config $config_path  --n_epoch $n_epoch --num_workers $num_workers --batch_size $batch_size
done  