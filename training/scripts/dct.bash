#!/bin/bash  

export CUDA_VISIBLE_DEVICES=6 

config_path="config/model/dct.yaml"  

log_dir="logs"
n_epoch=10  
n=2000

# log_dir="logs_try"
# n_epoch=2
# n=200

mkdir -p $log_dir  

# restore from here:


echo "Starting standard training..."  
python train.py -n $n --log_dir $log_dir --config $config_path  --n_epoch $n_epoch

echo "Starting semantic split training for task_id 1,2,3..."  
for task_id in 1 2 3; do  
    echo "Training with task_id=${task_id} ..."  
    python train.py -n $n --log_dir $log_dir --use_semantic_split --task_id $task_id --config $config_path  --n_epoch $n_epoch 
done  