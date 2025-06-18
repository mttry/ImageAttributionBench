#!/bin/bash  

export CUDA_VISIBLE_DEVICES=1

declare -A config_paths_splits
config_paths_splits[1]="config/model/dna_split1.yaml"
config_paths_splits[2]="config/model/dna_split2.yaml"
config_paths_splits[3]="config/model/dna_split3.yaml"

config_path_default="config/model/dna_default.yaml"  

log_dir="logs"
n_epoch=10  
n=2000

# log_dir="logs_try"
# n_epoch=1
# n=2000

mkdir -p $log_dir  


echo "Starting standard training..."  
python train.py -n $n --log_dir $log_dir --config $config_path_default  --n_epoch $n_epoch

echo "Starting semantic split training for task_id 1,2,3..."  
for task_id in 1; do  
    echo "Training with task_id=${task_id} ..."  
    config_path=${config_paths_splits[$task_id]}
    python train.py -n $n --log_dir $log_dir --use_semantic_split --task_id $task_id --config $config_path  --n_epoch $n_epoch 
done  