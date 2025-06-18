#!/bin/bash  

export CUDA_VISIBLE_DEVICES=1

config_path="config/model/dna_pretrain.yaml"  

log_dir="logs"
n_epoch=10  
n=2000

# log_dir="logs_try"
# n_epoch=1
# n=20

mkdir -p $log_dir  
root_dir="/remote-home/share/gzy/attribution/final_dataset_thats_real"

echo "Starting standard training..."  
python train.py -n $n --log_dir $log_dir --config $config_path  --n_epoch $n_epoch --root_dir $root_dir

echo "Starting semantic split training for task_id 1,2,3..."  
for task_id in 1; do  
    echo "Training with task_id=${task_id} ..."  
    python train.py -n $n --log_dir $log_dir --use_semantic_split --task_id $task_id --config $config_path  --n_epoch $n_epoch  --root_dir $root_dir --do_test
done  