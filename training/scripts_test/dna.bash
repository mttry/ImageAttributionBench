#!/bin/bash  
export CUDA_VISIBLE_DEVICES=1

declare -A config_paths_splits
config_paths_splits[1]="config/model/dna_split1.yaml"
config_paths_splits[2]="config/model/dna_split2.yaml"
config_paths_splits[3]="config/model/dna_split3.yaml"

config_path_default="config/model/dna_default.yaml"  

n=2000
batch_size=8

ckpt_dir="ckpt"
model_name="dna"
checkpoint_default="${ckpt_dir}/${model_name}/${model_name}_default.pth"
declare -A resume_checkpoints
resume_checkpoints[1]="${ckpt_dir}/${model_name}/${model_name}_split1.pth"
resume_checkpoints[2]="${ckpt_dir}/${model_name}/${model_name}_split2.pth"
resume_checkpoints[3]="${ckpt_dir}/${model_name}/${model_name}_split3.pth"
echo "Starting standard testing..."  
python test.py -n $n --config $config_path_default --resume_checkpoint $checkpoint_default --batch_size $batch_size

echo "Starting semantic split testing for task_id 1,2,3..."  
for task_id in 1 2 3; do  
    echo "Testing with task_id=${task_id} ..."
    checkpoint=${resume_checkpoints[$task_id]}
    config_path=${config_paths_splits[$task_id]}
    python test.py -n $n --use_semantic_split --task_id $task_id --config $config_path --resume_checkpoint $checkpoint --batch_size $batch_size
done