#!/bin/bash  

export CUDA_VISIBLE_DEVICES=3

config_path="config/model/hifi_net.yaml"  
n=2000
batch_size=8

ckpt_dir="ckpt"
model_name="hifi_net"
checkpoint_default="${ckpt_dir}/${model_name}/${model_name}_default.pth"
declare -A resume_checkpoints
resume_checkpoints[1]="${ckpt_dir}/${model_name}/${model_name}_split1.pth"
resume_checkpoints[2]="${ckpt_dir}/${model_name}/${model_name}_split2.pth"
resume_checkpoints[3]="${ckpt_dir}/${model_name}/${model_name}_split3.pth"
echo "Starting standard testing..."  
python test.py -n $n --config $config_path --resume_checkpoint $checkpoint_default --batch_size $batch_size

echo "Starting semantic split testing for task_id 1,2,3..."  
for task_id in 1 2 3; do  
    echo "Testing with task_id=${task_id} ..."
    checkpoint=${resume_checkpoints[$task_id]}
    python test.py -n $n --use_semantic_split --task_id $task_id --config $config_path --resume_checkpoint $checkpoint --batch_size $batch_size
done