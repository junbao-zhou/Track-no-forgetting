#!/bin/bash

name="test"

split="valid"
split="test"

model="/public/home/meijilin/zhoujunbao/SalsaNext/train_novel0/logs/2024-02-24-03:15:09-MiB-2"

devices="2"
CUDA_VISIBLE_DEVICES=${devices} python ./tasks/semantic/infer.py \
    -n ${name} \
    --split ${split} \
    --model ${model} \
    --is_lora \
    -l "debug_infer" \

