#!/bin/bash

name="test"

devices="1"
CUDA_VISIBLE_DEVICES=${devices} python ./tasks/semantic/train_novel.py \
    -n ${name} \
    -l "debug" \

