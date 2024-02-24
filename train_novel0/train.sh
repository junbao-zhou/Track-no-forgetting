#!/bin/bash

COMMAND="python ./tasks/semantic/train_novel.py $@"

echo $COMMAND

devices="1"
CUDA_VISIBLE_DEVICES=${devices} $COMMAND