#!/bin/bash

python3 predict.py \
--model-path checkpoints/llava-v1.5-7b-lora \
--model-base lmsys/vicuna-7b-v1.5 \
--question-file data/val/annotation.json \
--image-folder data/val/images \
--answers-file results/llava-v1.5-7b-lora/submission_val.json