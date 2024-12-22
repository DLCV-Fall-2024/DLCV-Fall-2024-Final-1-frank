#!/bin/bash

python3 predict.py \
--model-base lmsys/vicuna-7b-v1.5 \
--question-file data/test/annotation.json \
--image-folder data/test/images \
--model-path checkpoints/llava-v1.5-7b-lora \
--answers-file results/llava-v1.5-7b-lora/submission.json