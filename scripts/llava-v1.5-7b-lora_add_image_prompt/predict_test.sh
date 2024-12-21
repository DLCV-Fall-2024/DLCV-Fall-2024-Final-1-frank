#!/bin/bash

python3 predict.py \
--model-path checkpoints/llava-v1.5-7b-lora_add_image_prompt \
--model-base lmsys/vicuna-7b-v1.5 \
--question-file data/test/annotation.json \
--image-folder data/test/images \
--answers-file results/llava-v1.5-7b-lora_add_image_prompt/submission.json \
--add_region_prompt 