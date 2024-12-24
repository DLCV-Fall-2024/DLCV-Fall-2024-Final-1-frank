#!/bin/bash

python3 predict.py \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file data/test/annotation.json \
    --image-folder data/test/images \
    --add_seg_img_token \
    --add_detection_token \
    --model-path checkpoints/llava-v1.5-7b-lora_add_image_detection_token \
    --answers-file results/llava-v1.5-7b-lora_add_image_detection_token/submission_add_prompt_engineering.json 