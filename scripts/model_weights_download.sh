#!/bin/bash

# SAM
mkdir -p checkpoints/SAM/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    -O checkpoints/SAM/sam_vit_h_4b8939.pth

# depth anything
mkdir -p checkpoints/depth_anything/
wget "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true" \
    -O checkpoints/depth_anything/depth_anything_v2_vitl.pth

mkdir -p checkpoints/YOLO/
wget "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt" \
    -O checkpoints/YOLO/yolov8x.pt

# self-training ViT
gdown "https://drive.google.com/uc?id=1pjSnyWQlFqR1jNB5ysroTzcRwpP6oSz8" \
    -O checkpoints/vit_detection/vit_finetuned.pth