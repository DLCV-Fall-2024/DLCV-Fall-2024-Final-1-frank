#!/bin/bash

mkdir -p checkpoints/SAM/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O \
checkpoints/SAM/sam_vit_h_4b8939.pth
