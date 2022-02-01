#!/bin/sh

CUDA_VISIBLE_DEVICES=1 python scripts/localization_map_gen.py \
    --img_dir=/HDD/dataset/VOC2012/ \
    --checkpoint=/home/junehyoung/code/wsss_baseline/cls/checkpoints/drs_baseline/best.pth