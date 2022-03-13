#!/bin/sh

CUDA_VISIBLE_DEVICES=3 python scripts/localization_map_gen.py \
    --img_dir=/HDD/dataset/VOC2012/ \
    --checkpoint=/home/junehyoung/code/wsss_baseline2/cls/checkpoints/offset_smooth_b03d05/best.pth \
    --locmap_dir=localization_map_offset_smooth_b03d05