#!/bin/sh
CUDA_VISIBLE_DEVICES=0,1 python scripts/vistest.py \
    --img_dir=/HDD/dataset/VOC2012/ \
    --train_list=/home/junehyoung/code/wsss_baseline/voc2012_list/train_aug_cls.txt \
    --test_list=/home/junehyoung/code/wsss_baseline/voc2012_list/train_cls.txt \
    --lr=0.001 \
    --epoch=15 \
    --decay_points='5,10' \
    --save_folder=checkpoints/ \
    --show_interval=50 \
    --wandb_name=vis_test