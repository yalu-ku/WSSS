#!/bin/sh

CUDA_VISIBLE_DEVICES=1,2 python scripts/train_refine.py \
    --img_dir=/HDD/dataset/VOC2012/ \
    --train_list=/home/junehyoung/code/wsss_baseline/voc2012_list/train_aug_cls.txt \
    --test_list=/home/junehyoung/code/wsss_baseline/voc2012_list/train_cls.txt \
    --lr=0.0001 \
    --epoch=30 \
    --decay_points='10,20' \
    --save_folder=checkpoints/Refine_DRS_learnable \
    --show_interval=50 \
    --wandb_name refinement_learning