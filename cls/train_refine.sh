#!/bin/sh

CUDA_VISIBLE_DEVICES=1,2 python scripts/train_refine.py \
    --img_dir=/HDD/dataset/VOC2012/ \
    --train_list=/home/junehyoung/code/wsss_baseline/metadata/voc12/train_aug_cls.txt \
    --test_list=/home/junehyoung/code/wsss_baseline/metadata/voc12/train_cls.txt \
    --lr=0.0001 \
    --epoch=30 \
    --decay_points='10,20' \
    --save_folder=checkpoints/refinement_learning/deform_relu_init \
    --show_interval=50 \
    --wandb_name refinement_learning_deform_relu_init