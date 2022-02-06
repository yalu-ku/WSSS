#!/bin/sh
EXP_NAME=vgg_baseline_coco

CUDA_VISIBLE_DEVICES=2 python scripts/train_cls_coco.py \
    --img_dir=/HDD/dataset/COCO14 \
    --train_list=/home/junehyoung/code/wsss_baseline/metadata/coco14/train.txt \
    --test_list=/home/junehyoung/code/wsss_baseline/metadata/coco14/val.txt \
    --lr=0.001 \
    --epoch=15 \
    --decay_points='5,10' \
    --save_folder=checkpoints/${EXP_NAME} \
    --show_interval=50 \
    --batch_size 5 \
    --wandb_name=${EXP_NAME}