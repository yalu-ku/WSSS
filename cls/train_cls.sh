#!/bin/sh
EXP_NAME=VGG16

CUDA_VISIBLE_DEVICES=0 python3 scripts/train_cls.py \
    --img_dir=/root/datasets/VOC2012 \
    --train_list=/root/WSSS/metadata/voc12/train_aug_cls.txt \
    --test_list=/root/WSSS/metadata/voc12/train_cls.txt \
    --epoch=15 \
    --decay_points='5,10' \
    --save_folder=checkpoints/${EXP_NAME} \
    --show_interval=50 \
    --batch_size 30 \
    --pt_model '' \
    --wandb_name=${EXP_NAME}
