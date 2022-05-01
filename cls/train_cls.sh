#!/bin/sh
EXP_NAME=modified_wsss_2

CUDA_VISIBLE_DEVICES=1 python3 scripts/origin_train_cls.py \
    --img_dir=/root/datasets/VOC2012 \
    --train_list=/root/WSSS/metadata/voc12/train_aug_cls.txt \
    --test_list=/root/WSSS/metadata/voc12/train_cls.txt \
    --epoch=15 \
    --decay_points='5,10' \
    --save_folder=checkpoints/${EXP_NAME} \
    --show_interval=50 \
    --batch_size 5 \
    --wandb_name=${EXP_NAME}
