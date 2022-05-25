#!/bin/sh
EXP_NAME=XL_Meg73M_ImageNet1K

CUDA_VISIBLE_DEVICES=0 python3 scripts/visualization.py \
    --img_dir=/root/datasets/VOC2012 \
    --train_list=/root/WSSS/metadata/voc12/train_aug_cls.txt \
    --test_list=/root/WSSS/metadata/voc12/train_cls.txt \
    --lr=0.001 \
    --epoch=15 \
    --decay_points='5,10' \
    --save_folder=checkpoints/${EXP_NAME} \
    --show_interval=50 \
    --batch_size 30 \
    --wandb_name=${EXP_NAME} \
    --checkpoint=/root/WSSS/checkpoints/XL_Meg73M_ImageNet1K/XL_Meg73M_ImageNet1K_best.pth