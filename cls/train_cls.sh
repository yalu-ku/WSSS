#!/bin/sh
EXP_NAME=modified_wsss_1

python scripts/origin_train_cls.py \
    --img_dir=/root/datasets \
    --train_list=/root/datasets/wsss_baseline2/metadata/voc12/train_aug_cls.txt \
    --test_list=/root/datasets/wsss_baseline2/metadata/voc12/train_cls.txt \
    --lr=0.01 \
    --epoch=15 \
    --decay_points='5,10' \
    --save_folder=checkpoints/${EXP_NAME} \
    --show_interval=50 \
    --batch_size 5 \
    --wandb_name=${EXP_NAME}
