CUDA_VISIBLE_DEVICES=2 python3 scripts/pseudo_seg_label_gen.py \
    --checkpoint=/root/WSSS/cls/checkpoints/VGG16/best.pth \
    --refine_dir=VGG__seg
