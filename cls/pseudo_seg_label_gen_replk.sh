CUDA_VISIBLE_DEVICES=1 python3 scripts/pseudo_seg_label_gen.py \
    --checkpoint=/root/WSSS/checkpoints/XL_Meg73M_ImageNet1K/best.pth \
    --refine_dir=replk_seg