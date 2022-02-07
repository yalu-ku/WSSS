CUDA_VISIBLE_DEVICES=1 python scripts/pseudo_seg_label_gen.py \
    --img_dir=/HDD/dataset/VOC2012/ \
    --checkpoint=/home/junehyoung/code/wsss_baseline/cls/checkpoints/vgg_deform_offsetscaler_relu_init/best.pth \
    --refine_dir=vgg_deform_offsetscaler_relu_init_without_refinement