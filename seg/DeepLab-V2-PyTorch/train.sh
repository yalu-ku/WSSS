DATASET=voc12
CONFIG=configs/voc12_2gpu.yaml
LOG_DIR=Deeplabv2_pseudo_segmentation_labels_2gpu
GT_DIR=/home/aro/WSSS_container/datasets/VOC2012/replk_seg

# Training DeepLab-V2 using pseudo segmentation labels
CUDA_VISIBLE_DEVICES=1,2 python3 train.py --config_path ${CONFIG} --gt_path ${GT_DIR} --log_dir ${LOG_DIR}


# evaluation
CUDA_VISIBLE_DEVICES=3 python3 main.py test \
-c configs/${DATASET}.yaml \
-m data/models/${LOG_DIR}/deeplabv2_resnet101_msc/*/checkpoint_final.pth  \
--log_dir=${LOG_DIR}


# evaluate the model with CRF post-processing
CUDA_VISIBLE_DEVICES=3 python3 main.py crf \
-c configs/${DATASET}.yaml \
--log_dir=${LOG_DIR}
