import torch
import numpy as np
import cv2
import os

from utils.decode import decode_seg_map_sequence


def load_img_id_list(img_id_file):
    return open(img_id_file).read().splitlines()


def load_img_label_list_from_npy(img_name_list, dataset):
    cls_labels_dict = np.load(f'/home/junehyoung/code/wsss_baseline/voc2012_list/cls_labels.npy', allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]

def output_visualize(image, cam, label, gt_map, pred_map):
    
    image = np.transpose(image.clone().cpu().detach().numpy(), (1,2,0))  # H, W, C
    cam = np.transpose(cam, (1,2,0)) # H, W, C
    
    """ image denormalize """
    image *= [0.229, 0.224, 0.225]
    image += [0.485, 0.456, 0.406]
    image *= 255
    image = np.clip(image.transpose(2,0,1), 0, 255).astype(np.uint8) # C, H, W

    size = image.shape[1]

    """ visualize selected CAM outputs """
    label = label.clone().cpu().detach().numpy()
    label = np.nonzero(label)[0]

    selected_cam_image = np.zeros((len(label)+3, 3, size, size), dtype=np.uint8) # ((image, cam1, cam2, .., pseudo, gt), 3, 320, 320)
    selected_cam_image[0] = image
    
    for n, i in enumerate(label):
        cam_img = cam[:, :, i] # H, W
        cam_img *= 255
        cam_img = np.clip(cam_img, 0, 255)

        cam_img = cv2.applyColorMap(cam_img.astype(np.uint8), cv2.COLORMAP_JET) # H, W, 3
        cam_img = cam_img[:, :, ::-1]

        selected_cam_image[n+1] = cam_img.transpose(2, 0, 1)

    """ visualize semantic segmentaiton map """
    selected_cam_image[-1] = decode_seg_map_sequence(gt_map) * 255
    selected_cam_image[-2] = decode_seg_map_sequence(pred_map) * 255
        
    selected_cam_image = selected_cam_image.astype(np.float32) / 255.
        
    return selected_cam_image