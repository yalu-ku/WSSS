import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
sys.path.append(os.getcwd())

import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
import argparse
import cv2
import time
from PIL import Image 

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# from models.vgg_deform_scaling_learnable import vgg16
# from models.vgg_replkblock import vgg16
# from models.vgg import vgg16
# from models.vgg_offsetscaler_7x7 import vgg16
from models.origin_replk import replk
from utils.my_optim import reduce_lr
from utils.avgMeter import AverageMeter
from utils.LoadData import train_data_loader, valid_data_loader
# from utils.LoadData_coco import train_data_loader, valid_data_loader
from utils.Metrics import Cls_Accuracy, IOUMetric
from models.optim_factory import *
from utils.util import output_visualize, custom_visualization
from tqdm import trange, tqdm
import wandb 
import importlib 
from utils.LoadData import test_data_loader

def get_arguments():
    parser = argparse.ArgumentParser(description='WSSS baseline pytorch implementation')

    parser.add_argument("--img_dir", type=str, default='/root/datasets/VOC2012', help='Directory of training images')
    parser.add_argument("--train_list", type=str, default='/root/WSSS/metadata/voc12/train_aug_cls.txt')
    parser.add_argument("--test_list", type=str, default='/root/WSSS/metadata/voc12/train_cls.txt')
    parser.add_argument('--save_folder', default='checkpoints/XL_Meg73M_ImageNet1K_cam', help='Location to save checkpoint models')

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--crop_size", type=int, default=320)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--shuffle_val", action='store_false')
    parser.add_argument("--custom_vis", action='store_true')
    
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--decay_points", type=str, default='5,10')
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.20, help='object cues for the pseudo seg map generation')

    parser.add_argument('--show_interval', default=50, type=int, help='interval of showing training conditions')
    parser.add_argument('--save_interval', default=5, type=int, help='interval of save checkpoint models')
    parser.add_argument("--global_counter", type=int, default=0)
    
    ########################################################################################################################################
    #                                                        여기서 모델만 바꾸면 됨!                                                        #
    # model list | RepLKNet-31B_ImageNet-1K_384.pth | RepLKNet-31B_ImageNet-22K-to-1K_384.pth | RepLKNet-31L_ImageNet-22K-to-1K_384.pth    #
    #            | RepLKNet-31L_ImageNet-22K.pth    | RepLKNet-XL_MegData73M_ImageNet1K.pth   | RepLKNet-XL_MegData73M_pretrain.pth        #
    ########################################################################################################################################
    # parser.add_argument("--pt_model", type=str, default='/root/WSSS/metadata/RepLKNet-XL_MegData73M_ImageNet1K.pth')
    parser.add_argument("--pt_model", type=str, default='/root/WSSS/checkpoints/XL_Meg73M_ImageNet1K/XL_Meg73M_ImageNet1K_best.pth')
    ########################################################################################################################################
    return parser.parse_args()


def get_model(pt_model):
    if pt_model == '':
        model = replk(pretrained=False)
    else:
        model = replk(pretrained=pt_model)
    optimizer = create_optimizer(model, skip_list=None)
    for param in model.parameters():
        param.requires_grad = False
    
    for name, param in model.named_parameters():
        if name in ['head.weight','gap.weight']:
            param.requires_grad = True

    return model, optimizer


def validate(model, pretrained=''):
    print('\nvalidating ... ', flush=True, end='')
    
    model_name = pretrained.split('/')[-1][:-4]
    
    if 'cam_vis_a' not in os.listdir('/root/WSSS/cls/'):
        os.mkdir(f'/root/WSSS/cls/cam_vis_a/')
    
    if model_name not in os.listdir('/root/WSSS/cls/cam_vis_a'):
        os.mkdir(f'/root/WSSS/cls/cam_vis_a/{model_name}')
    
    mIOU = IOUMetric(num_classes=21)
    cls_acc_matrix = Cls_Accuracy()
    val_loss = AverageMeter()
    
    model.eval()
    
    with torch.no_grad():
        for idx, dat in enumerate(tqdm(data_loader)):
            img, label, sal_map, gt_map, img_name = dat
        
            B, _, H, W = img.size()
            
            label = label.to('cuda', non_blocking=True)
            img = img.to('cuda', non_blocking=True)
            
            logit, cam = model(img, label,size=(H,W))

            """ classification loss """
            loss = F.multilabel_soft_margin_loss(logit, label)
            cls_acc_matrix.update(logit, label)

            val_loss.update(loss.data.item(), img.size()[0])
            
            """ obtain CAMs """
            cam = cam.squeeze(0)
            cam, _ = torch.max(cam, 0)
            cam = cam.cpu().detach().numpy()
            gt_map = gt_map.detach().numpy()
            sal_map = sal_map.detach().numpy()

            """ segmentation label generation """
            
            img = img.squeeze(0)
            image = np.transpose(img.clone().cpu().detach().numpy(), (1,2,0))
            image *= [0.229, 0.224, 0.225]
            image += [0.485, 0.456, 0.406]
            print(image)
            image *= 255
            print(image)
            image = np.clip(image, 0, 255).astype(np.uint8)

            cam_img = cam * 255
            cam_img = np.clip(cam_img, 0, 255)
            cam_img = cv2.applyColorMap(cam_img.astype(np.uint8), cv2.COLORMAP_JET)
            image = cv2.addWeighted(image, 0.5, cam_img, 0.5, 0)
            img_name = img_name[0].split('/')[-1]
            cv2.imwrite(os.path.join("/root/WSSS/cls/cam_vis_a/",model_name, f"{img_name}"), image)

    
if __name__ == '__main__':
    args = get_arguments()
    
    
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    data_loader = valid_data_loader(args)

    model, optimizer = get_model(args.pt_model)
    model.cuda()
        
#     ckpt = torch.load(args.checkpoint, map_location='cpu')
#     model.load_state_dict(ckpt['model'], strict=True)

    # validate(model, args.pt_model)
    
    img = cv2.imread('/root/WSSS/cls/img.jpg')
    W,H,D = img.shape
    img = img.reshape(D, 1,W,H)
    img = torch.from_numpy(img)
    print(type(img))
    
    model.eval()
    label = torch.Tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
         0., 0.]])
            
    label = label.to('cuda', non_blocking=True)
    img = img.to('cuda', non_blocking=True)
            
    logit, cam = model(img, label,size=(H,W))
    
    img = img.squeeze(0)
    image = np.transpose(img.clone().cpu().detach().numpy(), (1,2,0))
    image *= [0.229, 0.224, 0.225]
    image += [0.485, 0.456, 0.406]
    image *= 255
    image = np.clip(image, 0, 255).astype(np.uint8)

    cam_img = cam * 255
    cam_img = np.clip(cam_img, 0, 255)
    cam_img = cv2.applyColorMap(cam_img.astype(np.uint8), cv2.COLORMAP_JET)
    image = cv2.addWeighted(image, 0.5, cam_img, 0.5, 0)
    
    cv2.imwrite("/root/WSSS/cls/img_cam.jpg", image)

    
    print('done!')