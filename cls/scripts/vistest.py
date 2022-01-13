import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import torch
import argparse
import cv2
import time
from PIL import Image 

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from models.vgg import vgg16
from utils.my_optim import reduce_lr
from utils.avgMeter import AverageMeter
from utils.LoadData import train_data_loader, valid_data_loader
from utils.Metrics import Cls_Accuracy, IOUMetric
from utils.util import output_visualize
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb 


def get_arguments():
    parser = argparse.ArgumentParser(description='WSSS baseline pytorch implementation')

    parser.add_argument("--wandb_name", type=str, default='', help='wandb name')

    parser.add_argument("--img_dir", type=str, default='', help='Directory of training images')
    parser.add_argument("--train_list", type=str, default='VOC2012_list/train_aug_cls.txt')
    parser.add_argument("--test_list", type=str, default='VOC2012_list/train_cls.txt')
    parser.add_argument('--save_folder', default='checkpoints/test', help='Location to save checkpoint models')
    parser.add_argument('--checkpoint', default='/home/junehyoung/code/wsss_baseline/cls/checkpoints/baseline_on_test2/best.pth', help='Location to save checkpoint models')

    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--crop_size", type=int, default=320)
    parser.add_argument("--num_classes", type=int, default=20)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--decay_points", type=str, default='5,10')
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.20, help='object cues for the pseudo seg map generation')

    parser.add_argument('--show_interval', default=50, type=int, help='interval of showing training conditions')
    parser.add_argument('--save_interval', default=5, type=int, help='interval of save checkpoint models')
    parser.add_argument("--global_counter", type=int, default=0)

    return parser.parse_args()


def get_model(args):
    model = vgg16(pretrained=True) 

    model = torch.nn.DataParallel(model).cuda()
    param_groups = model.module.get_parameter_groups()
    
    optimizer = optim.SGD([
        {'params': param_groups[0], 'lr': args.lr},
        {'params': param_groups[1], 'lr': 2*args.lr},
        {'params': param_groups[2], 'lr': 10*args.lr},
        {'params': param_groups[3], 'lr': 20*args.lr}], 
        momentum=0.9, 
        weight_decay=args.weight_decay, 
        nesterov=True
    )

    return  model, optimizer


def validate(current_epoch):
    print('\nvalidating ... ', flush=True, end='')
    
    mIOU = IOUMetric(num_classes=21)
    cls_acc_matrix = Cls_Accuracy()
    val_loss = AverageMeter()
    
    model.eval()
    
    with torch.no_grad():
        for idx, dat in enumerate(tqdm(val_loader)):
            if idx > 0:
                break

            img, label, sal_map, gt_map, img_name = dat
            print(img_name)
            
            B, _, H, W = img.size()
            
            label = label.to('cuda', non_blocking=True)
            img = img.to('cuda', non_blocking=True)
            
            logit, cam = model(img, label)

            """ classification loss """
            loss = F.multilabel_soft_margin_loss(logit, label)
            cls_acc_matrix.update(logit, label)

            val_loss.update(loss.data.item(), img.size()[0])
            
            """ obtain CAMs """
            cam = cam.cpu().detach().numpy()
            gt_map = gt_map.detach().numpy()
            sal_map = sal_map.detach().numpy()

            """ segmentation label generation """
            cam[cam < args.alpha] = 0  # object cue
            bg = np.zeros((B, 1, H, W), dtype=np.float32)
            pred_map = np.concatenate([bg, cam], axis=1)  # [B, 21, H, W]
            pred_map[:, 0, :, :] = (1. - sal_map) # background cue
            pred_map = pred_map.argmax(1) # channel-level maximum 

            mIOU.add_batch(pred_map, gt_map)
    
    """ validation score """
    res = mIOU.evaluate()
    val_miou = res["Mean_IoU"]
    val_pixel_acc = res["Pixel_Accuracy"]
    val_cls_acc = cls_acc_matrix.compute_avg_acc()
    
    """ tensorboard visualization """
    # for n in range(min(4, img.shape[0])):
    #     result_vis  = output_visualize(img[n], cam[n], label[n], gt_map[n], pred_map[n])
    # results = [image for image in result_vis]
    result_vis  = output_visualize(img[0], cam[0], label[0], gt_map[0], pred_map[0])
    results = [image for image in result_vis]
        # writer.add_images('valid output %d' % (n+1), result_vis, current_epoch)
    for i, result in enumerate(results):
        print(result.shape)
        result = result.reshape(320, 320, 3) * 255
        
        # image = Image.fromarray((result * 255).astype(np.uint8)).convert("RGB")
        # image.save(f'{i}.png')
        cv2.imwrite(f'{i}.png', result)
        
    # writer.add_scalar('valid loss', val_loss.avg, current_epoch)
    # writer.add_scalar('valid acc', val_cls_acc, current_epoch)
    # writer.add_scalar('valid mIoU', val_miou, current_epoch)
    # writer.add_scalar('valid Pixel Acc', val_pixel_acc, current_epoch)
    
    print('validating loss: %.4f' % val_loss.avg)
    print('validating acc: %.4f' % val_cls_acc)
    print('validating mIoU: %.4f' % val_miou)
    print('validating Pixel Acc: %.4f' % val_pixel_acc)
    
    return val_miou, val_loss.avg, val_cls_acc, val_pixel_acc, results

    
if __name__ == '__main__':
    args = get_arguments()
    nGPU = torch.cuda.device_count()

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # if not os.path.exists(args.logdir):
    #     os.makedirs(args.logdir)

    # writer = SummaryWriter(log_dir=args.logdir)
    
    val_loader = valid_data_loader(args)
    model, optimizer = get_model(args)

    # wandb 
    # wandb.init()
    # wandb.run.name = args.wandb_name 
    # wandb.config.update(args)
    # wandb.watch(model)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)

    for current_epoch in range(1):
        
        score, val_avg_loss, val_cls_acc, val_pixel_acc, result_vis = validate(current_epoch)

        # """wandb visualization"""
        # wandb.log({ 
        #            'Result Vis' : [wandb.Image(image.reshape(image.shape[1], image.shape[2], 3)) for image in result_vis], 
        #         #    'Result Vis' : [wandb.Image(image) for image in result_vis], 
        #         })