import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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

# from models.vgg import vgg16
# from models.vgg_deform import vgg16
# from models.vgg_deform_v2 import vgg16
# from models.vgg_drs import vgg16
# from models.vgg_baseline import vgg16
# from models.vgg_drs_deform import vgg16 
# from models.vgg_deform_midlayer import vgg16
# from models.vgg_reppoints import vgg16
# from models.vgg_drs_deform_every import vgg16
# from models.vgg_drs_deform_v2_every import vgg16
# from models.vgg_deform_scaling import vgg16
# from models.vgg_deform_scaling_learnable import vgg16
# from models.vgg_deform_v2_scaling import vgg16
# from models.vgg_deform_scaling_learnable import vgg16
# from models.resnet import resnet50
from models.vgg_coco import vgg16

from utils.my_optim import reduce_lr
from utils.avgMeter import AverageMeter
# from utils.LoadData import train_data_loader, valid_data_loader
from utils.LoadData_coco import train_data_loader, valid_data_loader
from utils.Metrics import Cls_Accuracy, IOUMetric
from utils.util import output_visualize, custom_visualization
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb 
import importlib 


def get_arguments():
    parser = argparse.ArgumentParser(description='WSSS baseline pytorch implementation')

    parser.add_argument("--wandb_name", type=str, default='', help='wandb name')
    # parser.add_argument("--network", type=str, default='models.vgg', help='wandb name')

    parser.add_argument("--img_dir", type=str, default='', help='Directory of training images')
    parser.add_argument("--train_list", type=str, default='/home/junehyoung/code/wsss_baseline/metadata/voc12/train_aug_cls.txt')
    parser.add_argument("--test_list", type=str, default='/home/junehyoung/code/wsss_baseline/metadata/voc12/train_cls.txt')
    parser.add_argument('--save_folder', default='checkpoints/test', help='Location to save checkpoint models')

    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--crop_size", type=int, default=320)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--shuffle_val", action='store_false')
    parser.add_argument("--custom_vis", action='store_true')

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
    # if 'vgg' in args.network:
    #     model = getattr(importlib.import_module(args.network), 'vgg16')(pretrained=True)
    # else:
    #     model = getattr(importlib.import_module(args.network), 'resnet50')(pretrained=True)

    model = vgg16(pretrained=True) 
    # model = resnet50(pretrained=True)

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
            img, label, sal_map, gt_map, _ = dat
            
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
    
    """wandb visualization"""
    if args.custom_vis:
        custom_visualization(args, valid_data_loader, model)
    else:
        results = []
        result_vis = output_visualize(img[0], cam[0], label[0], gt_map[0], pred_map[0])
        label_num = result_vis.shape[0] - 3

        for i in range(result_vis.shape[0]):
            vis = np.transpose(result_vis[i], (1, 2, 0)) * 255
            vis = vis.astype(np.uint8)
            image = Image.fromarray(vis).convert('RGB')
            results.append(image)

        titles = ['image'] + [f'CAM_{i}' for i in range(1, label_num+1)] + ['pseudo-mask', 'GT']
        wandb.log({ 
                'Result Visualization' : [wandb.Image(image, caption=titles[i]) for i, image in enumerate(results)], 
                })
    
    print('validating loss: %.4f' % val_loss.avg)
    print('validating acc: %.4f' % val_cls_acc)
    print('validating mIoU: %.4f' % val_miou)
    print('validating Pixel Acc: %.4f' % val_pixel_acc)
    
    return val_miou, val_loss.avg, val_cls_acc, val_pixel_acc
    

def train(current_epoch):
    train_loss = AverageMeter()
    cls_acc_matrix = Cls_Accuracy()

    model.train()
    
    global_counter = args.global_counter

    """ learning rate decay """
    res = reduce_lr(args, optimizer, current_epoch)

    for idx, dat in enumerate(train_loader):

        img, label, _ = dat
        label = label.to('cuda', non_blocking=True)
        img = img.to('cuda', non_blocking=True)

        logit = model(img)
        """ classification loss """
        loss = F.multilabel_soft_margin_loss(logit, label)

        """ backprop """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cls_acc_matrix.update(logit, label)
        train_loss.update(loss.data.item(), img.size()[0])
        
        global_counter += 1

        """ tensorboard log """
        if global_counter % args.show_interval == 0:
            train_cls_acc = cls_acc_matrix.compute_avg_acc()

            print('Epoch: [{}][{}/{}]\t'
                  'LR: {:.5f}\t'
                  'ACC: {:.5f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    current_epoch, idx+1, len(train_loader),
                    optimizer.param_groups[0]['lr'], 
                    train_cls_acc, loss=train_loss)
                 )

    args.global_counter = global_counter

    return train_cls_acc, train_loss.val, train_loss.avg
    
    
if __name__ == '__main__':
    args = get_arguments()
    
    nGPU = torch.cuda.device_count()
    print("start training the classifier, nGPU = %d" % nGPU)
    
    args.batch_size *= nGPU
    args.num_workers *= nGPU
    
    print('Running parameters:\n', args)
    
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    train_loader = train_data_loader(args)
    # val_loader = valid_data_loader(args)
    print('# of train dataset:', len(train_loader) * args.batch_size)
    # print('# of valid dataset:', len(val_loader) * args.batch_size)
    print()

    best_score = 0
    train_loss = 1e+5 
    model, optimizer = get_model(args)

    # wandb 
    wandb.init()
    wandb.run.name = args.wandb_name 
    wandb.config.update(args)
    wandb.watch(model)

    for current_epoch in range(1, args.epoch+1):
        
        train_cls_acc, loss, train_avg_loss = train(current_epoch)
        # score, val_avg_loss, val_cls_acc, val_pixel_acc = validate(current_epoch)

        """wandb visualization"""
        wandb.log({'Train Acc' : train_cls_acc,
                   'Train Avg Loss' : train_avg_loss,
                #    'Val mIoU' : score,
                #    'Val Avg Loss' : val_avg_loss,
                #    'Val Acc' : val_cls_acc,
                #    'Val Pixel Acc' : val_pixel_acc,   
                })

        """ save checkpoint """
        # if score > best_score:
        if train_loss < loss:
            # best_score = score
            loss = train_loss
            # print('\nSaving state, epoch : %d , mIoU : %.4f \n' % (current_epoch, score))
            print('\nSaving state, epoch : %d , Loss : %d \n' % (current_epoch, loss))
            state = {
                'model': model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                'epoch': current_epoch,
                'iter': args.global_counter
            }
            model_file = os.path.join(args.save_folder, 'best.pth')
            torch.save(state, model_file)
        # else:
            # print(f'\nStill best mIoU is {best_score:.4f}%\n')
