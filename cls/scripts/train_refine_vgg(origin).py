import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import torch
import argparse
import cv2
import time

import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

from models.vgg_refine import vgg16
from utils.my_optim import reduce_lr
from utils.avgMeter import AverageMeter
from utils.LoadData_refine import train_data_loader, valid_data_loader
from utils.Metrics import Cls_Accuracy, IOUMetric
from utils.util import output_visualize
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.util import output_visualize, custom_visualization
import wandb 
from PIL import Image 


def get_arguments():
    parser = argparse.ArgumentParser(description='DRS pytorch implementation')
    
    parser.add_argument("--wandb_name", type=str, default='', help='wandb name')
    
    parser.add_argument("--img_dir", type=str, default='', help='Directory of training images')
    parser.add_argument("--train_list", type=str, default='/home/junehyoung/code/wsss_baseline/metadata/voc12/train_aug_cls.txt')
    parser.add_argument("--test_list", type=str, default='/home/junehyoung/code/wsss_baseline/metadata/voc12/train_cls.txt')
    parser.add_argument('--save_folder', default='checkpoints/test', help='Location to save checkpoint models')
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--crop_size", type=int, default=320)
    parser.add_argument("--num_classes", type=int, default=20)
    
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--decay_points", type=str, default='5,10')
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--shuffle_val", action='store_false')
    parser.add_argument("--custom_vis", action='store_true')
    # parser.add_argument('--logdir', default='logs/test', type=str, help='Tensorboard log dir')
    
    parser.add_argument('--show_interval', default=50, type=int, help='interval of showing training conditions')
    parser.add_argument('--save_interval', default=5, type=int, help='interval of save checkpoint models')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.20, help='object cues for the pseudo seg map generation')

    return parser.parse_args()

def get_model(args):
    model = vgg16(pretrained=True)

    model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    return  model, optimizer

    
def validate(current_epoch):

    print('\nvalidating ... ', flush=True, end='')
    
    mIOU = IOUMetric(num_classes=21)
    
    val_loss = AverageMeter()
    
    model.eval()
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for idx, dat in tqdm(enumerate(val_loader)):
                
            img, label, sal_map, gt_map, att_map, img_name = dat
            
            B, _, H, W = img.size()
            
            label = label.to('cuda', non_blocking=True)
            img = img.to('cuda', non_blocking=True)
            att_map = att_map.to('cuda', non_blocking=True)
            
            refined_map = model(img, label)
            refined_map = refined_map[:, :-1]
            
            """ refinement loss """
            loss = criterion(refined_map, att_map)
            val_loss.update(loss.data.item(), img.size()[0])

            refined_map = refined_map.cpu().detach().numpy()
            gt_map = gt_map.cpu().detach().numpy()
            att_map = att_map.cpu().detach().numpy()
            sal_map = sal_map.cpu().detach().numpy()

            """ segmentation label generation """
            #refined_map = (refined_map - refined_map.min()) / (refined_map.max() - refined_map.min() + 1e-5)
            refined_map[refined_map < args.alpha] = 0  # object cue
            bg = np.zeros((B, 1, H, W), dtype=np.float32)
            pred_map = np.concatenate([bg, refined_map], axis=1)  # [B, 21, H, W]

            pred_map[:, 0, :, :] = (1. - sal_map) # background cue
            pred_map = pred_map.argmax(1)

            mIOU.add_batch(pred_map, gt_map)
            

    """ validation performance """
    res = mIOU.evaluate()
    val_miou = res["Mean_IoU"]
    val_pixel_acc = res["Pixel_Accuracy"]
    recall = res["Recall"]
    precision = res["Precision"]
    tp = res["True Positive"]
    tn = res["True Negative"]
    fp = res["False Positive"]
    
    """ wandb visualization """
    if args.custom_vis:
        custom_visualization(args, valid_data_loader, model)
    else:
        results = []
        result_vis = output_visualize(img[0], refined_map[0], label[0], gt_map[0], pred_map[0])
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
    print('validating Pixel Acc: %.4f' % val_pixel_acc)
    print('validating mIoU: %.4f' % val_miou)
    print('validating Precision: %.4f' % precision)
    print('validating Recall: %.4f' % recall)
    
    return val_miou, val_loss.avg, val_pixel_acc, recall, precision, tp, tn, fp


def train(current_epoch):
    train_loss = AverageMeter()

    model.train()
    
    criterion = nn.MSELoss()
    
    global_counter = args.global_counter

    """ learning rate decay """
    res = reduce_lr(args, optimizer, current_epoch)

    for idx, dat in enumerate(train_loader):
        img, label, sal_map, gt_map, att_map, img_name = dat
        
        label = label.to('cuda', non_blocking=True)
        img = img.to('cuda', non_blocking=True)
        att_map = att_map.to('cuda', non_blocking=True)
        sal_map = sal_map.to('cuda', non_blocking=True)

        refined_map = model(img)
        fg_map = refined_map[:, :-1]
        bg_map = refined_map[:, -1]
        bg_map = F.hardsigmoid(bg_map)
        bg_map = bg_map > 0.6 

        """ refinement loss """
        fg_loss = criterion(fg_map, att_map)
        bg_loss = F.mse_loss(bg_map, sal_map)

        loss = fg_loss + 0.5 * bg_loss 

        """ backprop """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), img.shape[0])
        global_counter += 1

        """ tensorboard log """
        if global_counter % args.show_interval == 0:
            # writer.add_scalar('train loss', train_loss.avg, global_counter)

            print('Epoch: [{}][{}/{}]\t'
                  'LR: {:.5f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    current_epoch, global_counter%len(train_loader), len(train_loader),
                    optimizer.param_groups[0]['lr'], loss=train_loss))

    args.global_counter = global_counter

    return train_loss.val, train_loss.avg, fg_loss, bg_loss 
    

                                   
if __name__ == '__main__':
    args = get_arguments()
    
    nGPU = torch.cuda.device_count()
    print("start training the refinement network , nGPU = %d" % nGPU)
    
    args.batch_size *= nGPU
    args.num_workers *= nGPU
                                   
    print('Running parameters:\n', args)
                                   
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # if not os.path.exists(args.logdir):
    #     os.makedirs(args.logdir)

    # writer = SummaryWriter(log_dir=args.logdir)
    
    train_loader = train_data_loader(args)
    val_loader = valid_data_loader(args)
    print('# of train dataset:', len(train_loader) * args.batch_size)
    print('# of valid dataset:', len(val_loader) * args.batch_size)
    print()

    best_score = 0
    model, optimizer = get_model(args)

    # wandb 
    wandb.init()
    wandb.run.name = args.wandb_name 
    wandb.config.update(args)
    wandb.watch(model)

    for current_epoch in range(1, args.epoch+1):
        loss, train_avg_loss, fg_loss, bg_loss = train(current_epoch)
        score, val_avg_loss, val_pixel_acc, recall, precision, tp, tn, fp = validate(current_epoch)

        """wandb visualization"""
        wandb.log({
                   'Val mIoU' : score,
                   'Recall' : recall,
                   'Precision' : precision,
                   'Foreground Loss' : fg_loss,
                   'Background Loss' : bg_loss,
                #    'True Positive' : tp,
                #    'True Negative' : tn,
                #    'False Postiive' : fp,
                   'Train Avg Loss' : train_avg_loss,
                   'Val Avg Loss' : val_avg_loss,
                   'Val Pixel Acc' : val_pixel_acc,   
                })

        """ save checkpoint """
        if score > best_score:
            best_score = score
            print('\nSaving state, epoch : %d , mIoU : %.4f \n' % (current_epoch, score))
            state = {
                'model': model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                'epoch': current_epoch,
                'iter': args.global_counter,
                'miou': score,
            }
            model_file = os.path.join(args.save_folder, 'best.pth')
            torch.save(state, model_file)