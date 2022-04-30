import sys
import os

from wsss_baseline2.cls.scripts.models.vgg import replknetwork


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
sys.path.append(os.getcwd())

import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
import argparse
# import cv2
import time
from PIL import Image 

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


from models.vgg_deform_scaling_learnable import vgg16

from utils.my_optim import reduce_lr
from utils.avgMeter import AverageMeter
from utils.LoadData import train_data_loader, valid_data_loader
# from utils.LoadData_coco import train_data_loader, valid_data_loader
from utils.Metrics import Cls_Accuracy, IOUMetric
from utils.util import output_visualize, custom_visualization
from tqdm import trange, tqdm
import wandb 
import importlib 

from wsss_baseline2.cls.scripts.models.vgg_v1 import *
from models.RepLKNet import *

# feature_path = '/root/wsss_baseline2/metadata/RepLKNet-31B_ImageNet-22K-to-1K_384.pth'
def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def get_arguments():
    parser = argparse.ArgumentParser(description='WSSS baseline pytorch implementation')

    parser.add_argument("--wandb_name", type=str, default='', help='wandb name')
    # parser.add_argument("--network", type=str, default='models.vgg', help='wandb name')

    parser.add_argument("--img_dir", type=str, default='/HDD/dataset/VOC2012', help='Directory of training images')
    parser.add_argument("--train_list", type=str, default='../metadata/voc12/train_aug_cls.txt')
    parser.add_argument("--test_list", type=str, default='../metadata/voc12/train_cls.txt')
    parser.add_argument('--save_folder', default='checkpoints/test', help='Location to save checkpoint models')

    # parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--input_size", type=int, default=384) #224
    parser.add_argument("--crop_size", type=float, default=320)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--shuffle_val", action='store_false')
    parser.add_argument("--custom_vis", action='store_true')

    # parser.add_argument("--lr", type=float, default=0.001)
    # parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--decay_points", type=str, default='5,10')
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.20, help='object cues for the pseudo seg map generation')

    parser.add_argument('--show_interval', default=50, type=int, help='interval of showing training conditions')
    parser.add_argument('--save_interval', default=5, type=int, help='interval of save checkpoint models')
    parser.add_argument("--global_counter", type=int, default=0)


    # -----------------------------------------------------
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='RepLKNet-31B', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    # parser.add_argument('--input_size', default=224, type=int, help='image input size')
    parser.add_argument('--test_size', default=-1, type=int, help='test input size')

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    # parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')    # Provided by code of ConvNeXt. Not used.
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='path_to_imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=20, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--with_small_kernel_merged', type=str2bool, default=False,
                        help='Merge small kernels to check the equivalency')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    # parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_amp', type=str2bool, default=False, 
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")
    parser.add_argument('--use_checkpoint', type=str2bool, default=False,
                        help="Use PyTorch's torch.util.checkpoint to save memory or not")


    return parser.parse_args()


def get_model(args):
    # if 'create_RepLKNet31B' in args.network:
    #     model = getattr(importlib.import_module(args.network), 'vgg16')(pretrained=True)
    #     print(model)
    # else:
    #     print('Not RepLKNet')
        
    # if 'vgg' in args.network:
    #     model = getattr(importlib.import_module(args.network), 'vgg16')(pretrained=True)
    # else:
    #     model = getattr(importlib.import_module(args.network), 'resnet50')(pretrained=True)

    model = replk(pretrained=True)
    # model = create_RepLKNet31B(num_classes=20, small_kernel_merged=False, use_checkpoint=False)
    # model = vgg16(pretrained=True)
    
    
    # model.fc = nn.Linear()
    # print(model)
    
    # model = resnet50(pretrained=True)

    ##
    
    model = torch.nn.DataParallel(model).cuda()
    param_groups = RepLKNet.get_parameter_groups
    # get_parameter_groups(param_groups)
    print('param_groups:', param_groups)
    ## 'get_parameter_groups' error
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

def create_RepLKNet31B(drop_path_rate=0.3, num_classes=20, use_checkpoint=True, small_kernel_merged=False):
        return RepLKNet(large_kernel_sizes=[31, 29, 27, 13], layers=[2, 2, 18, 2], channels=[128, 256, 512, 1024], drop_path_rate=drop_path_rate, small_kernel=5, num_classes=num_classes, use_checkpoint=use_checkpoint, use_sync_bn=False, small_kernel_merged=small_kernel_merged)
        ######################################################################## channels 절반식 

def validate(current_epoch):
    print('\nvalidating ... ', flush=True, end='')
    
    mIOU = IOUMetric(num_classes=21) #####################
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
            pred_map = np.concatenate([bg, cam],axis=1)  # [B, 21, H, W]
            pred_map[:, 0, :, :] = (1. - sal_map) # background cue
            pred_map = pred_map.argmax(1) # channel-level maximum 

            mIOU.add_batch(pred_map, gt_map)
    
    """ validation score """
    res = mIOU.evaluate()
    val_miou = res["Mean_IoU"]
    val_pixel_acc = res["Pixel_Accuracy"]
    recall = res["Recall"]
    precision = res["Precision"]
    tn = res["True Negative"]
    fp = res["False Positive"]
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
    print('validating Pixel Acc: %.4f' % val_pixel_acc)
    print('validating mIoU: %.4f' % val_miou)
    print('validating Precision: %.4f' % precision)
    print('validating Recall: %.4f' % recall)
    
    return val_miou, val_loss.avg, val_cls_acc, val_pixel_acc, recall, precision, tp, tn, fp 
 

def train(current_epoch):
    train_loss = AverageMeter()
    cls_acc_matrix = Cls_Accuracy()

    model.train()
    
    global_counter = args.global_counter

    """ learning rate decay """
    res = reduce_lr(args, optimizer, current_epoch)

    for idx, dat in enumerate(train_loader):

        img, label, _ = dat
        # print(dat)
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
    
    # nGPU = torch.cuda.device_count()
    # print("start training the classifier, nGPU = %d" % nGPU)
    
    # args.batch_size *= nGPU
    # args.num_workers *= nGPU
    
    print('Running parameters:\n', args)
    
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    train_loader = train_data_loader(args)
    val_loader = valid_data_loader(args)
    print('# of train dataset:', len(train_loader) * args.batch_size)
    print('# of valid dataset:', len(val_loader) * args.batch_size)
    print()

    best_score = 0
    model, optimizer = get_model(args)
    # print(model)

    # wandb 
    wandb.init()
    wandb.run.name = args.wandb_name 
    wandb.config.update(args)
    wandb.watch(model)

    for current_epoch in range(1, args.epoch+1):
        
        train_cls_acc, loss, train_avg_loss = train(current_epoch)
        score, val_avg_loss, val_cls_acc, val_pixel_acc, recall, precision, tp, tn, fp  = validate(current_epoch)

        """wandb visualization"""
        wandb.log({'Val mIoU' : score,
                   'Recall' : recall,
                   'Precision' : precision,
                #    'True Positive' : tp,
                #    'True Negative' : tn,
                #    'False Postiive' : fp,
                   'Train Acc' : train_cls_acc,
                   'Train Avg Loss' : train_avg_loss,
                   'Val Avg Loss' : val_avg_loss,
                   'Val Acc' : val_cls_acc,
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
        else:
            print(f'\nStill best mIoU is {best_score:.4f}\n')
