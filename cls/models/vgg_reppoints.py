import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import cv2
import numpy as np
import os

import torch
import torchvision.ops
from torch import nn

import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from typing import Optional, Tuple
from torchvision.extension import _assert_has_ops

# from reppoints_utils import PointGenerator
from models.deformable_conv import DeformConv2d

model_urls = {'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}

class VGG(nn.Module):

    def __init__(self, features, num_classes=20, init_weights=True):
        
        super(VGG, self).__init__()
        
        self.features = features
        
        self.layer1_conv1 = features[0]
        self.layer1_relu1 = features[1]
        self.layer1_conv2 = features[2]
        self.layer1_relu2 = features[3]
        self.layer1_maxpool = features[4]
        
        self.layer2_conv1 = features[5]
        self.layer2_relu1 = features[6]
        self.layer2_conv2 = features[7]
        self.layer2_relu2 = features[8]
        self.layer2_maxpool = features[9]
        
        self.layer3_conv1 = features[10]
        self.layer3_relu1 = features[11]
        self.layer3_conv2 = features[12]
        self.layer3_relu2 = features[13]
        self.layer3_conv3 = features[14]
        self.layer3_relu3 = features[15]
        self.layer3_maxpool = features[16]
        
        self.layer4_conv1 = features[17]
        self.layer4_relu1 = features[18]
        self.layer4_conv2 = features[19]
        self.layer4_relu2 = features[20]
        self.layer4_conv3 = features[21]
        self.layer4_relu3 = features[22]
        self.layer4_maxpool = features[23]
        
        self.layer5_conv1 = features[24]
        self.layer5_relu1 = features[25]
        self.layer5_conv2 = features[26]
        self.layer5_relu2 = features[27]
        self.layer5_conv3 = features[28]
        self.layer5_relu3 = features[29]
        
        self.extra_conv4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.activation = nn.ReLU()

        self.dcn_kernel = 3
        self.dcn_pad = 1

        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

        self.extra_offset_conv1 = nn.Conv2d(512, 18, kernel_size=3, stride=1, padding=1)
        self.extra_offset_conv2 = nn.Conv2d(512, 18, kernel_size=1, stride=1, padding=0)
        self.extra_deform_conv1 = DeformConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        if init_weights:
            self._initialize_weights()

            
    def forward(self, x, label=None, size=None):
        if size is None:
            size = x.size()[2:]
            
        # layer1
        x = self.layer1_conv1(x)
        x = self.layer1_relu1(x)
        x = self.layer1_conv2(x)
        x = self.layer1_relu2(x)
        x = self.layer1_maxpool(x)
        
        # layer2
        x = self.layer2_conv1(x)
        x = self.layer2_relu1(x)
        x = self.layer2_conv2(x)
        x = self.layer2_relu2(x)
        x = self.layer2_maxpool(x)
        
        # layer3
        x = self.layer3_conv1(x)
        x = self.layer3_relu1(x)
        x = self.layer3_conv2(x)
        x = self.layer3_relu2(x)
        x = self.layer3_conv3(x)
        x = self.layer3_relu3(x)
        x = self.layer3_maxpool(x)
        
        # layer4
        x = self.layer4_conv1(x)
        x = self.layer4_relu1(x)
        x = self.layer4_conv2(x)
        x = self.layer4_relu2(x)
        x = self.layer4_conv3(x)
        x = self.layer4_relu3(x)
        x = self.layer4_maxpool(x)
        
        # layer5
        x = self.layer5_conv1(x)
        x = self.layer5_relu1(x)

        x = self.layer5_conv2(x)
        x1 = self.layer5_relu2(x)
        x = self.layer5_conv3(x1)

        init_offset = self.extra_offset_conv1(x) # 1x1, 18
        init_offset_detached = init_offset.detach()
        offset1 = (1 - 0.1) * init_offset_detached + 0.1 * init_offset
        offset1 = (offset1 - self.dcn_base_offset.to(offset1.device)).float()
        

        deform = self.extra_deform_conv1(x1, offset1) # 3x3 512 
        deform = self.activation(deform)
        offset2 = self.extra_offset_conv2(deform) # 1x1, 18

        reppoints2 = init_offset_detached + offset2     

        # extra layer
        x = self.extra_conv4(x)
        
        logit = self.fc(x) 

        if label is None:
            return logit, reppoints2 
        else:
            cam = self.cam_normalize(x.detach(), size, label)
            return logit, cam

    def fc(self, x):
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(-1, 20)
        return x

    def cam_normalize(self, cam, size, label):
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=size, mode='bilinear', align_corners=True)
        cam /= F.adaptive_max_pool2d(cam, 1) + 1e-5 # normalize 
        cam = cam * label[:, :, None, None] # clean

        return cam

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                    
    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():

            if 'extra' in name:
                if 'weight' in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if 'weight' in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            if i > 13:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, dilation=2, padding=2)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }


def vgg16(pretrained=True, delta=0):
    model = VGG(make_layers(cfg['D1']))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)
        
    return model


# if __name__ == '__main__':
#     from reppoints_utils import (PointGenerator, get_points, offset_to_pts,
#                                 points2bbox)

#     # model = vgg16(pretrained=True)
#     # input = torch.randn(1, 3, 320, 320)
#     # logit, reppoints = model(input)
#     # print(reppoints.shape)

#     # # ========================RepPoints============================

#     # # hyperparameters 
#     # featmap_sizes = [reppoints.shape[-2:]] # (32, 32)
#     # img_num = reppoints.shape[0] # 1 
#     # point_strides = [4]
#     # point_generators = [PointGenerator() for _ in point_strides]
#     # num_points = 9
#     # transform_method = "minmax"
#     # moment_transfer = nn.Parameter(data=torch.zeros(2), requires_grad=True)
#     # moment_mul = 0.01

#     # center_list = get_points(featmap_sizes, img_num, point_strides, point_generators)
#     # pts_coordinate_preds_init = offset_to_pts(center_list, [reppoints], point_strides, num_points)

#     # bbox_pred_init = points2bbox(
#     #         pts_coordinate_preds_init[0].reshape(-1, 2 * 9), 
#     #         y_first=False, transform_method=transform_method)
#     # print(bbox_pred_init.shape)
#     # ==============================================================
#     import sys
#     sys.path.append(os.getcwd())
#     from utils.transforms import transforms 
#     from PIL import Image 
    
#     mean_vals = [0.485, 0.456, 0.406]
#     std_vals = [0.229, 0.224, 0.225]

#     img_name = "/home/junehyoung/code/wsss_baseline/cls/figure/dog.png"
    
#     tsfm_train = transforms.Compose([transforms.Resize(384),  
#                                      transforms.RandomHorizontalFlip(),
#                                      transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
#                                      transforms.RandomCrop(320),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize(mean_vals, std_vals),
#                                      ])
#     model = vgg16(pretrained=True)
#     img = Image.open(img_name).convert('RGB')
#     logit, reppoints = model(input)

#     featmap_sizes = [reppoints.shape[-2:]] # (32, 32)
#     img_num = reppoints.shape[0] # 1 
#     point_strides = [4]
#     point_generators = [PointGenerator() for _ in point_strides]
#     num_points = 9
#     transform_method = "minmax"

#     center_list = get_points(featmap_sizes, img_num, point_strides, point_generators)
#     pts_coordinate_preds_init = offset_to_pts(center_list, [reppoints], point_strides, num_points)

#     bbox_pred_init = points2bbox(
#             pts_coordinate_preds_init[0].reshape(-1, 2 * 9), 
#             y_first=False, transform_method=transform_method)
#     print(bbox_pred_init.shape)