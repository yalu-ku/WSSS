import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import cv2
import numpy as np
import os

from models.deformable_conv import DeformConv2d

model_urls = {'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}

class VGG(nn.Module):
    def __init__(self, features, num_classes=20, init_weights=True):
        
        super(VGG, self).__init__()
        self.features = features
        self.num_classes = num_classes
        # deform conv1 
        self.extra_offset_conv1 = nn.Conv2d(512, 18, kernel_size=3)
        self.extra_deform_conv1 = DeformConv2d(512, 512, kernel_size=3)
        self.extra_modulate_conv1 = nn.Conv2d(512, 9, kernel_size=3)

        # deform conv2 
        self.extra_offset_conv2 = nn.Conv2d(512, 18, kernel_size=3)
        self.extra_deform_conv2 = DeformConv2d(512, 512, kernel_size=3)
        self.extra_modulate_conv2 = nn.Conv2d(512, 9, kernel_size=3)

        # deform conv3 
        self.extra_offset_conv3 = nn.Conv2d(512, 18, kernel_size=3)
        self.extra_deform_conv3 = DeformConv2d(512, 512, kernel_size=3)
        self.extra_modulate_conv3 = nn.Conv2d(512, 9, kernel_size=3)

        self.extra_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)

        if init_weights:
            self._initialize_weights()

    def forward(self, x, label=None, size=None):
        if size is None:
            size = x.size()[2:] # (H, W)

        x = self.features(x)
        
        offset1 = self.extra_offset_conv1(x)
        # modulator1 = 2. * torch.sigmoid(self.extra_modulate_conv1(x))
        modulator1 = torch.sigmoid(self.extra_modulate_conv1(x))
        x = self.extra_deform_conv1(x, offset=offset1, mask=modulator1)

        offset2 = self.extra_offset_conv2(x)
        # modulator2 = 2. * torch.sigmoid(self.extra_modulate_conv2(x))
        modulator2 = torch.sigmoid(self.extra_modulate_conv2(x))
        x = self.extra_deform_conv2(x, offset=offset2, mask=modulator2)

        offset3 = self.extra_offset_conv3(x)
        # modulator3 = 2. * torch.sigmoid(self.extra_modulate_conv3(x))
        modulator3 = torch.sigmoid(self.extra_modulate_conv3(x))
        x = self.extra_deform_conv3(x, offset=offset3, mask=modulator3)

        x = self.extra_conv(x)
        
        logit = self.fc(x) 
        
        if label is None:
            return logit
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


if __name__ == '__main__':
    model = vgg16(pretrained=True)
    print(model)