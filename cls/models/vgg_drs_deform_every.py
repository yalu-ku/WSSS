import math

import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from typing import Optional, Tuple
from torchvision.extension import _assert_has_ops


def deform_conv2d(
    input: Tensor,
    offset: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    mask: Optional[Tensor] = None,
) -> Tensor:
    r"""
    Performs Deformable Convolution v2, described in
    `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168>`__ if :attr:`mask` is not ``None`` and
    Performs Deformable Convolution, described in
    `Deformable Convolutional Networks
    <https://arxiv.org/abs/1703.06211>`__ if :attr:`mask` is ``None``.

    Args:
        input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]):
            offsets to be applied for each position in the convolution kernel.
        weight (Tensor[out_channels, in_channels // groups, kernel_height, kernel_width]): convolution weights,
            split into groups of size (in_channels // groups)
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int, int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int, int]): height/width of padding of zeroes around
            each image. Default: 0
        dilation (int or Tuple[int, int]): the spacing between kernel elements. Default: 1
        mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width]):
            masks to be applied for each position in the convolution kernel. Default: None

    Returns:
        Tensor[batch_sz, out_channels, out_h, out_w]: result of convolution

    Examples::
        >>> input = torch.rand(4, 3, 10, 10)
        >>> kh, kw = 3, 3
        >>> weight = torch.rand(5, 3, kh, kw)
        >>> # offset and mask should have the same spatial size as the output
        >>> # of the convolution. In this case, for an input of 10, stride of 1
        >>> # and kernel size of 3, without padding, the output size is 8
        >>> offset = torch.rand(4, 2 * kh * kw, 8, 8)
        >>> mask = torch.rand(4, kh * kw, 8, 8)
        >>> out = deform_conv2d(input, offset, weight, mask=mask)
        >>> print(out.shape)
        >>> # returns
        >>>  torch.Size([4, 5, 8, 8])
    """

    _assert_has_ops()
    out_channels = weight.shape[0]

    use_mask = mask is not None

    if mask is None:
        mask = torch.zeros((input.shape[0], 0), device=input.device, dtype=input.dtype)

    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    weights_h, weights_w = weight.shape[-2:]
    _, n_in_channels, in_h, in_w = input.shape

    n_offset_grps = offset.shape[1] // (2 * weights_h * weights_w)
    n_weight_grps = n_in_channels // weight.shape[1]

    if n_offset_grps == 0:
        raise RuntimeError(
            "the shape of the offset tensor at dimension 1 is not valid. It should "
            "be a multiple of 2 * weight.size[2] * weight.size[3].\n"
            "Got offset.shape[1]={}, while 2 * weight.size[2] * weight.size[3]={}".format(
                offset.shape[1], 2 * weights_h * weights_w))

    return torch.ops.torchvision.deform_conv2d(
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        n_weight_grps,
        n_offset_grps,
        use_mask,)


class DeformConv2d(nn.Module):
    """
    See :func:`deform_conv2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(DeformConv2d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = Parameter(torch.empty(out_channels, in_channels // groups,
                                            self.kernel_size[0], self.kernel_size[1]))

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, offset: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
            offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
                out_height, out_width]): offsets to be applied for each position in the
                convolution kernel.
            mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width,
                out_height, out_width]): masks to be applied for each position in the
                convolution kernel.
        """
        return deform_conv2d(input, offset, self.weight, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation, mask=mask)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import cv2
import numpy as np
import os
import sys
sys.path.append(os.getcwd())

# from models.deformable_conv import DeformConv2d

model_urls = {'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}


class DRS_learnable(nn.Module):
    """ 
    DRS learnable setting
    hyperparameter X , additional training paramters O 
    """
    def __init__(self, channel):
        super(DRS_learnable, self).__init__()
        self.relu = nn.ReLU()
        
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()

        x = self.relu(x)
        
        """ 1: max extractor """
        x_max = self.global_max_pool(x).view(b, c, 1, 1)
        x_max = x_max.expand_as(x)
        
        """ 2: suppression controller"""
        control = self.global_avg_pool(x).view(b, c)
        control = self.fc(control).view(b, c, 1, 1)
        control = control.expand_as(x)
        """ 3: suppressor"""
        x = torch.min(x, x_max * control)
            
        return x
        
    
class DRS(nn.Module):
    """ 
    DRS non-learnable setting
    hyperparameter O , additional training paramters X
    """
    def __init__(self, delta):
        super(DRS, self).__init__()
        self.relu = nn.ReLU()
        self.delta = delta
        
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        x = self.relu(x)
        
        """ 1: max extractor """
        x_max = self.global_max_pool(x).view(b, c, 1, 1)
        x_max = x_max.expand_as(x)
        
        """ 2: suppression controller"""
        control = self.delta
        
        """ 3: suppressor"""
        x = torch.min(x, x_max * control)
        
        return x

    
class VGG(nn.Module):
    def __init__(self, features, delta=0, num_classes=20, init_weights=True):
        
        super(VGG, self).__init__()
        
        self.features = features
        
        self.layer1_conv1 = features[0]
        self.layer1_relu1 = DRS_learnable(64) if delta == 0 else DRS(delta)
        self.layer1_conv2 = features[2]
        self.layer1_relu2 = DRS_learnable(64) if delta == 0 else DRS(delta)
        self.layer1_maxpool = features[4]
        
        self.layer2_conv1 = features[5]
        self.layer2_relu1 = DRS_learnable(128) if delta == 0 else DRS(delta)
        self.layer2_conv2 = features[7]
        self.layer2_relu2 = DRS_learnable(128) if delta == 0 else DRS(delta)
        self.layer2_maxpool = features[9]
        
        self.layer3_conv1 = features[10]
        self.layer3_relu1 = DRS_learnable(256) if delta == 0 else DRS(delta)
        self.layer3_conv2 = features[12]
        self.layer3_relu2 = DRS_learnable(256) if delta == 0 else DRS(delta)
        self.layer3_conv3 = features[14]
        self.layer3_relu3 = DRS_learnable(256) if delta == 0 else DRS(delta)
        self.layer3_maxpool = features[16]
        
        self.layer4_conv1 = features[17]
        self.layer4_relu1 = DRS_learnable(512) if delta == 0 else DRS(delta)
        self.layer4_conv2 = features[19]
        self.layer4_relu2 = DRS_learnable(512) if delta == 0 else DRS(delta)
        self.layer4_conv3 = features[21]
        self.layer4_relu3 = DRS_learnable(512) if delta == 0 else DRS(delta)
        self.layer4_maxpool = features[23]
        
        self.layer5_conv1 = features[24]
        self.layer5_relu1 = DRS_learnable(512) if delta == 0 else DRS(delta)
        self.layer5_conv2 = features[26]
        self.layer5_relu2 = DRS_learnable(512) if delta == 0 else DRS(delta)
        self.layer5_conv3 = features[28]
        self.layer5_relu3 = DRS_learnable(512) if delta == 0 else DRS(delta)

        self.extra_offset_conv1 = nn.Conv2d(512, 18, kernel_size=3)
        self.extra_deform_conv1 = DeformConv2d(512, 512, kernel_size=3)
        self.layer6_relu1 = DRS_learnable(512) if delta == 0 else DRS(delta)

        self.extra_offset_conv2 = nn.Conv2d(512, 18, kernel_size=3)
        self.extra_deform_conv2 = DeformConv2d(512, 512, kernel_size=3)
        self.layer7_relu1 = DRS_learnable(512) if delta == 0 else DRS(delta)

        self.extra_offset_conv3 = nn.Conv2d(512, 18, kernel_size=3)
        self.extra_deform_conv3 = DeformConv2d(512, 512, kernel_size=3)
        self.layer8_relu1 = DRS_learnable(512) if delta == 0 else DRS(delta)

        self.extra_conv = nn.Conv2d(512, 20, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)
        
        if init_weights:
            self._initialize_weights(self.extra_offset_conv1)
            self._initialize_weights(self.extra_deform_conv1)
            self._initialize_weights(self.extra_offset_conv2)
            self._initialize_weights(self.extra_deform_conv2)
            self._initialize_weights(self.extra_offset_conv3)
            self._initialize_weights(self.extra_deform_conv3)
            self._initialize_weights(self.extra_conv)
            
            self._initialize_weights(self.layer1_relu1)
            self._initialize_weights(self.layer1_relu2)
            self._initialize_weights(self.layer2_relu1)
            self._initialize_weights(self.layer2_relu2)
            self._initialize_weights(self.layer3_relu1)
            self._initialize_weights(self.layer3_relu2)
            self._initialize_weights(self.layer3_relu3)
            self._initialize_weights(self.layer4_relu1)
            self._initialize_weights(self.layer4_relu2)
            self._initialize_weights(self.layer4_relu3)
            self._initialize_weights(self.layer5_relu1)
            self._initialize_weights(self.layer5_relu2)
            self._initialize_weights(self.layer5_relu3)
            self._initialize_weights(self.layer6_relu1)
            self._initialize_weights(self.layer7_relu1)
            self._initialize_weights(self.layer8_relu1)
            
        

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
        x = self.layer5_relu2(x)
        x = self.layer5_conv3(x)
        x = self.layer5_relu3(x)
        
        # extra layer
        offset1 = self.extra_offset_conv1(x)
        x = self.extra_deform_conv1(x, offset1)
        x = self.layer6_relu1(x)

        offset1 = self.extra_offset_conv2(x)
        x = self.extra_deform_conv2(x, offset1)
        x = self.layer7_relu1(x)
        
        offset1 = self.extra_offset_conv3(x)
        x = self.extra_deform_conv3(x, offset1)
        x = self.layer8_relu1(x)
        x = self.extra_conv(x)
        
        # ==============================
        
        logit = self.fc(x)
        
        if label is None:
            # for training
            return logit
        
        else:
            # for validation
            cam = self.cam_normalize(x.detach(), size, label)

            return logit, cam

    
    def fc(self, x):
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(-1, 20)
        return x
    
    
    def cam_normalize(self, cam, size, label):
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=size, mode='bilinear', align_corners=True)
        cam /= F.adaptive_max_pool2d(cam, 1) + 1e-5
        
        cam = cam * label[:, :, None, None] # clean
        
        return cam
    
    
    def _initialize_weights(self, layer):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                    

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():

            if 'extra' in name or 'fc' in name:
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

        
        
        
#######################################################################################################
        
    
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
    model = VGG(make_layers(cfg['D1']), delta=delta)
    
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)
        
    return model


if __name__ == '__main__':
    import copy
    
    model = vgg16(pretrained=True, delta=0)    
    input = torch.randn(2, 3, 321, 321)
    out = model(input)
    
    model.get_parameter_groups()
    print(out.shape)