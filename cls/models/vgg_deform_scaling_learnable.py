import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math

# from models.deformable_conv import DeformConv2d
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

model_urls = {'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}


class OffsetScaler(nn.Module):
    def __init__(self, in_ch=512, out_ch=18, delta=0.8):
        super(OffsetScaler, self).__init__()
        
        self.in_ch = in_ch 
        self.out_ch = out_ch 
        self.delta = delta 
        self.offset = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3)
        self.scale = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3)

    def forward(self, x):
        offset = self.offset(x)
        scale = self.scale(x)
        scale = F.hardsigmoid(scale, inplace=False) 
        scale += self.delta
        scaled_offset = offset * scale  
        
        return scaled_offset 

class OffsetScaler_Preserve(nn.Module):
    def __init__(self, in_ch=512, out_ch=18, delta=0.8):
        super(OffsetScaler_Preserve, self).__init__()
        
        self.in_ch = in_ch 
        self.out_ch = out_ch 
        self.delta = delta 
        self.offset = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3)
        self.scale = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3)

    def forward(self, x):
        offset = self.offset(x)
        scale = self.scale(x)
        scale = F.hardsigmoid(scale, inplace=False) 
        scale += self.delta
        scale = torch.max(torch.ones(scale.shape).cuda(), scale)
        scaled_offset = offset * scale  
        
        return scaled_offset 

class OffsetScaler_Smooth(nn.Module):
    def __init__(self, in_ch=512, out_ch=18, beta=0.7, delta=0.3):
        super(OffsetScaler_Smooth, self).__init__()
        
        self.in_ch = in_ch 
        self.out_ch = out_ch 
        self.delta = delta 
        self.beta = beta 
        self.offset = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3)
        self.scale = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        offset = self.offset(x)
        scale = self.scale(x)
        
        ############ Offset Smoothing ##########
        b, c, _, _ = scale.size()
        scale_max = self.global_max_pool(scale).view(b, c, 1, 1)
        scale_max = scale_max.expand_as(scale)
        control = self.delta 
        scale = torch.min(scale, scale_max * control)
        ########################################
        
        scale = F.hardsigmoid(scale, inplace=False) 
        scale += self.beta
        scaled_offset = offset * scale  
        
        return scaled_offset 

class OffsetScaler_Learnable(nn.Module):
    def __init__(self, in_ch=512, out_ch=18):
        super(OffsetScaler_Learnable, self).__init__()
        
        self.in_ch = in_ch 
        self.out_ch = out_ch 

        self.offset = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3)
        self.scale = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, out_ch),
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        offset = self.offset(x)
        scale = self.scale(x)
        scale = F.hardsigmoid(scale, inplace=False) 

        # learnable delta 
        # TODO : 512 -> 18 
        delta_lr = self.avg_pool(x).view(b, c)
        delta_lr = self.fc(delta_lr)#.view(b, c, 1, 1)
        delta_lr = F.hardsigmoid(delta_lr, inplace=False) 
        delta_lr = delta_lr.view(b, self.out_ch, 1, 1).expand_as(scale)
        scale += delta_lr
        scaled_offset = offset * scale  

        return scaled_offset 

class OffsetScaler_Log(nn.Module):
    def __init__(self, in_ch=512, out_ch=18):
        super(OffsetScaler_Log, self).__init__()
        
        self.in_ch = in_ch 
        self.out_ch = out_ch 
        self.offset = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3)
        self.scale = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3)

    def forward(self, x):
        offset = self.offset(x)
        scale = self.scale(x)
        scale = F.hardsigmoid(scale, inplace=False) 
<<<<<<< HEAD
        # scale = 0.8 - torch.log(1 - 0.9 * scale + 1e-6)
        # scale = 0.3 - torch.log10(1 - 0.9 * scale + 1e-6)
        scale = 0.3 - torch.log10(1 - 0.93 * scale + 1e-6)
=======
        scale = 0.8 - torch.log(1 - 0.9 * scale + 1e-6)
>>>>>>> e19c506cb3ce6fae34880bff76021eeecf5e9300
        # scale = 0.8 + torch.log(1 + 9 * scale + 1e-6)
        scaled_offset = offset * scale  
        
        return scaled_offset 
        

class VGG(nn.Module):
    def __init__(self, features, num_classes=20, init_weights=True):
        
        super(VGG, self).__init__()
        self.features = features
<<<<<<< HEAD
        # self.extra_offset_conv1 = OffsetScaler(512, 18, delta=0.2)
        # self.extra_offset_conv1 = OffsetScaler_Preserve(512, 18, delta=0.8)
        # self.extra_offset_conv1 = OffsetScaler_Learnable(512, 18)
        # self.extra_offset_conv1 = OffsetScaler_Log(512, 18)
        self.extra_offset_conv1 = OffsetScaler_Smooth(512, 18, beta=0.3, delta=0.7)
        self.extra_deform_conv1 = DeformConv2d(512, 512, kernel_size=3)
        self.relu1 = nn.ReLU(True)

        # self.extra_offset_conv2 = OffsetScaler(512, 18, delta=0.2)
        # self.extra_offset_conv2 = OffsetScaler_Preserve(512, 18, delta=0.8)
        # self.extra_offset_conv2 = OffsetScaler_Learnable(512, 18)
        # self.extra_offset_conv2 = OffsetScaler_Log(512, 18)
        self.extra_offset_conv2 = OffsetScaler_Smooth(512, 18, beta=0.3, delta=0.7)
        self.extra_deform_conv2 = DeformConv2d(512, 512, kernel_size=3)
        self.relu2 = nn.ReLU(True)

        # self.extra_offset_conv3 = OffsetScaler(512, 18, delta=0.2)
        # self.extra_offset_conv3 = OffsetScaler_Preserve(512, 18, delta=0.8)
        # self.extra_offset_conv3 = OffsetScaler_Learnable(512, 18)
        # self.extra_offset_conv3 = OffsetScaler_Log(512, 18)
        self.extra_offset_conv3 = OffsetScaler_Smooth(512, 18, beta=0.3, delta=0.7)
=======
        # self.extra_offset_conv1 = OffsetScaler(512, 18, delta=0.5)
        # self.extra_offset_conv1 = OffsetScaler_Learnable(512, 18)
        self.extra_offset_conv1 = OffsetScaler_Log(512, 18)
        self.extra_deform_conv1 = DeformConv2d(512, 512, kernel_size=3)
        self.relu1 = nn.ReLU(True)

        # self.extra_offset_conv2 = OffsetScaler(512, 18, delta=0.6)
        # self.extra_offset_conv2 = OffsetScaler_Learnable(512, 18)
        self.extra_offset_conv2 = OffsetScaler_Log(512, 18)
        self.extra_deform_conv2 = DeformConv2d(512, 512, kernel_size=3)
        self.relu2 = nn.ReLU(True)

        # self.extra_offset_conv3 = OffsetScaler(512, 18, delta=0.7)
        # self.extra_offset_conv3 = OffsetScaler_Learnable(512, 18)
        self.extra_offset_conv3 = OffsetScaler_Log(512, 18)
>>>>>>> e19c506cb3ce6fae34880bff76021eeecf5e9300
        self.extra_deform_conv3 = DeformConv2d(512, 512, kernel_size=3)
        self.relu3 = nn.ReLU(True)

        self.extra_conv = nn.Conv2d(512, 20, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)

        if init_weights:
            self._initialize_weights()

    def forward(self, x, label=None, size=None):
        if size is None:
            size = x.size()[2:] # (H, W)

        x = self.features(x)
        offset1 = self.extra_offset_conv1(x)
        x = self.extra_deform_conv1(x, offset1)
        x = self.relu1(x)

        offset2 = self.extra_offset_conv2(x)
        x = self.extra_deform_conv2(x, offset2)
        x = self.relu2(x)

        offset3 = self.extra_offset_conv3(x)
        x = self.extra_deform_conv3(x, offset3)
        x = self.relu3(x)

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
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, OffsetScaler):
                # offset initialization 
                n = m.offset.kernel_size[0] * m.offset.kernel_size[1] * m.offset.out_channels 
                m.offset.weight.data.normal_(0, math.sqrt(2. / n))
                if m.offset.bias is not None:
                    m.offset.bias.data.zero_()

                # scaler initialization 
                n = m.scale.kernel_size[0] * m.scale.kernel_size[1] * m.scale.out_channels 
                m.scale.weight.data.normal_(0, math.sqrt(2. / n))
                if m.scale.bias is not None:
                    m.scale.bias.data.zero_()

            elif isinstance(m, OffsetScaler_Learnable):
                # offset initialization 
                n = m.offset.kernel_size[0] * m.offset.kernel_size[1] * m.offset.out_channels 
                m.offset.weight.data.normal_(0, math.sqrt(2. / n))
                if m.offset.bias is not None:
                    m.offset.bias.data.zero_()

                # scaler initialization 
                n = m.scale.kernel_size[0] * m.scale.kernel_size[1] * m.scale.out_channels 
                m.scale.weight.data.normal_(0, math.sqrt(2. / n))
                if m.scale.bias is not None:
                    m.scale.bias.data.zero_()

            elif isinstance(m, OffsetScaler_Smooth):
                # offset initialization 
                n = m.offset.kernel_size[0] * m.offset.kernel_size[1] * m.offset.out_channels 
                m.offset.weight.data.normal_(0, math.sqrt(2. / n))
                if m.offset.bias is not None:
                    m.offset.bias.data.zero_()

                # scaler initialization 
                n = m.scale.kernel_size[0] * m.scale.kernel_size[1] * m.scale.out_channels 
                m.scale.weight.data.normal_(0, math.sqrt(2. / n))
                if m.scale.bias is not None:
                    m.scale.bias.data.zero_()

            elif isinstance(m, OffsetScaler_Preserve):
                # offset initialization 
                n = m.offset.kernel_size[0] * m.offset.kernel_size[1] * m.offset.out_channels 
                m.offset.weight.data.normal_(0, math.sqrt(2. / n))
                if m.offset.bias is not None:
                    m.offset.bias.data.zero_()

                # scaler initialization 
                n = m.scale.kernel_size[0] * m.scale.kernel_size[1] * m.scale.out_channels 
                m.scale.weight.data.normal_(0, math.sqrt(2. / n))
                if m.scale.bias is not None:
                    m.scale.bias.data.zero_()

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

    input = torch.randn(1, 3, 320, 320)
    output = model(input)

    print(output.shape)