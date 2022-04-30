# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Licensed under The MIT License [see LICENSE for details]
# Based on ConvNeXt, timm, DINO and DeiT code bases
# https://github.com/facebookresearch/ConvNeXt
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import torch
from torch import optim as optim

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
# from timm.optim.novograd import NovoGrad
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

import json

# try:
#     from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
#     has_apex = True
# except ImportError:
#     has_apex = False


def get_num_layer_for_replknet(var_name):
    """
    Divide [2, 2, 18, 2] layers into 12 groups; each group is 2 RepLK BLocks + 2 ConvFFN
    blocks, including possible neighboring transition;
    adapted from https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py
    """
    # num_max_layer = 12
    #   stem:  0
    #   stages.0.blocks.0+1+2+3,  ->   1   (and transitions.0)
    #   stages.1.blocks.0+1+2+3,  ->   2   (and transitions.1)
    #   stages.2.blocks.0+1+2+3,  ->   3
    #   stages.2.blocks.4+5+6+7,  ->   4
    #   stages.2.blocks.8+9+10+11,  ->   5
    #   ...
    #   stages.2.blocks.32+33+34+35,  ->   11   (and transitions.2)
    #   stages.3.blocks.0+1+2+3     ->  12
    if var_name.startswith("stem"):
        return 0
    elif var_name.startswith("stages"):
        stage_id = int(var_name.split('.')[1])
        block_id = int(var_name.split('.')[3])
        if stage_id in [0, 1]:
            layer_id = stage_id + 1
        elif stage_id == 2:
            layer_id = 3 + block_id // 4
        else:
            layer_id = 12
        return layer_id
    elif var_name.startswith('transitions'):
        transition_id = int(var_name.split('.')[1])
        if transition_id in [0, 1]:
            return transition_id + 1
        else:
            return 11
    else:
        return 13


class RepLKNetLayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_replknet(var_name)


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.
            print('the lr scale is', scale)

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    opt_lower = 'adamw'
    weight_decay = 0.05
    # if weight_decay and filter_bias_and_bn:
    if filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    # if 'fused' in opt_lower:
    #     assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=4e-3, weight_decay=weight_decay)
    # if hasattr(args, 'opt_eps') and args.opt_eps is not None:
    opt_args['eps'] = 1e-8
    # if hasattr(args, 'opt_betas') and args.opt_betas is not None:
    # opt_args['betas'] = None

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    # if opt_lower == 'sgd' or opt_lower == 'nesterov':
    #     opt_args.pop('eps', None)
    #     optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    # elif opt_lower == 'momentum':
    #     opt_args.pop('eps', None)
    #     optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    # elif opt_lower == 'adam':
    #     optimizer = optim.Adam(parameters, **opt_args)
    if opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    # elif opt_lower == 'nadam':
    #     optimizer = Nadam(parameters, **opt_args)
    # elif opt_lower == 'radam':
    #     optimizer = RAdam(parameters, **opt_args)
    # elif opt_lower == 'adamp':
    #     optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    # elif opt_lower == 'sgdp':
    #     optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    # elif opt_lower == 'adadelta':
    #     optimizer = optim.Adadelta(parameters, **opt_args)
    # elif opt_lower == 'adafactor':
    #     if not args.lr:
    #         opt_args['lr'] = None
    #     optimizer = Adafactor(parameters, **opt_args)
    # elif opt_lower == 'adahessian':
    #     optimizer = Adahessian(parameters, **opt_args)
    # elif opt_lower == 'rmsprop':
    #     optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    # elif opt_lower == 'rmsproptf':
    #     optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    # elif opt_lower == 'novograd':
    #     optimizer = NovoGrad(parameters, **opt_args)
    # elif opt_lower == 'nvnovograd':
    #     optimizer = NvNovoGrad(parameters, **opt_args)
    # elif opt_lower == 'fusedsgd':
    #     opt_args.pop('eps', None)
    #     optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    # elif opt_lower == 'fusedmomentum':
    #     opt_args.pop('eps', None)
    #     optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    # elif opt_lower == 'fusedadam':
    #     optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    # elif opt_lower == 'fusedadamw':
    #     optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    # elif opt_lower == 'fusedlamb':
    #     optimizer = FusedLAMB(parameters, **opt_args)
    # elif opt_lower == 'fusednovograd':
    #     opt_args.setdefault('betas', (0.95, 0.98))
    #     optimizer = FusedNovoGrad(parameters, **opt_args)
    # else:
    #     assert False and "Invalid optimizer"

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
