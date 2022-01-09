# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
import src.pix2pixHD.deeplabv3plus.deeplabv3plus.resnet_atrous as atrousnet
import src.pix2pixHD.deeplabv3plus.deeplabv3plus.xception as xception
from src.pix2pixHD.deeplabv3plus.deeplabv3plus.resnext_101_64x4d import ResNeXt


def build_backbone(backbone_name, pretrained=True, os=16):
    if backbone_name == 'res50_atrous':
        net = atrousnet.resnet50_atrous(pretrained=pretrained, os=os)
    elif backbone_name == 'res101_atrous':
        net = atrousnet.resnet101_atrous(pretrained=pretrained, os=os)
    elif backbone_name == 'res152_atrous':
        net = atrousnet.resnet152_atrous(pretrained=pretrained, os=os)
    elif backbone_name == 'xception' or backbone_name == 'Xception':
        net = xception.xception(pretrained=pretrained, os=os)
    elif backbone_name == 'resnext101_atrous':
        net = ResNeXt(os=os, pretrain=pretrained)
    elif backbone_name == 'xception3strides' or backbone_name == 'Xception3strides':
        net = xception.xception(pretrained=pretrained, os=os, stride3=True)
    else:
        raise ValueError('backbone.py: The backbone named %s is not supported yet.' % backbone_name)
    print(f'===> Backbone: {net.__class__.__name__}')
    return net
