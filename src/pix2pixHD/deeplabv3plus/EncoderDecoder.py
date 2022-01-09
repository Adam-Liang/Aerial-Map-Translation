# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from models.deeplabv3plus.backbone import build_backbone
from models.deeplabv3plus.ASPP import ASPP


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()

        self.offset_conv = Decoder(4)
        self.seed_map_conv = Decoder(cfg.MODEL_NUM_CLASSES)
        self.apply(self.init_bn)

    def init_output(self, n_sigma=2):
        with torch.no_grad():
            output_conv = self.offset_conv.output_conv
            print('initialize last layer with size: ',
                  output_conv.weight.size())

            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2:2 + n_sigma, :, :].fill_(0)
            output_conv.bias[2:2 + n_sigma].fill_(1)

    @staticmethod
    def init_bn(m):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, SynchronizedBatchNorm2d):
            m.eps = 0.001
            m.momentum = 0.1

    def forward(self, x, bbox=None):
        x = self.backbone(x)
        layers = self.backbone.get_layers()

        offset = self.offset_conv(x)
        seed_map = self.seed_map_conv(x)

        return torch.cat([offset, seed_map], dim=1)


def conv3x3(in_planes, out_planes, stride=1, atrous=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1 * atrous, dilation=atrous, bias=False)


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = SynchronizedBatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class BasicBlock(nn.Module):
    expansion = 1
    bn_mom = 0.0003

    def __init__(self, inplanes, planes, stride=1, atrous=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, atrous)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = SynchronizedBatchNorm2d(planes, momentum=self.bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = SynchronizedBatchNorm2d(planes, momentum=self.bn_mom)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(2048, 512))
        self.layers.append(BasicBlock(inplanes=512, planes=512))
        self.layers.append(UpsamplerBlock(512, 128))
        self.layers.append(BasicBlock(inplanes=128, planes=128))
        self.layers.append(UpsamplerBlock(128, 32))
        self.layers.append(BasicBlock(inplanes=32, planes=32))
        self.output_conv = nn.ConvTranspose2d(
            32, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output

def get_deeplabv3plus_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    if name.lower() == 'edxception':
        cfg.MODEL_OUTPUT_STRIDE = 16
        return Net(cfg)


if __name__ == '__main__':
    from models.deeplabv3plus import Configuration

    cfg = Configuration()
    model = Net(cfg)
    print(model.__class__.__name__)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(y.shape)
    print(nn.ConvTranspose2d(8, out_channels=4, kernel_size=2, stride=2, padding=0, output_padding=0,
                             bias=True).weight.shape)
    # print(model)
    pass
