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


class deeplabv3plus(nn.Module):
    def __init__(self, cfg):
        super(deeplabv3plus, self).__init__()
        self.backbone = None
        self.backbone_layers = None
        input_channel = 2048
        self.aspp = ASPP(dim_in=input_channel,
                         dim_out=cfg.MODEL_ASPP_OUTDIM,
                         rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                         bn_mom=cfg.TRAIN_BN_MOM)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE // 4)

        indim = 256
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1,
                      padding=cfg.MODEL_SHORTCUT_KERNEL // 2, bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(cfg.MODEL_ASPP_OUTDIM + cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,
                      bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1, bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        # branch 2
        self.shortcut_conv2 = nn.Sequential(
            nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1,
                      padding=cfg.MODEL_SHORTCUT_KERNEL // 2, bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
        )

        self.cat_conv2 = nn.Sequential(
            nn.Conv2d(cfg.MODEL_ASPP_OUTDIM + cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,
                      bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1, bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        # self.offset_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, 4, kernel_size=1, stride=1, padding=0, bias=True)
        self.compress = nn.Conv2d(input_channel, 256, 1, 1, padding=0, bias=True)
        self.upsample4_2 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.offset_conv = Decoder(4)

        self.bbox_attention1 = nn.Sequential(
            nn.Conv2d(cfg.MODEL_NUM_CLASSES, input_channel, 1, 1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.bbox_attention2 = nn.Sequential(
            nn.Conv2d(cfg.MODEL_NUM_CLASSES, indim, 1, 1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()

    def init_output(self, n_sigma=2):
        with torch.no_grad():
            output_conv = self.offset_conv.output_conv
            print('initialize last layer with size: ',
                  output_conv.weight.size())

            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2:2 + n_sigma, :, :].fill_(0)
            output_conv.bias[2:2 + n_sigma].fill_(1)

    def forward(self, x, bbox=None):

        # b, c, h, w = output.shape
        # if bbox is None:
        #     bbox = torch.zeros(b, 9, h, w).to(output.device)
        # else:
        #     bbox = F.interpolate(bbox, size=(h, w))

        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        # layers[0]

        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow1 = self.shortcut_conv(layers[0])
        feature_cat1 = torch.cat([feature_aspp, feature_shallow1], 1)
        result1 = self.cat_conv(feature_cat1)
        result1 = self.cls_conv(result1)
        result1 = self.upsample4(result1)

        feature_shallow2 = self.shortcut_conv2(layers[0])
        compress = self.compress(layers[-1])
        compress = self.upsample4_2(compress)
        feature_cat2 = torch.cat([compress, feature_shallow2], 1)
        result2 = self.cat_conv2(feature_cat2)
        result2 = self.offset_conv(result2)

        return torch.cat([result2, result1], dim=1)


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

        self.layers.append(UpsamplerBlock(256, 128))
        self.layers.append(BasicBlock(inplanes=128, planes=128))
        self.layers.append(BasicBlock(inplanes=128, planes=128))

        self.output_conv = nn.ConvTranspose2d(
            128, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


if __name__ == '__main__':
    from models.deeplabv3plus import Configuration

    cfg = Configuration()
    model = deeplabv3plus(cfg)
    print(model.__class__.__name__)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(y.shape)
    print(nn.ConvTranspose2d(8, out_channels=4, kernel_size=2, stride=2, padding=0, output_padding=0,
                             bias=True).weight.shape)
    # print(model)
    pass
