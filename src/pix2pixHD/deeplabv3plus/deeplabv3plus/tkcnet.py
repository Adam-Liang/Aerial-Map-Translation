###########################################################################
# Created by: Tianyi Wu
# Email: wutianyi@ict.ac.cn
# Copyright (c) 2018
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate, normalize
from model.deeplabv3plus.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from model.deeplabv3plus.backbone import build_backbone

__all__ = ['TKCNet']


class TKCNet(nn.Module):
    """Tree-structured Kronecker Convolutional Networks for Semantic Segmentation,
      Note that:
        In our pytorch implementation of TKCN: for KConv(r_1,r_2), we use AvgPool2d(kernel_size = r_2, stride=1)
        and Conv2d( kernel_size =3, dilation = r_1) to approximate it.
        The original codes (caffe)  will be relesed later .
    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
    """

    def __init__(self, cfg, TFA='l'):
        super().__init__()
        if TFA.lower() == 's':
            r1=[6, 10, 20]
            r2=[3, 7, 15]
        elif TFA.lower() == 'l':
            r1 = [10, 20, 30]
            r2 = [7, 15, 25]
        else:
            raise NotImplementedError

        self.head = TFAHead(2048, cfg.MODEL_NUM_CLASSES, SynchronizedBatchNorm2d,
                            norm_kwargs={'momentum': cfg.TRAIN_BN_MOM},
                            r1=r1, r2=r2)
        self.backbone = None
        self.backbone_layers = None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(cfg.MODEL_BACKBONE, pretrained=True,os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()

    def forward(self, x):
        # print("in tkcnet.forward(): input_size: ", x.size())  #input.size == crop_size
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        imsize = x.size()[2:]
        x = self.head(layers[-1])
        out = interpolate(x, imsize, mode='bilinear', align_corners=True)
        # out1 = [out]
        # out1 = tuple(out1)
        return out


class TFAHead(nn.Module):
    """
    TFA_S r1, r2= {(6,3), (10, 7), (20, 15)}
    TFA_S r1, r2= {(10,7), (20, 15), (30, 25)}
       input:
        x: B x C x H x W  (C = 2048)
       output: B x nClass x H x W
    """

    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs, r1, r2):
        super(TFAHead, self).__init__()
        # TFA module
        inter_channels = in_channels // 4  # 2048-->512
        self.TFA_level_1 = self._make_level(2048, inter_channels, r1[0], r2[0], norm_layer, norm_kwargs)
        self.TFA_level_list = nn.ModuleList()
        for i in range(1, len(r1)):
            self.TFA_level_list.append(self._make_level(inter_channels, inter_channels, r1[i], r2[i],
                                                        norm_layer, norm_kwargs))

        # segmentation subnetwork
        self.conv51 = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels * len(r1), inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **norm_kwargs),
            nn.ReLU())
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def _make_level(self, inChannel, outChannel, r1, r2, norm_layer, norm_kwargs):
        # print(norm_kwargs)
        avg_agg = nn.AvgPool2d(r2, stride=1, padding=r2 // 2)
        conv = nn.Sequential(nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=1, padding=r1, dilation=r1),
                             norm_layer(outChannel, **norm_kwargs),
                             nn.ReLU())
        return nn.Sequential(avg_agg, conv)

    def forward(self, x):
        # print(*self.TFA_level_list)
        TFA_out_list = []
        TFA_out_list.append(x)
        level_1_out = self.TFA_level_1(x)
        TFA_out_list.append(level_1_out)
        for i, layer in enumerate(self.TFA_level_list):
            if i == 0:
                output1 = layer(level_1_out)
                TFA_out_list.append(output1)
            else:
                output1 = layer(output1)
                TFA_out_list.append(output1)
        TFA_out = torch.cat(TFA_out_list, 1)
        # print(TFA_out.shape)

        out = self.conv51(TFA_out)  # B x 4096 x H x W  --> B x 512 x H x W
        out = self.conv6(out)  # B x nClass x H x W
        return out



class TFAHeadwithoutCls(nn.Module):
    """
    TFA_S r1, r2= {(6,3), (10, 7), (20, 15)}
    TFA_S r1, r2= {(10,7), (20, 15), (30, 25)}
       input:
        x: B x C x H x W  (C = 2048)
       output: B x nClass x H x W
    """

    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs, r1, r2):
        super().__init__()
        # TFA module
        inter_channels = in_channels // 4  # 2048-->512
        self.TFA_level_1 = self._make_level(2048, inter_channels, r1[0], r2[0], norm_layer, norm_kwargs)
        self.TFA_level_list = nn.ModuleList()
        for i in range(1, len(r1)):
            self.TFA_level_list.append(self._make_level(inter_channels, inter_channels, r1[i], r2[i],
                                                        norm_layer, norm_kwargs))

        # segmentation subnetwork
        self.conv51 = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels * len(r1), inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **norm_kwargs),
            nn.ReLU())

    def _make_level(self, inChannel, outChannel, r1, r2, norm_layer, norm_kwargs):
        # print(norm_kwargs)
        avg_agg = nn.AvgPool2d(r2, stride=1, padding=r2 // 2)
        conv = nn.Sequential(nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=1, padding=r1, dilation=r1),
                             norm_layer(outChannel, **norm_kwargs),
                             nn.ReLU())
        return nn.Sequential(avg_agg, conv)

    def forward(self, x):
        # print(*self.TFA_level_list)
        TFA_out_list = []
        TFA_out_list.append(x)
        level_1_out = self.TFA_level_1(x)
        TFA_out_list.append(level_1_out)
        for i, layer in enumerate(self.TFA_level_list):
            if i == 0:
                output1 = layer(level_1_out)
                TFA_out_list.append(output1)
            else:
                output1 = layer(output1)
                TFA_out_list.append(output1)
        TFA_out = torch.cat(TFA_out_list, 1)
        # print(TFA_out.shape)

        out = self.conv51(TFA_out)  # B x 4096 x H x W  --> B x 512 x H x W
        out = self.conv6(out)  # B x nClass x H x W
        return out

class TKCNetv2(nn.Module):
    """Tree-structured Kronecker Convolutional Networks for Semantic Segmentation,
      Note that:
        In our pytorch implementation of TKCN: for KConv(r_1,r_2), we use AvgPool2d(kernel_size = r_2, stride=1)
        and Conv2d( kernel_size =3, dilation = r_1) to approximate it.
        The original codes (caffe)  will be relesed later .
    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
    """

    def __init__(self, cfg, TFA='l'):
        super().__init__()
        if TFA.lower() == 's':
            r1=[6, 10, 20]
            r2=[3, 7, 15]
        elif TFA.lower() == 'l':
            r1 = [10, 20, 30]
            r2 = [7, 15, 25]
        else:
            raise NotImplementedError
        self.head = TFAHead(2048, cfg.MODEL_ASPP_OUTDIM, SynchronizedBatchNorm2d,
                            norm_kwargs={'momentum': cfg.TRAIN_BN_MOM},
                            r1=[10, 20, 30], r2=[7, 15, 25])
        self.backbone = None
        self.backbone_layers = None
        indim = 256
        cfg.MODEL_SHORTCUT_DIM = 96
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1,
                      padding=cfg.MODEL_SHORTCUT_KERNEL // 2, bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
        )
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
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
        self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(cfg.MODEL_BACKBONE, pretrained=True,os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()


    def forward(self, x):
        # print("in tkcnet.forward(): input_size: ", x.size())  #input.size == crop_size
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        imsize = x.size()[2:]
        x = self.head(layers[-1])
        x = self.up4(x)
        shallow = self.shortcut_conv(layers[0])
        x = self.cat_conv(torch.cat([x, shallow], dim=1))
        x = self.cls_conv(x)
        out = interpolate(x, imsize, mode='bilinear', align_corners=True)
        # out1 = [out]
        # out1 = tuple(out1)
        return out

if __name__ == '__main__':
    from model.deeplabv3plus import Configuration
    cfg = Configuration()
    model = TKCNetv2(cfg)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(y.shape)