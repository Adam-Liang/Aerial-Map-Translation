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
from models.deeplabv3plus.deeplabv3plus import Decoder


class deeplabv3plusmultioffset(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.MODEL_AUX_OUT == 2 + 2 * cfg.MODEL_NUM_CLASSES
        self.cfg = cfg
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
        self.offset_conv = Decoder(cfg.MODEL_AUX_OUT)
        self.seed_map_conv = Decoder(cfg.MODEL_NUM_CLASSES)

        self.bbox_attention1 = nn.Sequential(
            nn.Conv2d(cfg.MODEL_NUM_CLASSES + 1, indim, 3, 1, padding=1, bias=True),
            nn.Sigmoid()
        )

        self.bbox_attention2 = nn.Sequential(
            nn.Conv2d(cfg.MODEL_NUM_CLASSES + 1, input_channel, 3, 1, padding=1, bias=True),
            nn.Sigmoid()
        )
        # self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.init_output()

        self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()
        self.apply(self.init_bn)

    def init_output(self, n_sigma=2):
        """
        sigma, 2 * n_class, n_class
        :param n_sigma:
        :return:
        """
        with torch.no_grad():
            output_conv = self.offset_conv.output_conv
            print('initialize last layer with size: ', output_conv.weight.size())

            output_conv.weight[:, 0:n_sigma, :, :].fill_(0)
            output_conv.bias[0:2].fill_(1)

            output_conv.weight[:, n_sigma: 2 * self.cfg.MODEL_NUM_CLASSES, :, :].fill_(0)
            output_conv.bias[2:2 + n_sigma].fill_(0)

    @staticmethod
    def init_bn(m):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, SynchronizedBatchNorm2d):
            m.eps = 0.001
            m.momentum = 0.1

    def forward(self, x, bbox=None):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        _, _, h0, w0 = layers[0].shape
        _, _, h2, w2 = layers[2].shape
        b, _, _, _ = x.shape

        if bbox is None:
            bbox_1 = torch.zeros(b, self.cfg.MODEL_NUM_CLASSES + 1, h0, w0).to(x.device)
            bbox_2 = torch.zeros(b, self.cfg.MODEL_NUM_CLASSES + 1, h2, w2).to(x.device)
        else:
            bbox_1 = F.interpolate(bbox, size=(h0, w0))
            bbox_2 = F.interpolate(bbox, size=(h2, w2))

        bbox_attention_1 = self.bbox_attention1(bbox_1)
        bbox_attention_2 = self.bbox_attention2(bbox_2)
        layers[0] = layers[0] * bbox_attention_1
        layers[2] = layers[2] * bbox_attention_2

        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        feature = self.cat_conv(feature_cat)

        offset = self.offset_conv(feature)
        seed_map = self.seed_map_conv(feature)

        return torch.cat([offset, seed_map], dim=1)


class deeplabv3pluslabel(deeplabv3plusmultioffset):
    def __init__(self, cfg):
        super().__init__(cfg)
        input_channel = 2048
        indim = 256
        self.label_attention1 = nn.Sequential(
            nn.Conv2d(cfg.MODEL_NUM_CLASSES + 1, indim, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.label_attention2 = nn.Sequential(
            nn.Conv2d(cfg.MODEL_NUM_CLASSES + 1, input_channel, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, label=None):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        _, _, h0, w0 = layers[0].shape
        _, _, h2, w2 = layers[2].shape
        b, _, _, _ = x.shape

        if label is None:
            label = torch.zeros(b, self.cfg.MODEL_NUM_CLASSES + 1, 1, 1).to(x.device)
        else:
            assert label.size(-1) == 1 and label.size(-2) == 1

        bbox_attention_1 = self.bbox_attention1(label)
        bbox_attention_2 = self.bbox_attention2(label)
        layers[0] = layers[0] * bbox_attention_1
        layers[2] = layers[2] * bbox_attention_2

        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        feature = self.cat_conv(feature_cat)

        offset = self.offset_conv(feature)
        seed_map = self.seed_map_conv(feature)

        return torch.cat([offset, seed_map], dim=1)


class deeplabv3plus3branch(deeplabv3plusmultioffset):
    def __init__(self, cfg):
        super().__init__(cfg)
        # background
        self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES + 1, 1, 1, padding=0)
        self.seg = False

    def forward(self, x, bbox=None):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        _, _, h0, w0 = layers[0].shape
        _, _, h2, w2 = layers[2].shape
        b, _, _, _ = x.shape

        if bbox is None:
            bbox_1 = torch.zeros(b, self.cfg.MODEL_NUM_CLASSES + 1, h0, w0).to(x.device)
            bbox_2 = torch.zeros(b, self.cfg.MODEL_NUM_CLASSES + 1, h2, w2).to(x.device)
        else:
            bbox_1 = F.interpolate(bbox, size=(h0, w0), mode='bilinear', align_corners=True)
            bbox_2 = F.interpolate(bbox, size=(h2, w2), mode='bilinear', align_corners=True)

        bbox_attention_1 = self.bbox_attention1(bbox_1)
        bbox_attention_2 = self.bbox_attention2(bbox_2)
        layers[0] = layers[0] * bbox_attention_1
        layers[2] = layers[2] * bbox_attention_2

        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        feature = self.cat_conv(feature_cat)

        offset = self.offset_conv(feature)
        seed_map = self.seed_map_conv(feature)
        cls = self.cls_conv(feature)
        cls = self.upsample4(cls)
        if self.seg:
            return torch.cat([offset, seed_map], dim=1), cls
        else:
            return torch.cat([offset, seed_map], dim=1)
