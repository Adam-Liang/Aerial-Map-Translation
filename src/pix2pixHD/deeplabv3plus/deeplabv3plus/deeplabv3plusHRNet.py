# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.deeplabv3plus.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from model.deeplabv3plus.backbone import build_backbone
from model.deeplabv3plus.ASPP import ASPP
from model.hrnet import get_hrnet_basebone


class deeplabv3plusHR(nn.Module):
    def __init__(self, cfg, pretrain=False):
        super().__init__()
        self.backbone = None
        self.backbone_layers = None
        input_channel = 720
        self.aspp = ASPP(dim_in=input_channel,
                         dim_out=cfg.MODEL_ASPP_OUTDIM,
                         rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                         bn_mom=cfg.TRAIN_BN_MOM)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.cls_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg.MODEL_ASPP_OUTDIM,
                out_channels=cfg.MODEL_ASPP_OUTDIM,
                kernel_size=1,
                stride=1,
                padding=0),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=cfg.MODEL_ASPP_OUTDIM,
                out_channels=cfg.MODEL_NUM_CLASSES,
                kernel_size=1,
                stride=1,
                padding=0)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = get_hrnet_basebone(cfg.MODEL_NUM_CLASSES, pretrain=pretrain)

    def forward(self, x):
        x = self.backbone(x)
        aspp = self.aspp(x)
        aspp = self.dropout1(aspp)
        result = self.cls_conv(aspp)
        result = self.upsample4(result)
        return result


class deeplabv3plusHRMultiASPP(nn.Module):
    """
    [48, 96, 192, 384]
    """

    def __init__(self, cfg, pretrain=False):
        super().__init__()
        self.backbone = None
        self.backbone_layers = None
        input_channel = [48, 96, 192, 384]
        self.aspp1 = ASPP(dim_in=input_channel[0],
                          dim_out=cfg.MODEL_ASPP_OUTDIM,
                          rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                          bn_mom=cfg.TRAIN_BN_MOM)
        self.aspp2 = ASPP(dim_in=input_channel[1],
                          dim_out=cfg.MODEL_ASPP_OUTDIM,
                          rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                          bn_mom=cfg.TRAIN_BN_MOM)
        self.aspp3 = ASPP(dim_in=input_channel[2],
                          dim_out=cfg.MODEL_ASPP_OUTDIM,
                          rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                          bn_mom=cfg.TRAIN_BN_MOM)
        self.aspp4 = ASPP(dim_in=input_channel[3],
                          dim_out=cfg.MODEL_ASPP_OUTDIM,
                          rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                          bn_mom=cfg.TRAIN_BN_MOM)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.cls_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg.MODEL_ASPP_OUTDIM,
                out_channels=cfg.MODEL_ASPP_OUTDIM,
                kernel_size=1,
                stride=1,
                padding=0),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=cfg.MODEL_ASPP_OUTDIM,
                out_channels=cfg.MODEL_NUM_CLASSES,
                kernel_size=1,
                stride=1,
                padding=0)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = get_hrnet_basebone(cfg.MODEL_NUM_CLASSES, pretrain=pretrain, combine=False)

    def forward(self, x):
        x = self.backbone(x)
        x0_w, x0_h = x[0].size(2), x[0].size(3)
        aspp1 = self.aspp1(x[0])
        aspp2 = self.aspp2(x[1])
        aspp3 = self.aspp3(x[2])
        aspp4 = self.aspp4(x[3])
        aspp1 = F.interpolate(aspp1, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        aspp2 = F.interpolate(aspp2, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        aspp3 = F.interpolate(aspp3, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        aspp4 = F.interpolate(aspp4, size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        aspp = aspp1 + aspp2 + aspp3 + aspp4
        aspp = self.dropout1(aspp)
        result = self.cls_conv(aspp)
        result = self.upsample4(result)
        return result


class deeplabv3plusHRMultiASPPSELoss(nn.Module):
    """
    [48, 96, 192, 384]
    """

    def __init__(self, cfg, pretrain=False):
        super().__init__()
        from model.deeplabv3plus.deeplabv3plusEnc import Encoding, Mean
        self.backbone = None
        self.backbone_layers = None
        input_channel = [48, 96, 192, 384]
        self.aspp1 = ASPP(dim_in=input_channel[0],
                          dim_out=cfg.MODEL_ASPP_OUTDIM,
                          rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                          bn_mom=cfg.TRAIN_BN_MOM)
        self.aspp2 = ASPP(dim_in=input_channel[1],
                          dim_out=cfg.MODEL_ASPP_OUTDIM,
                          rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                          bn_mom=cfg.TRAIN_BN_MOM)
        self.aspp3 = ASPP(dim_in=input_channel[2],
                          dim_out=cfg.MODEL_ASPP_OUTDIM,
                          rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                          bn_mom=cfg.TRAIN_BN_MOM)
        self.aspp4 = ASPP(dim_in=input_channel[3],
                          dim_out=cfg.MODEL_ASPP_OUTDIM,
                          rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                          bn_mom=cfg.TRAIN_BN_MOM)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)

        se_channel = input_channel[-1] // 4
        self.se_loss = nn.Sequential(
            nn.Conv2d(input_channel[-1], se_channel, 1, bias=False),
            SynchronizedBatchNorm2d(se_channel, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(True),
            Encoding(D=se_channel, K=32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            Mean(dim=1),
            nn.Linear(se_channel, cfg.MODEL_NUM_CLASSES)
        )

        self.cls_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg.MODEL_ASPP_OUTDIM,
                out_channels=cfg.MODEL_ASPP_OUTDIM,
                kernel_size=1,
                stride=1,
                padding=0),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=cfg.MODEL_ASPP_OUTDIM,
                out_channels=cfg.MODEL_NUM_CLASSES,
                kernel_size=1,
                stride=1,
                padding=0)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = get_hrnet_basebone(cfg.MODEL_NUM_CLASSES, pretrain=pretrain, combine=False)

    def forward(self, x):
        x = self.backbone(x)
        x0_w, x0_h = x[0].size(2), x[0].size(3)
        aspp1 = self.aspp1(x[0])
        aspp2 = self.aspp2(x[1])
        aspp3 = self.aspp3(x[2])
        aspp4 = self.aspp4(x[3])
        aspp1 = F.interpolate(aspp1, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        aspp2 = F.interpolate(aspp2, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        aspp3 = F.interpolate(aspp3, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        aspp4 = F.interpolate(aspp4, size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        aspp = aspp1 + aspp2 + aspp3 + aspp4
        aspp = self.dropout1(aspp)
        result = self.cls_conv(aspp)
        result = self.upsample4(result)

        if self.training:
            return result, self.se_loss(x[-1])
        else:
            return result


from model.deeplabv3plus.deeplabv3plusPSPASPP import PSPASPP


class deeplabv3plusHRPSPASPPthick(nn.Module):
    def __init__(self, cfg, pretrain=False):
        super().__init__()
        self.backbone = None
        self.backbone_layers = None
        input_channel = 720
        cfg.MODEL_ASPP_OUTDIM = 512
        self.pspaspp = PSPASPP(in_channels=input_channel,
                               out_channels=cfg.MODEL_ASPP_OUTDIM,
                               rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                               norm_layer=SynchronizedBatchNorm2d,
                               norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})
        self.dropout1 = nn.Dropout(0.5)

        self.cls_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg.MODEL_ASPP_OUTDIM,
                out_channels=cfg.MODEL_ASPP_OUTDIM,
                kernel_size=1,
                stride=1,
                padding=0),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=cfg.MODEL_ASPP_OUTDIM,
                out_channels=cfg.MODEL_NUM_CLASSES,
                kernel_size=1,
                stride=1,
                padding=0)
        )

        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = get_hrnet_basebone(cfg.MODEL_NUM_CLASSES, pretrain=pretrain)

    def forward(self, x):
        x = self.backbone(x)
        aspp = self.pspaspp(x)
        aspp = self.dropout1(aspp)
        result = self.cls_conv(aspp)
        result = self.upsample4(result)
        return result


class deeplabv3plusHRMultiPSPASPPthickseloss(nn.Module):

    """
    [48, 96, 192, 384]
    """

    def __init__(self, cfg, pretrain=False):
        super().__init__()
        from model.deeplabv3plus.deeplabv3plusEnc import Encoding, Mean
        self.backbone = None
        self.backbone_layers = None
        input_channel = [48, 96, 192, 384]
        cfg.MODEL_ASPP_OUTDIM = 512
        self.pspaspp1 = PSPASPP(in_channels=input_channel[0],
                                out_channels=cfg.MODEL_ASPP_OUTDIM,
                                rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                                norm_layer=SynchronizedBatchNorm2d,
                                norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})
        self.pspaspp2 = PSPASPP(in_channels=input_channel[1],
                                out_channels=cfg.MODEL_ASPP_OUTDIM,
                                rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                                norm_layer=SynchronizedBatchNorm2d,
                                norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})
        self.pspaspp3 = PSPASPP(in_channels=input_channel[2],
                                out_channels=cfg.MODEL_ASPP_OUTDIM,
                                rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                                norm_layer=SynchronizedBatchNorm2d,
                                norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})
        self.pspaspp4 = PSPASPP(in_channels=input_channel[3],
                                out_channels=cfg.MODEL_ASPP_OUTDIM,
                                rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                                norm_layer=SynchronizedBatchNorm2d,
                                norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)

        se_channel = input_channel[-1] // 4
        self.se_loss = nn.Sequential(
            nn.Conv2d(input_channel[-1], se_channel, 1, bias=False),
            SynchronizedBatchNorm2d(se_channel, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(True),
            Encoding(D=se_channel, K=32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            Mean(dim=1),
            nn.Linear(se_channel, cfg.MODEL_NUM_CLASSES)
        )

        self.cls_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg.MODEL_ASPP_OUTDIM,
                out_channels=cfg.MODEL_ASPP_OUTDIM,
                kernel_size=1,
                stride=1,
                padding=0),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=cfg.MODEL_ASPP_OUTDIM,
                out_channels=cfg.MODEL_NUM_CLASSES,
                kernel_size=1,
                stride=1,
                padding=0)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = get_hrnet_basebone(cfg.MODEL_NUM_CLASSES, pretrain=pretrain, combine=False)

    def forward(self, x):
        x = self.backbone(x)
        x0_w, x0_h = x[0].size(2), x[0].size(3)
        pspaspp1 = self.pspaspp1(x[0])
        pspaspp2 = self.pspaspp2(x[1])
        pspaspp3 = self.pspaspp3(x[2])
        pspaspp4 = self.pspaspp4(x[3])
        pspaspp1 = F.interpolate(pspaspp1, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        pspaspp2 = F.interpolate(pspaspp2, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        pspaspp3 = F.interpolate(pspaspp3, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        pspaspp4 = F.interpolate(pspaspp4, size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        pspaspp = pspaspp1 + pspaspp2 + pspaspp3 + pspaspp4
        pspaspp = self.dropout1(pspaspp)
        result = self.cls_conv(pspaspp)
        result = self.upsample4(result)
        if self.training:
            return result, self.se_loss(x[-1])
        else:
            return result


class deeplabv3plusHRMultiPSPASPPthick(nn.Module):

    """
    [48, 96, 192, 384]
    """

    def __init__(self, cfg, pretrain=False):
        super().__init__()
        self.backbone = None
        self.backbone_layers = None
        input_channel = [48, 96, 192, 384]
        cfg.MODEL_ASPP_OUTDIM = 512
        self.pspaspp1 = PSPASPP(in_channels=input_channel[0],
                                out_channels=cfg.MODEL_ASPP_OUTDIM,
                                rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                                norm_layer=SynchronizedBatchNorm2d,
                                norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})
        self.pspaspp2 = PSPASPP(in_channels=input_channel[1],
                                out_channels=cfg.MODEL_ASPP_OUTDIM,
                                rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                                norm_layer=SynchronizedBatchNorm2d,
                                norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})
        self.pspaspp3 = PSPASPP(in_channels=input_channel[2],
                                out_channels=cfg.MODEL_ASPP_OUTDIM,
                                rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                                norm_layer=SynchronizedBatchNorm2d,
                                norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})
        self.pspaspp4 = PSPASPP(in_channels=input_channel[3],
                                out_channels=cfg.MODEL_ASPP_OUTDIM,
                                rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                                norm_layer=SynchronizedBatchNorm2d,
                                norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.cls_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg.MODEL_ASPP_OUTDIM,
                out_channels=cfg.MODEL_ASPP_OUTDIM,
                kernel_size=1,
                stride=1,
                padding=0),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=cfg.MODEL_ASPP_OUTDIM,
                out_channels=cfg.MODEL_NUM_CLASSES,
                kernel_size=1,
                stride=1,
                padding=0)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = get_hrnet_basebone(cfg.MODEL_NUM_CLASSES, pretrain=pretrain, combine=False)

    def forward(self, x):
        x = self.backbone(x)
        x0_w, x0_h = x[0].size(2), x[0].size(3)
        pspaspp1 = self.pspaspp1(x[0])
        pspaspp2 = self.pspaspp2(x[1])
        pspaspp3 = self.pspaspp3(x[2])
        pspaspp4 = self.pspaspp4(x[3])
        pspaspp1 = F.interpolate(pspaspp1, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        pspaspp2 = F.interpolate(pspaspp2, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        pspaspp3 = F.interpolate(pspaspp3, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        pspaspp4 = F.interpolate(pspaspp4, size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        pspaspp = pspaspp1 + pspaspp2 + pspaspp3 + pspaspp4
        pspaspp = self.dropout1(pspaspp)
        result = self.cls_conv(pspaspp)
        result = self.upsample4(result)
        return result


# Not Implemented yet
class deeplabv3plusHREncPSPASPPthick(nn.Module):
    def __init__(self, cfg, pretrain=False):
        from model.deeplabv3plus.deeplabv3plusEnc import _EncHead

        super().__init__()
        self.backbone = None
        self.backbone_layers = None
        input_channel = 720
        cfg.MODEL_ASPP_OUTDIM = 512

        self.head = _EncHead(
            in_channels=input_channel, out_channels=input_channel, nclass=cfg.MODEL_NUM_CLASSES,
            se_loss=True, lateral=False,
            norm_layer=SynchronizedBatchNorm2d, norm_kwargs={'momentum': cfg.TRAIN_BN_MOM},
        )

        self.pspaspp = PSPASPP(in_channels=input_channel,
                               out_channels=cfg.MODEL_ASPP_OUTDIM,
                               rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                               norm_layer=SynchronizedBatchNorm2d,
                               norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})

        self.dropout1 = nn.Dropout(0.5)

        self.cls_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg.MODEL_ASPP_OUTDIM,
                out_channels=cfg.MODEL_ASPP_OUTDIM,
                kernel_size=1,
                stride=1,
                padding=0),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=cfg.MODEL_ASPP_OUTDIM,
                out_channels=cfg.MODEL_NUM_CLASSES,
                kernel_size=1,
                stride=1,
                padding=0)
        )

        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = get_hrnet_basebone(cfg.MODEL_NUM_CLASSES, pretrain=pretrain)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = list(x)
        aspp = self.pspaspp(x[0])
        aspp = self.dropout1(aspp)
        result = self.cls_conv(aspp)
        result = self.upsample4(result)
        if self.training:
            return result, x[1]
        else:
            return result


# Not Implemented yet
class deeplabv3plusHRPSPASPPthickSEloss(nn.Module):
    def __init__(self, cfg, pretrain=False):
        from model.deeplabv3plus.deeplabv3plusEnc import Encoding, Mean

        super().__init__()
        self.backbone = None
        self.backbone_layers = None

        input_channel = 720
        cfg.MODEL_ASPP_OUTDIM = 512
        se_channel = 48

        self.pspaspp = PSPASPP(in_channels=input_channel,
                               out_channels=cfg.MODEL_ASPP_OUTDIM,
                               rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                               norm_layer=SynchronizedBatchNorm2d,
                               norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})
        self.dropout1 = nn.Dropout(0.5)

        self.se_loss = nn.Sequential(
            # nn.AvgPool2d(kernel_size=7, stride=4, padding=3),
            nn.Conv2d(input_channel, se_channel, 1, bias=False),
            SynchronizedBatchNorm2d(se_channel, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(True),
            Encoding(D=se_channel, K=32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            Mean(dim=1),
            nn.Linear(se_channel, cfg.MODEL_NUM_CLASSES)
        )

        self.cls_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg.MODEL_ASPP_OUTDIM,
                out_channels=cfg.MODEL_ASPP_OUTDIM,
                kernel_size=1,
                stride=1,
                padding=0),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=cfg.MODEL_ASPP_OUTDIM,
                out_channels=cfg.MODEL_NUM_CLASSES,
                kernel_size=1,
                stride=1,
                padding=0)
        )

        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = get_hrnet_basebone(cfg.MODEL_NUM_CLASSES, pretrain=pretrain, combine=False)

    def forward(self, x):
        x = self.backbone(x)
        aspp = self.pspaspp(x)
        aspp = self.dropout1(aspp)
        result = self.cls_conv(aspp)
        result = self.upsample4(result)

        if self.training:
            return result, self.se_loss(x)
        else:
            return result


if __name__ == '__main__':
    from model.deeplabv3plus import Configuration

    cfg = Configuration()
    # model = deeplabv3plusHRPSPASPPthick(cfg, pretrain=True)
    # model = deeplabv3plusHREncPSPASPPthick(cfg, pretrain=False)
    # model = deeplabv3plusHRPSPASPPthickSEloss(cfg, pretrain=False)
    # model = deeplabv3plusHRMultiASPP(cfg, pretrain=False)
    # model = deeplabv3plusHRMultiPSPASPPthick(cfg, pretrain=False)
    model = deeplabv3plusHRMultiASPPSELoss(cfg, pretrain=False)
    # model.eval()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    for i in y:
        print(i.shape)
    # print(model)
    pass
