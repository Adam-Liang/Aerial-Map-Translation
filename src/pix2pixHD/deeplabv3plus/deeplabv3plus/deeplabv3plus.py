# ----------------------------------------
# Written by Linwei Chen
# ----------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.pix2pixHD.deeplabv3plus.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from src.pix2pixHD.deeplabv3plus.deeplabv3plus.backbone import build_backbone
from src.pix2pixHD.deeplabv3plus.deeplabv3plus.ASPP import ASPP


class deeplabv3plus(nn.Module):
    def __init__(self, cfg):
        super(deeplabv3plus, self).__init__()
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

        self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()
        self.init_output()
        self.apply(self.init_bn)

    def init_output(self, n_sigma=2):
        with torch.no_grad():
            output_conv = self.offset_conv.output_conv
            print('initialize last layer with size: ', output_conv.weight.size())

            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2:2 + n_sigma, :, :].fill_(0)
            output_conv.bias[2:2 + n_sigma].fill_(1)
            if self.cfg.MODEL_AUX_OUT == 6:
                output_conv.weight[:, 2 + n_sigma:4 + n_sigma, :, :].fill_(0)
                output_conv.bias[2 + n_sigma:4 + n_sigma].fill_(0)
                pass
            elif self.cfg.MODEL_AUX_OUT == 5:
                output_conv.weight[:, 2 + n_sigma:3 + n_sigma, :, :].fill_(0)
                output_conv.bias[2 + n_sigma:3 + n_sigma].fill_(0)
                pass

    @staticmethod
    def init_bn(m):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, SynchronizedBatchNorm2d):
            m.eps = 0.001
            m.momentum = 0.1

    @staticmethod
    def init_bn_for_seg(m):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, SynchronizedBatchNorm2d):
            m.eps = 0.001
            m.momentum = 0.1

    def apply_init_bn(self):
        self.apply(self.init_bn)

    def apply_init_bn_for_seg(self):
        self.apply(self.init_bn_for_seg)

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

        return offset,seed_map,feature
        # return torch.cat([offset, seed_map], dim=1)


class deeplabv3pluslabel(deeplabv3plus):
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


class deeplabv3plus3branch(deeplabv3plus):
    # TODO: multi-scales
    def __init__(self, cfg):
        super().__init__(cfg)
        # background
        self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES + 1, 1, 1, padding=0)
        self.seg = False

    def forward(self, x, bbox=None):
        _, _, ih, iw = x.shape
        x = F.interpolate(x, size=(ih - ih % 16, iw - iw % 16), mode='bilinear', align_corners=True)
        # print(x.shape)

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
        # print(cls.shape)
        # print(offset.shape)
        # print(seed_map.shape)

        # 可行性？
        offset = F.interpolate(offset, size=(ih, iw), mode='bilinear', align_corners=True)
        # y, x, sigma
        offset[:, 0] = offset[:, 0] * float(ih / (ih - ih % 16))
        offset[:, 1] = offset[:, 1] * float(iw / (iw - iw % 16))
        seed_map = F.interpolate(seed_map, size=(ih, iw), mode='bilinear', align_corners=True)
        cls = F.interpolate(cls, size=(ih, iw), mode='bilinear', align_corners=True)
        # print(cls.shape)
        inst_seg = torch.cat([offset, seed_map], dim=1)
        if self.seg:
            return inst_seg, cls
        else:
            return inst_seg


class deeplabv3plus3branchsegattention(deeplabv3plus3branch):
    def __init__(self, cfg):
        super().__init__(cfg)
        indim = 256
        input_channel = 2048
        self.seg_attention1 = nn.Sequential(
            nn.Conv2d(cfg.MODEL_NUM_CLASSES + 1, indim, 3, 1, padding=1, bias=True),
            nn.Sigmoid()
        )

        self.seg_attention2 = nn.Sequential(
            nn.Conv2d(cfg.MODEL_NUM_CLASSES + 1, input_channel, 3, 1, padding=1, bias=True),
            nn.Sigmoid()
        )
        pass

    def forward(self, x, bbox=None):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        _, _, h0, w0 = layers[0].shape
        _, _, h2, w2 = layers[2].shape
        b, _, _, _ = x.shape

        if bbox is None:
            feature_aspp = self.aspp(layers[-1])
            feature_aspp = self.dropout1(feature_aspp)
            feature_aspp = self.upsample_sub(feature_aspp)

            feature_shallow = self.shortcut_conv(layers[0])
            feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
            feature = self.cat_conv(feature_cat)
        else:
            bbox_1 = F.interpolate(bbox, size=(h0, w0), mode='bilinear', align_corners=True)
            bbox_2 = F.interpolate(bbox, size=(h2, w2), mode='bilinear', align_corners=True)

            feature_aspp = self.aspp(layers[-1] * self.bbox_attention2(bbox_2))
            feature_aspp = self.dropout1(feature_aspp)
            feature_aspp = self.upsample_sub(feature_aspp)

            feature_shallow = self.shortcut_conv(layers[0] * self.bbox_attention1(bbox_1))
            feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
            feature = self.cat_conv(feature_cat)

        cls = self.cls_conv(feature)
        cls = self.upsample4(cls)
        semantic_out = cls.softmax(dim=1).detach()

        seg_1 = F.interpolate(semantic_out, size=(h0, w0), mode='bilinear', align_corners=True)
        seg_2 = F.interpolate(semantic_out, size=(h2, w2), mode='bilinear', align_corners=True)

        feature_aspp_seg_att = self.aspp(layers[-1] * self.seg_attention2(seg_2))
        feature_aspp_seg_att = self.dropout1(feature_aspp_seg_att)
        feature_aspp_seg_att = self.upsample_sub(feature_aspp_seg_att)

        feature_shallow_seg_att = self.shortcut_conv(layers[0] * self.seg_attention1(seg_1))
        feature_cat_seg_att = torch.cat([feature_aspp_seg_att, feature_shallow_seg_att], 1)
        feature_seg_att = self.cat_conv(feature_cat_seg_att)
        offset = self.offset_conv(feature_seg_att)
        seed_map = self.seed_map_conv(feature_seg_att)

        if self.seg:
            return torch.cat([offset, seed_map], dim=1), cls
        else:
            return torch.cat([offset, seed_map], dim=1)


class deeplabv3plus_1(deeplabv3plus):
    def __init__(self, cfg):
        super().__init__(cfg)
        # background
        del self.seed_map_conv
        self.seed_map_conv = nn.Sequential(
            nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )

        self.seg = False

    def forward(self, x, bbox=None):
        return super().forward(x, bbox)


class deeplabv3plus3branch_1(deeplabv3plus3branch):
    def __init__(self, cfg):
        super().__init__(cfg)
        # background
        del self.seed_map_conv
        self.seed_map_conv = nn.Sequential(
            nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )

        self.seg = False

    def forward(self, x, bbox=None):
        return super().forward(x, bbox)


class deeplabv3plus3branch2(deeplabv3plus):
    def __init__(self, cfg):
        super().__init__(cfg)
        del self.offset_conv, self.seed_map_conv
        self.offset_conv = DecoderThick(cfg.MODEL_AUX_OUT)
        self.seed_map_conv = DecoderThick(cfg.MODEL_NUM_CLASSES)
        # background
        self.cls_conv = DecoderThick(cfg.MODEL_NUM_CLASSES + 1)
        self.seg = False
        self.apply_init_bn()

    def apply_init_bn(self):
        self.apply(self.init_bn)

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
        # cls = self.upsample4(cls)
        if self.seg:
            return torch.cat([offset, seed_map], dim=1), cls
        else:
            return torch.cat([offset, seed_map], dim=1)


# TODO:DIS
class deeplabv3plus3branch2DIS(deeplabv3plus3branch2):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.img_rec = nn.Sequential(
            nn.Conv2d(in_channels=cfg.MODEL_NUM_CLASSES + 1, out_channels=16, kernel_size=3,
                      padding=1, stride=1, bias=True),
            SynchronizedBatchNorm2d(16, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                      padding=1, stride=1, bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3,
                      padding=1, stride=1, bias=True),
        )
        self.apply(self.init_bn)
        self.rec_loss = nn.MSELoss()

    def forward(self, x, bbox=None):
        return super().forward(x, bbox)

    def reconstruct(self, img):
        pass


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
        self.output_conv = nn.ConvTranspose2d(
            128, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


class DecoderThick(nn.Module):
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
    model = deeplabv3plus3branch(cfg)
    print(model)
    print(model.__class__.__name__)
    model.eval()
    model.seg = True
    x = torch.randn(2, 3, 511, 511)
    y = model(x)
    for item in y:
        print(item.shape)
    # print(nn.ConvTranspose2d(8, out_channels=4, kernel_size=2, stride=2, padding=0, output_padding=0,
    #                          bias=True).weight.shape)
    # print(model)

    model = deeplabv3pluslabel(cfg)
    y = model(x)
    print(y.shape)
    pass
