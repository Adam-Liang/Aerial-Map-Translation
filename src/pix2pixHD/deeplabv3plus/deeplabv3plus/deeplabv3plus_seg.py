"""
deeplabv3+ only for Segmentation
"""

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
        self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()

    def forward(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers() #长度为3的list，比较前面的两个结果和最后的一个结果
        # for l in layers:
        #     print(l.shape)
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)
        # print(feature_aspp.shape)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        feature_map = self.cat_conv(feature_cat)
        result = self.cls_conv(feature_map)
        result = self.upsample4(result)
        return result,feature_map

    def get_paras(self):
        backbone_params=self.backbone.parameters()
        base_params = list(map(id, self.backbone.parameters())) # 注意此处不能使用backbone_params，会导致该迭代器被疯狂使用，后面无法通过正确性检验
        global_params = filter(lambda p: id(p) not in base_params, self.parameters())
        # num_bb=sum(1 for _ in backbone_params) # 分割正确性检验
        # num_gl=sum(1 for _ in global_params)
        # num_all=sum(1 for _ in self.parameters())
        return global_params,backbone_params

    def pretrain(self,ptpath):
        pt_dict = torch.load(ptpath)
        model_dict = self.state_dict()
        import collections
        new_pt_dict=collections.OrderedDict()

        for name,params in pt_dict.items():
            names = name.split('.')
            if names[0]=='_aspp':
                new_name='aspp.'
                if names[1]=='_branches':
                    new_name+='branch'
                    tmpdit={'0.1':'1.0.','0.2':'1.1.','1.0':'2.0.','1.1':'2.1.','2.0':'3.0.','3.0':'4.0.','4.0':'5_'}
                    new_name+=tmpdit[names[2]+'.'+names[3]]
                    for i in range(4,len(names)):
                        new_name+=names[i]
                        if i <len(names)-1:
                            new_name+='.'
                elif names[1]=='_conv_concat':
                    new_name+='conv_cat.'
                    new_name+=names[2]
                    new_name += names[3]
            elif names[0]=='_feature_extractor':
                new_name = 'backbone.'
                if names[2]=='0' or names[2]=='1':
                    if names[3]=='_batch_norm':
                        new_name+='bn'
                    elif names[3]=='_conv':
                        new_name+='conv'
                    new_name+=str(int(names[2])+1)
                    new_name+='.'
                    new_name+=names[4]
                elif names[2]=='2': #很多block
                    if int(names[4])<=19:
                        new_name+='block'
                        new_name+=str(int(names[4])+1)
                        new_name += '.'
                        if names[5]=='_separable_conv_block':
                            new_name+='sepconv'
                            tmp={'1':'1','3':'2','5':'3'}
                            new_name+=tmp[names[6]]
                            new_name+='.'
                            tmp={'_conv_depthwise':'depthwise','_batch_norm_depthwise':'bn1','_conv_pointwise':'pointwise','_batch_norm_pointwise':'bn2'}
                            new_name+=tmp[names[7]]
                            new_name+='.'
                            # if names[8]=='num_batches_tr':
                            #     names[8]='num_batches_tracked'
                            new_name+=names[8]
                        elif names[5]=='_conv_skip_connection' or names[5]=='_batch_norm_shortcut':
                            tmp={'_conv_skip_connection':'skip','_batch_norm_shortcut':'skipbn'}
                            new_name+=tmp[names[5]]
                            new_name+='.'
                            new_name+=names[6]
                    elif int(names[4])==20:
                        new_name+='conv'
                        new_name+=str(int(names[6])+3)
                        new_name+='.'
                        tmp = {'_conv_depthwise':'depthwise','_batch_norm_depthwise':'bn1','_conv_pointwise':'pointwise','_batch_norm_pointwise':'bn2'}
                        new_name+=tmp[names[7]]
                        new_name+='.'
                        new_name+=names[8]
            elif names[0]=='_refine_decoder':
                if names[1]=='_decoder':
                    new_name='shortcut_conv.'
                    new_name+=names[2]
                    new_name+='.'
                    new_name+=names[3]
                elif names[1]=='_concat_layers':
                    new_name=name # 该部分结构不匹配，因此不改动名称
            elif names[0]=='_logits_layer':
                new_name = name  # 最终预测部分label数不一样，因此不加载
            new_pt_dict[new_name]=params
        not_dict = {k: v for k, v in new_pt_dict.items() if (k not in model_dict)}
        nothave_dict={k: v for k, v in model_dict.items() if (k not in new_pt_dict)}
        new_pt_dict = {k: v for k, v in new_pt_dict.items() if ( k in model_dict)}
        new_pt_dict.pop('aspp.branch2.0.weight')
        model_dict.update(new_pt_dict)
        self.load_state_dict(model_dict)

        pass

def get_params(model, key):
    print('????')
    for m in model.named_modules():
        print(m)
        if key == '1x':
            if (any([(i in m[0]) for i in ('pretrained_net', 'encoder', 'backbone')])) and isinstance(m[1],
                                                                                                      nn.Conv2d):
                print(m[0])
                for p in m[1].parameters():
                    yield p
        elif key == '10x':
            if (not any([(i in m[0]) for i in ('pretrained_net', 'encoder', 'backbone')])) and isinstance(m[1],
                                                                                                          nn.Conv2d):
                for p in m[1].parameters():
                    yield p


if __name__ == '__main__':
    pass
    from models.deeplabv3plus import Configuration

    cfg = Configuration()
    model = deeplabv3plus(cfg)
    # get_params(model=model, key='1x')
    key = '10x'
    for m in model.named_modules():
        if key == '1x':
            if (any([(i in m[0]) for i in ('pretrained_net', 'encoder', 'backbone')])) and isinstance(m[1],
                                                                                                      nn.Conv2d):
                print(m[0])
                for p in m[1].parameters():
                    pass
        elif key == '10x':
            if (not any([(i in m[0]) for i in ('pretrained_net', 'encoder', 'backbone')])) and isinstance(m[1],
                                                                                                          nn.Conv2d):
                print(m[0])
                for p in m[1].parameters():
                    pass