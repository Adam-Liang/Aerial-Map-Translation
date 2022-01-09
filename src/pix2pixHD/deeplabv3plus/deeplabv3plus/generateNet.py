# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
from model.deeplabv3plus.deeplabv3plus import deeplabv3plus


class Configuration():
    def __init__(self):
        self.MODEL_NAME = 'deeplabv3plus'
        self.MODEL_BACKBONE = 'xception'
        self.MODEL_OUTPUT_STRIDE = 16
        self.MODEL_ASPP_OUTDIM = 256
        self.MODEL_SHORTCUT_DIM = 48
        self.MODEL_SHORTCUT_KERNEL = 1
        self.MODEL_NUM_CLASSES = 21
        self.TRAIN_BN_MOM = 0.0003


def generate_net(cfg=None, use_default=True):
    if use_default:
        from model.deeplabv3plus.deeplab_config import Configuration
        cfg = Configuration()
    if cfg.MODEL_NAME == 'deeplabv3plus' or cfg.MODEL_NAME == 'deeplabv3+':
        return deeplabv3plus(cfg)
    # if cfg.MODEL_NAME == 'supernet' or cfg.MODEL_NAME == 'SuperNet':
    # 	return SuperNet(cfg)
    # if cfg.MODEL_NAME == 'eanet' or cfg.MODEL_NAME == 'EANet':
    # 	return EANet(cfg)
    # if cfg.MODEL_NAME == 'danet' or cfg.MODEL_NAME == 'DANet':
    # 	return DANet(cfg)
    # if cfg.MODEL_NAME == 'deeplabv3plushd' or cfg.MODEL_NAME == 'deeplabv3+hd':
    # 	return deeplabv3plushd(cfg)
    # if cfg.MODEL_NAME == 'danethd' or cfg.MODEL_NAME == 'DANethd':
    # 	return DANethd(cfg)
    else:
        raise ValueError('generateNet.py: network %s is not support yet' % cfg.MODEL_NAME)


if __name__ == '__main__':
    model = generate_net()
    print(model)
    x = torch.randn(4, 3, 31, 31)
    out = model(x)
    print(out.size())
