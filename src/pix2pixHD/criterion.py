import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from src.utils.train_utils import get_device, model_accelerate


##############################################################################
# Losses
##############################################################################
def get_GANLoss(args):
    # return nn.DataParallel(GANLoss(use_lsgan=args.use_lsgan))
    return GANLoss(use_lsgan=args.use_lsgan)


def get_VGGLoss(args):
    # return nn.DataParallel(VGGLoss(args))
    return VGGLoss(args)


def get_DFLoss(args=None):
    # return nn.DataParallel(DiscriminatorFeaturesLoss())
    return DiscriminatorFeaturesLoss()


def get_low_level_loss(args, low_level_loss=None):
    if low_level_loss is None:
        low_level_loss =args.low_level_loss
    if  low_level_loss== 'L1':
        L = nn.DataParallel(nn.L1Loss())
    elif low_level_loss== 'L2':
        L = nn.DataParallel(nn.MSELoss())
    elif low_level_loss == 'smoothL1':
        L = nn.DataParallel(nn.SmoothL1Loss())
    else:
        raise NotImplementedError
    return L


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.DataParallel(nn.MSELoss())
            # self.loss = nn.MSELoss()
        else:
            self.loss = nn.DataParallel(nn.BCELoss())
            # self.loss = nn.BCELoss()
        print(f'===> {self.__class__.__name__} | use_lsgan:{use_lsgan} | loss:{self.loss}')

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.shape).fill_(self.real_label)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.shape).fill_(self.fake_label)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class VGGLoss(nn.Module):
    def __init__(self, args):
        super(VGGLoss, self).__init__()
        assert args.vgg_type in ('vgg16', 'vgg19')
        vgg = Vgg16 if args.vgg_type == 'vgg16' else Vgg19
        self.vgg = nn.DataParallel(vgg()).to(get_device(args))
        # self.vgg = vgg().to(get_device(args))
        self.vgg.eval()
        self.criterion = nn.DataParallel(nn.L1Loss())
        # self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

        print(f'===> {self.__class__.__name__} | vgg:{args.vgg_type} | loss:{self.criterion}')

    def forward(self, x, y):
        with torch.no_grad():
            x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class DiscriminatorFeaturesLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.DataParallel(nn.L1Loss())
        # self.l1 = nn.L1Loss()

    def forward(self, ds_fake, ds_real):
        """
        :param ds_fake: [D1:[layer1_outs, layer2_outs ...], D2, D3]
        :param ds_real:
        :return:
        """
        loss = 0

        for scale in range(len(ds_real)):
            # last is D_outs, do not use as features
            for l in range(len(ds_real[scale]) - 1):
                loss += self.l1(ds_fake[scale][l], ds_real[scale][l].detach())
        return loss / float(len(ds_real))


from torchvision import models


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
