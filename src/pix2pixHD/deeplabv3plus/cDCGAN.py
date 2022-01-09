import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class DCGAN(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.G = Generator(d)
        self.D = Discriminator(d)

    def G_forward(self, input, label):
        return self.G(input, label)

    def D_forward(self, input, label):
        return self.D(input, label)


class Generator(nn.Module):
    # initializers
    def __init__(self, d=128, c=3):
        super().__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d * 2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d * 2)
        self.deconv1_2 = nn.ConvTranspose2d(10, d * 2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d * 2)
        self.deconv2 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 2)
        self.deconv3 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, c, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        return x


class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128, n_class=9, in_channel=12):
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channel, d // 2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(n_class, d // 2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = torch.sigmoid(self.conv4(x))

        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


if __name__ == '__main__':
    b, c, h, w = 2, 3, 4, 4
    label = torch.randint(low=0, high=10, size=(b, 9, 128, 128)).float()
    # x = torch.rand(b, 100, 32, 32)
    x = torch.rand(b, 12, 128, 128)
    model = Discriminator()
    # model = Generator()
    d_out = model(x, label)
    print(d_out.shape)
    pass
