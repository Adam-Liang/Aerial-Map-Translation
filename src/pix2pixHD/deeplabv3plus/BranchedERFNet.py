"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import models.erfnet as erfnet


class BranchedERFNet(nn.Module):
    def __init__(self, num_classes, encoder=None):
        super().__init__()
        self.num_classes = num_classes

        print('Creating branched erfnet with {} classes'.format(num_classes))

        if encoder is None:
            self.encoder = erfnet.Encoder(sum(num_classes))
        else:
            self.encoder = encoder

        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(erfnet.Decoder(n))

    def init_output(self, n_sigma=2):
        with torch.no_grad():
            output_conv = self.decoders[0].output_conv
            print('initialize last layer with size: ',
                  output_conv.weight.size())

            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2:2 + n_sigma, :, :].fill_(0)
            output_conv.bias[2:2 + n_sigma].fill_(1)
            if self.num_classes[0] == 6:
                output_conv.weight[:, 2 + n_sigma:4 + n_sigma, :, :].fill_(0)
                output_conv.bias[2 + n_sigma:4 + n_sigma].fill_(0)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)

        return torch.cat([decoder.forward(output) for decoder in self.decoders], 1)


class ERFNet3branch(BranchedERFNet):
    def __init__(self, num_classes, encoder=None):
        super().__init__(num_classes, encoder)
        self.cls_conv = erfnet.Decoder(num_classes[1] + 1)
        self.seg = False
        self.init_output()

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)
        cls = self.cls_conv(output)

        if self.seg:
            return torch.cat([decoder.forward(output) for decoder in self.decoders], 1), cls
        else:
            return torch.cat([decoder.forward(output) for decoder in self.decoders], 1)


if __name__ == '__main__':
    model = ERFNet3branch(num_classes=[3, 1])
    x = torch.rand(2, 3, 32, 32)
    y = model(x)
    for key in model.state_dict():
        print(key)
    for i in y:
        print(i.shape)
