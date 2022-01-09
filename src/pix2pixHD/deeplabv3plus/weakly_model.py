import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.erfnet import DownsamplerBlock, non_bottleneck_1d
import models.erfnet as erfnet
from models.deeplabv3plus.deeplabv3plusselflearning import SelfLearningBase


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(
            128, num_classes, 1, stride=1, padding=0, bias=True)

        self.bbox_attention = nn.Sequential(
            nn.Conv2d(in_channels=num_classes, out_channels=128, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor, bbox=None, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        b, c, h, w = output.shape
        if bbox is not None:
            bbox = F.interpolate(bbox, size=(h, w), mode='bilinear', align_corners=True)
            bbox_attention = self.bbox_attention(bbox)
            output = output * bbox_attention

        if predict:
            output = self.output_conv(output)

        return output


class LabelEncoder(Encoder):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        del self.bbox_attention
        self.label_attention = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=128, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor, bbox=None, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        b, c, h, w = output.shape
        if bbox is None:
            labels = torch.zeros(b, 9, 1, 1).to(output.device)
        else:
            labels = bbox.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0].float()

        label_attention = self.label_attention(labels)

        output = output * label_attention

        if predict:
            output = self.output_conv(output)

        return output


class BranchedERFNet(nn.Module):
    def __init__(self, num_classes, encoder=None):
        super().__init__()

        print('Creating branched erfnet with {} classes'.format(num_classes))

        if encoder is None:
            self.encoder = Encoder(sum(num_classes))
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

    def forward(self, input, bbox=None, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, bbox, predict=True)
        else:
            output = self.encoder(input, bbox)

        return torch.cat([decoder.forward(output) for decoder in self.decoders], 1)


class ERFNet3branch(BranchedERFNet):
    def __init__(self, num_classes, encoder=None):
        super().__init__(num_classes, encoder)
        self.cls_conv = erfnet.Decoder(num_classes[1] + 1)
        self.seg = False

    def forward(self, input, bbox=None, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, bbox, predict=True)
        else:
            output = self.encoder(input, bbox)

        cls = self.cls_conv(output)

        if self.seg:
            return torch.cat([decoder.forward(output) for decoder in self.decoders], 1), cls
        else:
            return torch.cat([decoder.forward(output) for decoder in self.decoders], 1)


class LabelERFNet3branch(BranchedERFNet):
    def __init__(self, num_classes):
        super().__init__(num_classes, encoder=LabelEncoder(num_classes=sum(num_classes)))
        self.cls_conv = erfnet.Decoder(num_classes[1] + 1)
        self.seg = False

    def forward(self, input, bbox=None, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, bbox, predict=True)
        else:
            output = self.encoder(input, bbox)

        cls = self.cls_conv(output)

        if self.seg:
            return torch.cat([decoder.forward(output) for decoder in self.decoders], 1), cls
        else:
            return torch.cat([decoder.forward(output) for decoder in self.decoders], 1)


class ERFNetBBox(SelfLearningBase):
    def __init__(self, num_classes, encoder=None):  # use encoder to pass pretrained encoder
        super().__init__()
        if encoder is None:
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = erfnet.Decoder(num_classes)

    def forward(self, input: torch.Tensor, bbox=None, only_encode=False):
        if only_encode:
            result = self.encoder.forward(input, bbox, predict=True)
        else:
            output = self.encoder.forward(input, bbox, predict=False)  # predict=False by default
            result = self.decoder.forward(output)

        if self.mode == 'refine':
            bbox[:, 0] = 1.
            result = bbox * result
            return result
        elif self.mode == 'self_learning':
            probs = result.softmax(dim=1)
            bbox[:, 0] = 1.
            probs = bbox * probs
            return probs
        elif self.mode == 'refine_learning':
            bbox[:, 0] = 1.
            result = bbox * result
            probs = result.softmax(dim=1)
            return probs
        elif self.mode == 'normal':
            assert bbox is None
            return result
        elif self.mode == 'info':
            assert bbox is not None
            return result
        else:
            raise NotImplementedError


if __name__ == '__main__':
    model = ERFNetBBox(num_classes=21)
    x = torch.rand(2, 3, 32, 32)
    y = model(x, bbox=torch.zeros(2, 21, 32, 32))
    print(y.shape)
    y_ = model(x)
    print(y_.shape)
    pass
