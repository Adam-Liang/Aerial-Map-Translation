import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from src.utils.train_utils import get_device, model_accelerate


###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_G(args, input_nc=None):
    if input_nc is None:
        #input_nc = args.label_nc
        input_nc = args.input_nc
        if args.use_instance:
            input_nc += 1
        if args.feat_num > 0:
            input_nc += args.feat_num

    norm_layer = get_norm_layer(norm_type=args.norm)
    if args.netG == 'global':
        netG = GlobalGenerator(input_nc, args.output_nc, args.ngf, args.n_downsample_global, args.n_blocks_global,
                               norm_layer)
    elif args.netG == 'local':
        netG = LocalEnhancer(input_nc, args.output_nc, args.ngf, args.n_downsample_global, args.n_blocks_global,
                             args.n_local_enhancers, args.n_blocks_local, norm_layer)
    elif args.netG == 'encoder':
        netG = Encoder(input_nc, args.output_nc, args.ngf, args.n_downsample_global, norm_layer)
    else:
        raise ('generator not implemented!')
    print(netG)
    netG.apply(weights_init)
    netG = nn.DataParallel(netG).to(get_device(args))
    return netG


def get_E(args):
    norm_layer = get_norm_layer(norm_type=args.norm)
    netE = Encoder(args.output_nc, args.feat_num, args.ngf, args.n_downsample_global, norm_layer)
    netE.apply(weights_init)
    print(netE)
    netE = nn.DataParallel(netE).to(get_device(args))
    return netE


def get_D(args, input_nc=None):
    if input_nc is None:
        #input_nc = args.label_nc + args.output_nc
        input_nc = args.input_nc + args.output_nc
        if args.use_instance:
            input_nc += 1

    norm_layer = get_norm_layer(norm_type=args.norm)
    netD = MultiscaleDiscriminator(input_nc, args.ndf, args.n_layers_D, norm_layer, args.use_lsgan, args.num_D,
                                   args.use_ganFeat_loss)
    print(netD)
    netD.apply(weights_init)
    netD = nn.DataParallel(netD).to(get_device(args))
    return netD


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)



##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####           
        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global_1 = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer).model_1
        model_global_2 = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                         norm_layer).model_2
        model_global_2 = [model_global_2[i] for i in
                        range(len(model_global_2) - 3)]  # get rid of final convolution layers
        # self.model = nn.Sequential(*model_global)
        self.model_1 = nn.Sequential(*model_global_1)
        self.model_2 = nn.Sequential(*model_global_2)
        self.pre2= nn.Sequential(nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=1, padding=0),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))
        self.mix2 = nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global* 2), nn.ReLU(True),
                                  nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global * 2), nn.ReLU(True),
                                  nn.Conv2d(ngf_global * 2, ngf_global * 2, kernel_size=3, padding=1),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample            
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [
                nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))#为对象self添加属性
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.pre3= nn.Sequential(nn.Conv2d(ngf_global * 4, ngf_global*2, kernel_size=1, padding=0),
                                  norm_layer(ngf_global* 2), nn.ReLU(True))
        self.mix3=nn.Sequential(nn.Conv2d(ngf_global*4, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True),
                                nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True),
                                nn.Conv2d(ngf_global*2, ngf_global*2, kernel_size=3, padding=1),
                                norm_layer(ngf_global*2), nn.ReLU(True))

    def forward(self, input,input2=None,input3=None):  #input2: 小尺寸的高层特征   input3:大尺寸的低层特征
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        # output_prev = self.model(input_downsampled[-1])
        output_prev= self.model_1(input_downsampled[-1])
        # if not (input2 is None):
        #     input2=self.pre2(input2)
        #     output_prev=self.mix2(torch.cat((output_prev,input2),dim=1))
        output_prev=self.model_2(output_prev)
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            tmp=model_downsample(input_i) + output_prev
            # if not (input3 is None):
            #     input3=self.pre3(input3)
            #     tmp=self.mix3(torch.cat((tmp,input3),dim=1))
            output_prev = model_upsample(tmp)
            # output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True) #将会改变输入的原数据

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model_1 = nn.Sequential(*model[:7])
        self.model_2 = nn.Sequential(*model[7:])

        self.mix2=nn.Sequential(nn.Conv2d(ngf*4, ngf*2, kernel_size=3, padding=1),
                                norm_layer(ngf*2), nn.ReLU(True),
                                nn.Conv2d(ngf*2, ngf*2, kernel_size=3, padding=1),
                                norm_layer(ngf*2), nn.ReLU(True),
                                nn.Conv2d(ngf*2, ngf*2, kernel_size=3, padding=1),
                                norm_layer(ngf*2), nn.ReLU(True))

    def forward(self, input ,input2=None):
        out=self.model_1(input)
        if input2!=None:
            out=self.mix2(torch.cat((out,input2),dim=1))
        out=self.model_2(out)
        return out

    # Define a resnet block


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        self.output_nc = output_nc

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf), nn.ReLU(True)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b + 1] == int(i)).nonzero()  # n x 4
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]] = mean_feat
        return outputs_mean


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


def test_model(model_func=Encoder):
    model = model_func(21, 3)
    x = torch.rand(3, 21, 32, 32)
    y = model(x)
    # print(y)
    try:
        print(y.shape)
    except:
        pass

    try:
        print(model)
    except:
        pass

    try:
        for i in y:
            for j in i:
                print(j.shape)
    except:
        pass


if __name__ == '__main__':
    from src.pix2pixHD.train_config import config

    args = config()
    # model = get_G(args, 21)
    model = get_D(args, 21)
    model = get_G(args, 21)
    x = torch.rand(3, 21, 32, 32)
    y = model(x)
    # print(y)
    try:
        print(y.shape)
    except:
        pass

    try:
        print(model)
    except:
        pass

    try:
        for i in y:
            for j in i:
                print(j.shape)
    except:
        pass
