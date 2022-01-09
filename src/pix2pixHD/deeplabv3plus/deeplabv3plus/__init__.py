class Configuration():
    def __init__(self):
        self.MODEL_NAME = 'deeplabv3plus'
        self.MODEL_BACKBONE = 'xception'
        self.MODEL_OUTPUT_STRIDE = 16
        self.MODEL_ASPP_OUTDIM = 256
        self.MODEL_SHORTCUT_DIM = 48
        self.MODEL_SHORTCUT_KERNEL = 1
        self.MODEL_NUM_CLASSES = 8
        self.MODEL_AUX_OUT = 4
        self.TRAIN_BN_MOM = 0.0003


def get_deeplabv3plus_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from src.pix2pixHD.deeplabv3plus.deeplabv3plus.deeplabv3plus import deeplabv3plus, deeplabv3plus3branch, deeplabv3plus3branch2, \
        deeplabv3pluslabel, deeplabv3plus3branch_1, deeplabv3plus_1, deeplabv3plus3branchsegattention
    from src.pix2pixHD.deeplabv3plus.deeplabv3plus.deeplabv3plus_seg import deeplabv3plus as deeplabv3plus_seg
    from src.pix2pixHD.deeplabv3plus.deeplabv3plus.deeplabv3plusselflearning import deeplabv3plus as deeplabv3plus_seg_selflearning
    from src.pix2pixHD.deeplabv3plus.deeplabv3plus.deeplabv3plus_multi_offset import deeplabv3plusmultioffset
    if name.lower() == 'deeplabv3plusxception-8os':
        cfg.MODEL_OUTPUT_STRIDE = 8
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusxception' or name.lower() == 'deeplabv3plusxception-16os':
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusxception_1':
        return deeplabv3plus_1(cfg)
    elif name.lower() == 'deeplabv3plusxceptiondensity':
        cfg.MODEL_AUX_OUT = 6
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusxceptiondensityx':
        cfg.MODEL_AUX_OUT = 5
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusxception5':
        cfg.MODEL_AUX_OUT = 5
        return deeplabv3plus(cfg)

    elif name.lower() == 'deeplabv3plus3branch':
        return deeplabv3plus3branch(cfg)
    elif name.lower() == 'deeplabv3plus3branchsegattention':
        return deeplabv3plus3branchsegattention(cfg)

    elif name.lower() == 'deeplabv3plus3branch1.1':
        return deeplabv3plus3branch_1(cfg)
    elif name.lower() == 'deeplabv3plus3branchdensity':
        cfg.MODEL_AUX_OUT = 6
        return deeplabv3plus3branch(cfg)
    elif name.lower() == 'deeplabv3plus3branchdensityx':
        cfg.MODEL_AUX_OUT = 5
        return deeplabv3plus3branch(cfg)

    elif name.lower() == 'deeplabv3plusmultioffset':
        cfg.MODEL_AUX_OUT = 2 + n_class * 2
        return deeplabv3plusmultioffset(cfg)

    elif name.lower() == 'deeplabv3plus3branch2':
        return deeplabv3plus3branch2(cfg)

    elif name.lower() == 'deeplabv3plusseg':
        return deeplabv3plus_seg(cfg)
    elif name.lower() == 'deeplabv3plussegselflearning':
        return deeplabv3plus_seg_selflearning(cfg)

    elif name.lower() == 'deeplabv3pluslabel':
        return deeplabv3pluslabel(cfg)

    elif name.lower() == 'deeplabv3plusxceptionthick' or name.lower() == 'deeplabv3plusthick':
        cfg.MODEL_ASPP_OUTDIM = 512
        cfg.MODEL_SHORTCUT_DIM = 96
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusatrousresnet101-8os':
        cfg.MODEL_BACKBONE = 'res101_atrous'
        cfg.MODEL_OUTPUT_STRIDE = 8
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusatrousresnet101' or name.lower() == 'deeplabv3plusatrousresnet101-16os':
        cfg.MODEL_BACKBONE = 'res101_atrous'
        return deeplabv3plus(cfg)

    elif name.lower() == 'deeplabv3plusatrousresnext101-8os':
        cfg.MODEL_BACKBONE = 'resnext101_atrous'
        cfg.MODEL_OUTPUT_STRIDE = 8
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusatrousresnext101' or name.lower() == 'deeplabv3plusatrousresnext101-16os':
        cfg.MODEL_BACKBONE = 'resnext101_atrous'
        return deeplabv3plus(cfg)

    elif name.lower() == 'deeplabv3plusatrousresnet152-8os':
        cfg.MODEL_BACKBONE = 'res152_atrous'
        cfg.MODEL_OUTPUT_STRIDE = 8
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusatrousresnet152' or name.lower() == 'deeplabv3plusatrousresnet152-16os':
        cfg.MODEL_BACKBONE = 'res152_atrous'
        return deeplabv3plus(cfg)
    else:
        raise Exception(f'*** model name wrong, {name} not legal')


def _deeplabv3plus_get_model_test():
    name_list = ['deeplabv3plusxception-8os', 'deeplabv3plusxception', 'deeplabv3plusxception-16os',
                 'deeplabv3plusatrousresnet101-8os', 'deeplabv3plusatrousresnet101-16os',
                 'deeplabv3plusatrousresnet101',
                 'deeplabv3plusatrousresnet152', 'deeplabv3plusatrousresnet152-16os',
                 'deeplabv3plusatrousresnet152-8s']
    for name in name_list:
        model = get_deeplabv3plus_model(name=name, n_class=21)
        print(model)


def get_deeplab_multidenseaspp_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.MultiDenseASPP import DeepLabv3PlusMultiDenseASPP, DeepLabv3PlusMultiDenseASPPv2, \
        DeepLabv3PlusMultiDenseASPPv3
    if name == 'deeplabv3plusmultidenseaspp':
        return DeepLabv3PlusMultiDenseASPP(cfg)
    elif name == 'deeplabv3plusmultidenseasppv2':
        return DeepLabv3PlusMultiDenseASPPv2(cfg, me=True)
    elif name == 'deeplabv3plusmultidenseasppv3':
        return DeepLabv3PlusMultiDenseASPPv3(cfg, v=3, me=True)
    elif name == 'deeplabv3plusmultidenseasppv3.1':
        return DeepLabv3PlusMultiDenseASPPv3(cfg, v=3.1, me=True)

    raise Exception(f'*** model name wrong, {name} not legal')


def get_deeplabv3plusCA_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.deeplabv3plusCA import deeplabv3plusCA
    if name == 'deeplabv3plusca':
        return deeplabv3plusCA(cfg)

    raise Exception(f'*** model name wrong, {name} not legal')


def get_deeplabv3plusdeconv_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.deeplabv3plusDeconv import deeplabv3plusdeconv
    if name == 'deeplabv3plusdeconv':
        return deeplabv3plusdeconv(cfg)


def get_deeplabv3plusbalanceddenseaspp_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.BalancedDenseASPP import DeepLabv3PlusBalancedDenseASPP
    if name == 'deeplabv3plusbdaspp':
        return DeepLabv3PlusBalancedDenseASPP(cfg)
    elif name == 'deeplabv3plusbdasppca':
        return DeepLabv3PlusBalancedDenseASPP(cfg, ca=True)
    elif name == 'deeplabv3plusbdasppprime':
        return DeepLabv3PlusBalancedDenseASPP(cfg, prime=True)
    elif name == 'deeplabv3plusbdasppprimeca':
        return DeepLabv3PlusBalancedDenseASPP(cfg, prime=True, ca=True)
    elif name == 'deeplabv3plusbdasppthick256':
        cfg.MODEL_ASPP_OUTDIM = 1024
        return DeepLabv3PlusBalancedDenseASPP(cfg, prime=False, ca=False, me=True, dense_out=256)
    raise Exception(f'*** model name wrong, {name} not legal')


def _test_get_deeplabv3plusbalanceddenseaspp_model():
    model = get_deeplabv3plusbalanceddenseaspp_model(name='deeplabv3plusbdasppthick256', n_class=21)
    print(model)
    pass


def get_deeplabv3plusAD_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.deeplabv3plusAdversaralLoss import deeplabv3plusAD
    if name == 'deeplabv3plusad':
        return deeplabv3plusAD(cfg)
    raise Exception(f'*** model name wrong, {name} not legal')


def get_deeplabv3plusPS_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.deeplabv3plusPixelShuffle import deeplabv3plusPS, deeplabv3plusPSALL, \
        deeplabv3plusPSALLv2, deeplabv3plusPSALLv3, deeplabv3plusPSALLv4, deeplabv3plusPSALL1_1
    if name == 'deeplabv3plusps4':
        return deeplabv3plusPS(cfg, PS=4)
    elif name == 'deeplabv3plusps2':
        return deeplabv3plusPS(cfg, PS=2)
    elif name == 'deeplabv3pluspsall':
        return deeplabv3plusPSALL(cfg)
    elif name == 'deeplabv3pluspsallv1.1':
        return deeplabv3plusPSALL1_1(cfg)
    elif name == 'deeplabv3pluspsallv2':
        return deeplabv3plusPSALLv2(cfg, us_dilation_rate=1)
    elif name == 'deeplabv3pluspsallv2.1':
        return deeplabv3plusPSALLv2(cfg, us_dilation_rate=2)
    elif name == 'deeplabv3pluspsallv2.2':
        return deeplabv3plusPSALLv2(cfg, us_dilation_rate=4)
    elif name == 'deeplabv3pluspsallv3':
        return deeplabv3plusPSALLv3(cfg, me=True)
    elif name == 'deeplabv3pluspsallv4':
        return deeplabv3plusPSALLv4(cfg, me=True)
    raise Exception(f'*** model name wrong, {name} not legal')


def get_deeplabv3plusJPU_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.deeplabv3plusJPU import deeplabv3plusJPU, deeplabv3plusJPUv2
    if name == 'deeplabv3plusjpu256':
        return deeplabv3plusJPU(cfg, jpu='same width')
    elif name == 'deeplabv3plusjpu256v2':
        return deeplabv3plusJPUv2(cfg)
    raise Exception(f'*** model name wrong, {name} not legal')


def get_deeplabv3plusKCASPP_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.deeplabv3plusKCASPP import deeplabv3plusKCASPP, deeplabv3plusKCASPPv2
    if name == 'deeplabv3pluskcaspp':
        return deeplabv3plusKCASPP(cfg)
    elif name == 'deeplabv3pluskcasppv2':
        return deeplabv3plusKCASPPv2(cfg)
    raise Exception(f'*** model name wrong, {name} not legal')


def get_tkcnet_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.tkcnet import TKCNet, TKCNetv2
    if name == 'tkcnet':
        return TKCNet(cfg, TFA='l')
    elif name == 'tkcnetsmall':
        return TKCNet(cfg, TFA='s')
    elif name == 'tkcnetv2small':
        return TKCNetv2(cfg, TFA='s')
    raise Exception(f'*** model name wrong, {name} not legal')


def get_danet_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.danet import DANet, deeplabv3plusDA, \
        deeplabv3plusASPPDA, deeplabv3plusDAASPP, deeplabv3plusPA, DANetwithHRNet
    if name == 'danet':
        return DANet(cfg, only_fuse=True)
    elif name == 'danet-8os':
        cfg.MODEL_OUTPUT_STRIDE = 8
        return DANet(cfg, only_fuse=True)
    elif name == 'danetmultioutputs':
        return DANet(cfg, only_fuse=False)
    elif name == 'danetwithdeeplab':
        return deeplabv3plusDA(cfg)
    elif name == 'danetdeeplabpa':
        return deeplabv3plusPA(cfg)

    elif name == 'danetwithhrnet':
        return DANetwithHRNet(cfg, only_fuse=True)

    elif name == 'danetwithhrnetmultioutputs':
        return DANetwithHRNet(cfg, only_fuse=False)


    elif name == 'asppdanet':
        return deeplabv3plusASPPDA(cfg, only_fuse=True)
    elif name == 'danetaspp':
        return deeplabv3plusDAASPP(cfg, only_fuse=True)
    elif name == 'danetaspp-8os':
        cfg.MODEL_OUTPUT_STRIDE = 8
        return deeplabv3plusDAASPP(cfg, only_fuse=True)

    elif name == 'asppdanetmultioutputs':
        return deeplabv3plusASPPDA(cfg, only_fuse=False)
    elif name == 'danetasppmultioutputs':
        return deeplabv3plusDAASPP(cfg, only_fuse=False)
    elif name == 'danetasppmultioutputs-8os':
        cfg.MODEL_OUTPUT_STRIDE = 8
        return deeplabv3plusDAASPP(cfg, only_fuse=False)
    raise Exception(f'*** model name wrong, {name} not legal')


def get_deeplabv3pluspspaspp_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.deeplabv3plusPSPASPP import deeplabv3plusPSPASPPthick, deeplabv3plusPSPASPP
    if name == 'deeplabv3pluspspasppthick':
        return deeplabv3plusPSPASPPthick(cfg)
    elif name == 'deeplabv3pluspspaspp':
        return deeplabv3plusPSPASPP(cfg)
    raise Exception(f'*** model name wrong, {name} not legal')


def get_deeplabv3plusHR_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.deeplabv3plusHRNet import deeplabv3plusHR, deeplabv3plusHRPSPASPPthick, \
        deeplabv3plusHRPSPASPPthickSEloss, deeplabv3plusHREncPSPASPPthick, deeplabv3plusHRMultiPSPASPPthick, \
        deeplabv3plusHRMultiASPP, deeplabv3plusHRMultiASPPSELoss, deeplabv3plusHRMultiPSPASPPthickseloss
    if name == 'deeplabv3plushr48_not_pre':
        return deeplabv3plusHR(cfg, pretrain=False)
    elif name == 'deeplabv3plushr48':
        return deeplabv3plusHR(cfg, pretrain=True)
    elif name == 'deeplabv3plushr48pspasppthick':
        return deeplabv3plusHRPSPASPPthick(cfg, pretrain=True)

    elif name == 'deeplabv3plushr48multipspasppthick':
        return deeplabv3plusHRMultiPSPASPPthick(cfg, pretrain=True)
    elif name == 'deeplabv3plushr48multipspasppthickseloss':
        return deeplabv3plusHRMultiPSPASPPthickseloss(cfg, pretrain=True)

    elif name == 'deeplabv3plushr48multiaspp':
        return deeplabv3plusHRMultiASPP(cfg, pretrain=True)
    elif name == 'deeplabv3plushr48multiasppseloss':
        return deeplabv3plusHRMultiASPPSELoss(cfg, pretrain=True)

    elif name == 'deeplabv3plushr48encpspasppthick':
        return deeplabv3plusHREncPSPASPPthick(cfg, pretrain=True)
    elif name == 'deeplabv3plushr48pspasppthickseloss':
        return deeplabv3plusHRPSPASPPthickSEloss(cfg, pretrain=True)
    raise Exception(f'*** model name wrong, {name} not legal')


def get_deeplabv3plusRF_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.deeplabv3plusRF import deeplabv3plusRF
    if name == 'deeplabv3plusrf':
        return deeplabv3plusRF(cfg)
    raise Exception(f'*** model name wrong, {name} not legal')


def get_EncNet_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.deeplabv3plusEnc import EncNet, deeplabv3plusEnc, deeplabv3plusEncv2, deeplabv3plusEncv3, \
        deeplabv3plusEncv2_1
    if name == 'encnet':
        return EncNet(cfg)
    elif name == 'encnet-8os':
        cfg.MODEL_OUTPUT_STRIDE = 8
        return EncNet(cfg)
    elif name == 'deeplabv3plusenc' or name == 'deeplabv3plusenc512':
        return deeplabv3plusEnc(cfg, enc_out=512)
    elif name == 'deeplabv3plusencv2':
        return deeplabv3plusEncv2(cfg)
    elif name == 'deeplabv3plusencv2.1':
        return deeplabv3plusEncv2_1(cfg)
    elif name == 'deeplabv3plusencv3':
        return deeplabv3plusEncv3(cfg)
    elif name == 'deeplabv3plusenc1024':
        return deeplabv3plusEnc(cfg, enc_out=1024)
    raise Exception(f'*** model name wrong, {name} not legal')


def get_deeplabv3plusupsampleandcls_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.deeplabv3plusupsampleandcls import deeplabv3plus
    if name == 'deeplabv3plusupsampleandcls':
        return deeplabv3plus(cfg)
    raise Exception(f'*** model name wrong, {name} not legal')


def get_deeplabv3plusgff_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.deeplabv3plusGFF import GFFASPP, deeplabv3plusGFF
    if name == 'deeplabv3plusgff':
        return deeplabv3plusGFF(cfg)
    elif name == 'gffaspp':
        return deeplabv3plusGFF(cfg)
    raise Exception(f'*** model name wrong, {name} not legal')


def get_deeplabv3plusTriAtt_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.deeplabv3plusTriatt import deeplabv3plusTriAtt
    if name == 'deeplabv3plustriatt':
        return deeplabv3plusTriAtt(cfg)
    raise Exception(f'*** model name wrong, {name} not legal')


def get_deeplabv3plusDUsampling_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.deeplabv3plusDUsample import deeplabv3plusDUpsample
    if name == 'deeplabv3plusdupsample':
        return deeplabv3plusDUpsample(cfg)
    raise Exception(f'*** model name wrong, {name} not legal')


def get_deeplabv3plusExFuse_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.deeplabv3plusExFuse import deeplabv3plusdap
    if name == 'deeplabv3plusexfusedap':
        return deeplabv3plusdap(cfg)
    raise Exception(f'*** model name wrong, {name} not legal')


if __name__ == '__main__':
    import torch

    model = get_deeplabv3plus_model(name='deeplabv3plusxceptiondensity', n_class=20)
    h, w = 160, 160
    x = torch.rand(2, 3, h, w)
    y = model(x)
    density_map = y[0, 2 + 2:4 + 2]  # 2 x h x w
    density_map[0] = density_map[0].softmax(dim=-1) * float(w) / 1024.
    density_map[1] = density_map[1].softmax(dim=-2) * float(h) / 1024.
    xym_s = density_map  #
    xym_s[0] = xym_s[0].cumsum(dim=-1)  # w->x
    xym_s[1] = xym_s[1].cumsum(dim=-2)  # h->y
    print(y.shape)
    # print(y[0,5].softmax(dim=0))
    print(xym_s)

    xm = torch.linspace(0, 2, 2048).view(
        1, 1, -1).expand(1, 1024, 2048)
    ym = torch.linspace(0, 1, 1024).view(
        1, -1, 1).expand(1, 1024, 2048)
    xym = torch.cat((xm, ym), 0)
    print(xym[:, 1:h, 1:w])
    print(xym[:, 1:h + 1, 1:w + 1] - xym_s)
    pass
