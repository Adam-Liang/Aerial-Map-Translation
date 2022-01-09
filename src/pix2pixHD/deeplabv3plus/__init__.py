def get_model(args, name=None, n_class=None):
    from models.BranchedERFNet import BranchedERFNet, ERFNet3branch
    from models.deeplabv3plus import get_deeplabv3plus_model
    if name is None:
        name = args.model_name
    if n_class is None:
        n_class = args.n_class

    if name == "erfnet":
        model = BranchedERFNet(num_classes=(4, n_class), encoder=None)
    elif name == "erfnet5":
        model = BranchedERFNet(num_classes=(5, n_class), encoder=None)
    elif name == "erfnetdensity":
        model = BranchedERFNet(num_classes=(6, n_class), encoder=None)

    elif name == "erfnet3branch":
        model = ERFNet3branch(num_classes=(4, n_class), encoder=None)
    elif "deeplabv3plus" in name:
        model = get_deeplabv3plus_model(name=name, n_class=n_class)
    elif "edxception" in name:
        model = get_deeplabv3plus_model(name=name, n_class=n_class)
    else:
        raise RuntimeError("model \"{}\" not available".format(name))
    print('===> Model:', model.__class__.__name__)
    return model


def get_weakly_model(args, name=None, n_class=None):
    from models.weakly_model import BranchedERFNet, ERFNet3branch, LabelERFNet3branch, ERFNetBBox
    from models.deeplabv3plus import get_deeplabv3plus_model
    if name is None:
        name = args.model_name

    if n_class is None:
        n_class = args.n_class

    if name == "erfnet":
        model = BranchedERFNet(num_classes=(4, n_class), encoder=None)
    elif name == "erfnetdensity":
        model = BranchedERFNet(num_classes=(6, n_class), encoder=None)
    elif name == "erfnetbbox":
        model = ERFNetBBox(num_classes=n_class, encoder=None)
    elif name == "erfnet3branch":
        model = ERFNet3branch(num_classes=(4, n_class), encoder=None)
    elif name == "labelerfnet3branch":
        model = LabelERFNet3branch(num_classes=(4, n_class))
    elif "deeplabv3plus" in name:
        model = get_deeplabv3plus_model(name=name, n_class=n_class)

    else:
        raise RuntimeError("model \"{}\" not available".format(name))
    print('===> Model:', model.__class__.__name__)
    return model
