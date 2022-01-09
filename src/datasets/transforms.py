"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import collections
import random

import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from torchvision import transforms as T

import torch
import math
import numbers
from datasets.photo_augmentations import PhotometricDistortForInstanceSegmentation

KEYS = ('image', 'instance', 'label', 'smask')
# MEAN = [0.485, 0.456, 0.406]
MEAN = [0.5, 0.5, 0.5]
# STD = [0.229, 0.224, 0.225]
STD = [0.5, 0.5, 0.5]


def get_transform(args):
    """
    1. 'object_center',  'crop', 'resize_crop'
    2. 'color'
    3. default Hflip
    4. 'bbox'
    :param args:
    :return:
    """
    aug = args.aug
    transform_list = []

    # crop ways
    if 'object_center' in aug:
        transform_list.append(CropRandomObject(size=args.crop_size, prefer=False))
    elif 'object_prefer' in aug:
        transform_list.append(CropRandomObject(size=args.crop_size, prefer=True))

    if 'resize_crop' in aug or 'resized_crop'in aug:
        transform_list.append(RandomResizedCrop(size=args.crop_size))
    elif 'crop' in aug:
        transform_list.append(RandomCrop(size=args.crop_size))
    else:
        pass

    # use Photometric
    if 'color' in aug:
        transform_list.append(PhotometricDistortForInstanceSegmentation(p=1.0))

    # HFlip:
    transform_list.append(RandomHorizontalFlip())

    # ToTensor
    transform_list.append(ToTensor())

    # Get bbox target
    if 'bbox' in aug:
        transform_list.append(GetBBoxOneHotTensor(args.n_class))

    transform_list.append(Normalize(mean=MEAN, std=STD))

    # Train transform
    print('===> Transform:')
    for i, tf in enumerate(transform_list):
        print(i + 1, tf.__class__.__name__)
    print('==========\n')
    train_transform = T.Compose(transform_list)

    # Val transform
    val_transform_list = [StrideAlign(), ToTensor()]
    if 'bbox' in aug:
        val_transform_list.append(GetBBoxOneHotTensor(args.n_class))

    val_transform_list.append(Normalize(mean=MEAN, std=STD))

    print('===> Transform:')
    for i, tf in enumerate(val_transform_list):
        print(i + 1, tf.__class__.__name__)
    print('==========\n')
    val_transform = T.Compose(val_transform_list)

    return train_transform, val_transform


class GetGrabCutMask:
    def __init__(self):
        pass

    def __call__(self, sample):
        # Tensor
        instance = sample['instance'].squeeze(dim=0)
        instance_ids = instance.unique()
        instance_ids = instance_ids[instance_ids != 0]
        label = sample['label'].squeeze(dim=0)
        instance_to_cls = {int(i): int(label[instance == i].unique()) for i in instance_ids}

        h, w = instance.shape
        instance = instance.numpy()
        res = []
        for inst_id in instance_to_cls:
            index = np.argwhere(instance == inst_id)
            ul = index.min(axis=0)
            br = index.max(axis=0)
            bbox = list(ul) + list(br)
            hw = br - ul
            area = int(hw[1] + 1) * int(hw[0] + 1)
            res.append({
                'cls': instance_to_cls[inst_id],
                'area': area,
                'bbox': bbox,
                'instance_id': inst_id
            })
        res.sort(key=lambda x: x['area'], reverse=True)

        one_hot_mask = np.zeros((self.n_class, h, w))
        one_hot_mask[0, :, :] = 1
        for inst in res:
            u, l, b, r = inst['bbox']
            one_hot_mask[inst['cls'], u:b + 1, l:r + 1] = 1
            one_hot_mask[0, u:b + 1, l:r + 1] = 0
        bbox_target = torch.from_numpy(one_hot_mask).float()
        sample['bbox'] = bbox_target
        return sample


class GetBBoxOneHotTensor:
    def __init__(self, n_class):
        if n_class in (8, 20):
            self.n_class = n_class + 1
        elif n_class in (9, 21):
            self.n_class = n_class
        else:
            raise NotImplementedError

    def __call__(self, sample):
        # Tensor
        instance = sample['instance'].squeeze(dim=0)
        instance_ids = instance.unique()
        instance_ids = instance_ids[instance_ids != 0]
        label = sample['label'].squeeze(dim=0)
        instance_to_cls = {int(i): int(label[instance == i].unique()) for i in instance_ids}

        h, w = instance.shape
        instance = instance.numpy()
        res = []
        for inst_id in instance_to_cls:
            index = np.argwhere(instance == inst_id)
            ul = index.min(axis=0)
            br = index.max(axis=0)
            bbox = list(ul) + list(br)
            hw = br - ul
            area = int(hw[1] + 1) * int(hw[0] + 1)
            res.append({
                'cls': instance_to_cls[inst_id],
                'area': area,
                'bbox': bbox,
                'instance_id': inst_id
            })
        res.sort(key=lambda x: x['area'], reverse=True)

        one_hot_mask = np.zeros((self.n_class, h, w))
        one_hot_mask[0, :, :] = 1
        for inst in res:
            u, l, b, r = inst['bbox']
            one_hot_mask[inst['cls'], u:b + 1, l:r + 1] = 1
            one_hot_mask[0, u:b + 1, l:r + 1] = 0
        bbox_target = torch.from_numpy(one_hot_mask).float()
        sample['bbox'] = bbox_target
        # sample['bbox_info'] = res

        # instance_mask = np.zeros(instance.shape)
        # for inst in res:
        #     u, l, b, r = inst['bbox']
        #     instance_mask[u:b + 1, l:r + 1] = inst['instance_id']
        # bbox_instance_target = torch.from_numpy(instance_mask).float()

        return sample


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}


class StrideAlign(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, stride=16, resize_both=True, down_sample=None, interpolation=Image.BILINEAR):
        self.stride = stride
        self.interpolation = interpolation
        self.down_sample = down_sample
        self.resize_both = resize_both

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        w, h = sample['image'].size
        if self.down_sample is not None:
            w, h = w / self.down_sample, h / self.down_sample
        w_stride, h_stride = math.floor(w / self.stride), math.floor(h / self.stride)
        h_w_resize = (int(h_stride * self.stride), int(w_stride * self.stride))
        sample['image'] = F.resize(sample['image'], h_w_resize, self.interpolation)
        if self.resize_both:
            sample['instance'] = F.resize(sample['instance'], h_w_resize, Image.NEAREST)
            sample['label'] = F.resize(sample['label'], h_w_resize, Image.NEAREST)
            sample['smask'] = F.resize(sample['smask'], h_w_resize, Image.NEAREST)
        # img_resized.show()
        return sample

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            sample['image'] = F.hflip(sample['image'])
            sample['label'] = F.hflip(sample['label'])
            sample['instance'] = F.hflip(sample['instance'])
            sample['smask'] = F.hflip(sample['smask'])
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees=10, resample=Image.BILINEAR, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, sample):
        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)
        sample['image'] = F.rotate(sample['image'], angle, self.resample, self.expand, self.center)
        sample['label'] = F.rotate(sample['label'], angle, Image.NEAREST, self.expand, self.center)
        sample['instance'] = F.rotate(sample['instance'], angle, Image.NEAREST, self.expand, self.center)
        sample['smask'] = F.rotate(sample['smask'], angle, Image.NEAREST, self.expand, self.center)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.25, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(sample['image'], self.scale, self.ratio)
        sample['image'] = F.resized_crop(sample['image'], i, j, h, w, self.size, self.interpolation)
        sample['label'] = F.resized_crop(sample['label'], i, j, h, w, self.size, Image.NEAREST)
        sample['instance'] = F.resized_crop(sample['instance'], i, j, h, w, self.size, Image.NEAREST)
        sample['smask'] = F.resized_crop(sample['smask'], i, j, h, w, self.size, Image.NEAREST)
        return sample

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        # print(type(img), type(target), type(bbox_target))
        sample['image'] = F.normalize(sample['image'], self.mean, self.std)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CropRandomObject:

    def __init__(self, keys=KEYS, object_key="instance", size=100, prefer=True):
        self.keys = keys
        self.object_key = object_key
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.prefer = prefer

    def __call__(self, sample):
        """
        sample by label uniquely

        class_names = ('person', 'rider', 'car', 'truck',
                   'bus', 'train', 'motorcycle', 'bicycle')
        class_ids = (24, 25, 26, 27, 28, 31, 32, 33)
        prefer 6, 5, 4, 7
        :param sample:
        :return:
        """

        def get_prefer(unique_labels: np.array):
            # print(f'unique_labels:{unique_labels}')
            label_prefer = []
            for i in (6, 5, 4, 7):
                if i in unique_labels:
                    label_prefer.append(i)
            if len(label_prefer) > 0:
                random_label = np.random.choice(np.array(label_prefer), 1)
            else:
                random_label = np.random.choice(unique_labels, 1)
            # print(f'random_label:{random_label}')
            return random_label

        object_map = np.array(sample[self.object_key], copy=False)
        h, w = object_map.shape

        unique_objects = np.unique(object_map)
        unique_objects = unique_objects[unique_objects != 0]

        label = np.array(sample['label'])
        label[label == 255] = 0
        unique_labels = np.unique(label)
        unique_labels = unique_labels[unique_labels != 0]
        instance_to_label = {int(i): int(np.unique(label[object_map == i])) for i in unique_objects}
        # print('instance_to_label',instance_to_label)
        label_to_instance = {}
        for i in unique_objects:
            i = int(i)
            if label_to_instance.get(instance_to_label[i]) is None:
                label_to_instance[instance_to_label[i]] = []
                label_to_instance[instance_to_label[i]].append(i)
            else:
                label_to_instance[instance_to_label[i]].append(i)
        # print('label_to_instance', label_to_instance)

        if unique_labels.size > 0:
            if self.prefer:
                random_label = get_prefer(unique_labels)
            else:
                random_label = np.random.choice(unique_labels, 1)

            random_id = np.random.choice(label_to_instance[int(random_label)], 1)

            y, x = np.where(object_map == random_id)
            ym, xm = np.mean(y), np.mean(x)

            i = int(np.clip(ym - self.size[1] / 2, 0, h - self.size[1]))
            j = int(np.clip(xm - self.size[0] / 2, 0, w - self.size[0]))

        else:
            i = random.randint(0, h - self.size[1])
            j = random.randint(0, w - self.size[0])

        for k in self.keys:
            assert (k in sample)

            sample[k] = F.crop(sample[k], i, j, self.size[1], self.size[0])

        return sample


class RandomCrop(T.RandomCrop):

    def __init__(self, keys=KEYS, size=100):

        super().__init__(size, pad_if_needed=True)
        self.keys = keys

    def __call__(self, sample):

        params = None

        for k in self.keys:

            assert (k in sample)

            if params is None:
                params = self.get_params(sample[k], self.size)

            sample[k] = F.crop(sample[k], *params)

        return sample


class RandomRotation_(T.RandomRotation):

    def __init__(self, keys=KEYS, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.keys = keys

        if isinstance(self.resample, collections.Iterable):
            assert (len(keys) == len(self.resample))

    def __call__(self, sample):

        angle = self.get_params(self.degrees)

        for idx, k in enumerate(self.keys):

            assert (k in sample)

            resample = self.resample
            if isinstance(resample, collections.Iterable):
                resample = resample[idx]

            sample[k] = F.rotate(sample[k], angle, resample,
                                 self.expand, self.center)

        return sample


class Resize(T.Resize):

    def __init__(self, keys=KEYS, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.keys = keys

        if isinstance(self.interpolation, collections.Iterable):
            assert (len(keys) == len(self.interpolation))

    def __call__(self, sample):

        for idx, k in enumerate(self.keys):

            assert (k in sample)

            interpolation = self.interpolation
            if isinstance(interpolation, collections.Iterable):
                interpolation = interpolation[idx]

            sample[k] = F.resize(sample[k], self.size, interpolation)

        return sample


class ToTensor(object):

    def __init__(self, keys=KEYS, type=(torch.FloatTensor, torch.ByteTensor, torch.ByteTensor, torch.ByteTensor)):

        if isinstance(type, collections.Iterable):
            assert (len(keys) == len(type))

        self.keys = keys
        self.type = type

    def __call__(self, sample):

        for idx, k in enumerate(self.keys):

            assert (k in sample)

            sample[k] = F.to_tensor(sample[k])

            t = self.type
            if isinstance(t, collections.Iterable):
                t = t[idx]

            if t == torch.ByteTensor:
                sample[k] = sample[k] * 255

            sample[k] = sample[k].type(t)

        return sample
