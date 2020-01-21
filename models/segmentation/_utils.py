from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as VF
import random


class _SimpleSegmentationModel(nn.Module):
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, aux_classifier=None):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result


class ComposeJoint(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, target):
        for transform in self.transforms:
            if isinstance(transform, list):
                x = transform[0](x)
                target = transform[1](target)
            else:
                x, target = transform(x, target)

        return x, target


class RandomHorizontalFlipJoint(object):
    def __call__(self, img, target):
        if random.random() < .5:
            return (VF.hflip(img), VF.hflip(target))
        return img, target


class RandomResizedCropJoint(transforms.RandomResizedCrop):
    def __init__(self, **kwargs):
        super(RandomResizedCropJoint, self).__init__(**kwargs)

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = VF.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        target = VF.resized_crop(target, i, j, h, w, self.size, self.interpolation)
        return img, target


class remap_ambiguous(object):
    def __call__(self, img):
        img[img == 255] = 21
        return img


