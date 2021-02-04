import numpy as np
from PIL import Image
import torch
import numbers
import random
import torchvision
import torchvision.transforms.functional as F


class ClipToTensor(object):
    """
    Convert a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    to a torch.FloatTensor of shape (C x m x H x W) in the range [0, 1.0]
    """

    def __init__(self, numpy=False):

        self.numpy = numpy

    def __call__(self, clip):

        w, h = clip[0].size
        th_clip = torch.zeros([3, len(clip), int(h), int(w)])
        for id, img in enumerate(clip):
            img = np.array(img, copy=False)

            # convert img from (H, W, C) to (C, W, H) format
            img_convert = torch.from_numpy(img).permute(2, 0, 1).contiguous()
            th_clip[:, id, :, :] = img_convert
        
        return th_clip.float()


class ImgToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    See ``ToTensor`` for more details.
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    """
    # handle PIL Image
    def __init__(self, div=True):

        self.div = div

        return

    def __call__(self, img):
        img = np.array(img, copy=False)
        img_convert = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        return img_convert.float() / 255.0 if self.div else img_convert.float()


class ClipRandomCrop(object):
    def __init__(self, size):

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))

        else:
            self.size = size

    def __call__(self, clip):

        w, h = clip[0].size
        th, tw = self.size

        output = []
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in clip:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                output.append(img)
            else:
                output.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return output


class ClipCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, clip):

        return [self.worker(img) for img in clip]


class ClipNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


class ClipResize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, clip):

        return [self.worker(img) for img in clip]
