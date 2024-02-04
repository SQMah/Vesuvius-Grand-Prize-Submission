from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np
import random


class ChannelInvert(ImageOnlyTransform):
    """Reverse channels of an input HWC image.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def apply(self, img, **params):
        # Assuming img is an HWC image.
        img = img[..., ::-1]
        return img

    def get_transform_init_args_names(self):
        # This function returns a tuple of parameter names that will be
        # used to initialize the transform object, if necessary.
        return ()


class FourthAugment(ImageOnlyTransform):
    """Custom transformation that shuffles channels in the input image."""

    def __init__(self, always_apply=False, p=0.5):
        super(FourthAugment, self).__init__(always_apply, p)

    def apply(self, img, **params):
        # Assuming img is an HWC image.
        in_chans = img.shape[-1]
        image_tmp = np.zeros_like(img)
        cropping_num = random.randint(12, 16)

        start_idx = random.randint(0, in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)

        start_paste_idx = random.randint(0, in_chans - cropping_num)

        tmp = np.arange(start_paste_idx, start_paste_idx + cropping_num)
        np.random.shuffle(tmp)

        cutout_idx = random.randint(0, 2)
        temporal_random_cutout_idx = tmp[:cutout_idx]

        image_tmp[..., start_paste_idx:start_paste_idx + cropping_num] = img[..., crop_indices]

        if random.random() > 0.4:
            image_tmp[..., temporal_random_cutout_idx] = 0
        return image_tmp

    def get_transform_init_args_names(self):
        return ()
