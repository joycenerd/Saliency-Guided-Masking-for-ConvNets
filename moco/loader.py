# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random

import numpy as np
from torchvision import transforms

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, strategy):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.base_transform = base_transform
        self.hp_transform = transforms.Compose([HighPassFilter(kernel_size=(11,11), sigma=5)])
        self.blur_transform = transforms.Compose([
            transforms.GaussianBlur(kernel_size=(31,31), sigma=10),
            normalize
        ])
        self.norm = transforms.Compose([normalize])
        self.strategy = strategy

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)

        if self.strategy == "hp":
            q = self.norm(q)
            k = self.norm(k)
            q_hp = self.hp_transform(q)
            k_hp = self.hp_transform(k)
            v_hp = k_hp.clone().detach()
            
            return [q, k, q_hp, k_hp, v_hp]
        
        elif self.strategy == "blur":
            q_blur = self.blur_transform(q)
            v_blur = self.blur_transform(k)

            q = self.norm(q)
            k = self.norm(k)
            v = k.clone().detach()

            return [q, k, v, q_blur, v_blur]

        elif self.strategy == "mean":
            q = self.norm(q)
            k = self.norm(k)
            v = k.clone().detach()
            return [q, k, v]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        if len(self.sigma) == 2:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
        else:
            sigma = self.sigma[0]
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class HighPassFilter(object):
    """High pass filter augmentation: original image minus image after low pass filter (GaussianBlur)"""
    def __init__(self, kernel_size, sigma):
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self,x):
        T = transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)
        x_lp = T(x)
        x_hp = x - x_lp
        return x_hp

