""" Image Transformation """


import torch
import torchvision.transforms as T
from torch import Tensor

import random
from typing import Dict, Tuple, Union
from PIL import Image

from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


class RandomOrder(object):
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img: Image.Image, targets: dict) -> Tuple[Image.Image, dict]:
        random.shuffle(self.transforms)
        for transform in self.transforms:
            img, targets = transform(img, targets)
        return img, targets
    

class RandomSizeCrop(object):
    def __init__(self, num_class, min_scale=0.8):
        self.num_class = num_class
        self.min_scale = min_scale

    def __call__(self, img: Image.Image, targets: Dict[str, Tensor]) -> Tuple[Image.Image, Dict[str, Tensor]]:
        img_w, img_h = img.size
        scale_w = random.uniform(self.min_scale, 1)
        scale_h = random.uniform(self.min_scale, 1)
        new_w = int(img_w * scale_w)
        new_h = int(img_h * scale_h)

        region = T.RandomCrop.get_params(img, (new_h, new_w))

        # fix bboxes
        i, j, h, w = region
        boxes = box_cxcywh_to_xyxy(targets['boxes']) * torch.as_tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)

        mask = ~torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        targets['labels'][mask] = self.num_class

        targets['boxes'] = box_xyxy_to_cxcywh(cropped_boxes.reshape(-1, 4) / torch.as_tensor([new_w, new_h, new_w, new_h], dtype=torch.float32))

        return T.functional.crop(img, *region), targets


class Normalize(object):
    def __init__(self):
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, img: Image.Image, targets: Dict[str, Tensor]) -> Tuple[Image.Image, Dict[str, Tensor]]:
        img = self.transform(img)
        boxes = box_cxcywh_to_xyxy(targets['boxes']).clamp(0, 1)
        targets['boxes'] = box_xyxy_to_cxcywh(boxes)
        return img, targets


class Resize(object):
    def __init__(self, *args, **kwargs):
        self.transform = T.Resize(*args, **kwargs)

    def __call__(self, img: Image.Image, targets: dict) -> Tuple[Image.Image, dict]:
        return self.transform(img), targets


class RandomVerticalFlip(object):
    def __init__(self, p=.5):
        self.p = p
    
    def __call__(self, img: Image.Image, targets: Dict[str, Tensor]) -> Tuple[Image.Image, Dict[str, Tensor]]:
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            targets['boxes'][..., 1] = 1 - targets['boxes'][..., 1]
        return img, targets


class RandomHorizontalFlip(object):
    def __init__(self, p=.5):
        self.p = p

    def __call__(self, img: Image.Image, targets: Dict[str, Tensor]) -> Tuple[Image.Image, Dict[str, Tensor]]:
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            targets['boxes'][..., 0] = 1 - targets['boxes'][..., 0]
        return img, targets


class ColorJitter(object):
    def __init__(self, *args, **kwargs):
        self.transform = T.ColorJitter(*args, **kwargs)

    def __call__(self, img: Image.Image, targets: dict) -> Tuple[Image.Image, dict]:
        return self.transform(img), targets


class Compose(object):
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img: Image.Image, targets: dict) -> Tuple[Union[Tensor, Image.Image], dict]:
        for transform in self.transforms:
            img, targets = transform(img, targets)
        return img, targets