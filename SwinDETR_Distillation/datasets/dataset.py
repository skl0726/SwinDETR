""" COCO Dataset """


import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from glob import glob
from typing import List, Tuple, Dict

import os

import datasets.transforms as T


class COCODataset(Dataset):
    def __init__(self, root: str, annotation: str, target_height: str, target_width: str, num_class: int, train=True):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())

        self.target_height = target_height
        self.target_width = target_width
        self.num_class = num_class

        if train:
            self.transforms = T.Compose([
                T.RandomOrder([
                    T.RandomHorizontalFlip(),
                    T.RandomSizeCrop(num_class)
                ]),
                T.Resize((target_height, target_width)),
                T.ColorJitter(brightness=.2, contrast=.1, saturation=.1, hue=0),
                T.Normalize()
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((target_height, target_width)),
                T.Normalize()
            ])

        self.new_index = {}
        classes = []
        for i, (k, v) in enumerate(self.coco.cats.items()):
            self.new_index[k] = i
            classes.append(v['name'])

        # with open('classes.txt', 'w') as f:
        #     f.write(str(classes))

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx) -> Tuple[Tensor, dict]:
        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.root, img_info['file_name'])

        image = Image.open(img_path).convert('RGB')
        annotations = self._load_annotations(img_id, img_info['width'], img_info['height'])

        if len(annotations) == 0:
            targets = {
                'boxes': torch.zeros(1, 4, dtype=torch.float32),
                'labels': torch.as_tensor([self.num_class], dtype=torch.int64)
            }
        else:
            targets = {
                'boxes': torch.as_tensor(annotations[..., :-1], dtype=torch.float32),
                'labels': torch.as_tensor(annotations[..., -1], dtype=torch.int64)
            }

        image, targets = self.transforms(image, targets)

        return image, targets

    def _load_annotations(self, img_id, img_width, img_height) -> np.ndarray:
        ans = []

        for annotation in self.coco.imgToAnns[img_id]:
            cat = self.new_index[annotation['category_id']]
            bbox = annotation['bbox']

            # convert from [tlx, tly, w, h] to [cx, cy, w, h]
            bbox[0] += bbox[2] / 2
            bbox[1] += bbox[3] / 2

            bbox = [val / img_height if i % 2 else val / img_width for i, val in enumerate(bbox)]

            ans.append(bbox + [cat])

        return np.asarray(ans)
    

def collate_function(batch: List[Tuple[Tensor, dict]]) -> Tuple[Tensor, Tuple[Dict[str, Tensor]]]:
    batch = tuple(zip(*batch))
    return torch.stack(batch[0]), batch[1]