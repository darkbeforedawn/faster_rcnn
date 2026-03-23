from typing import Literal, List
from pathlib import Path
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import albumentations as A
import cv2
from PIL import Image
import numpy as np


def rpn_collate_fn(batch):
    # batch is a list of tuple(s) each containing [img, list_of_bboxes, list_of_labels]
    # of shapes [(3, h, w), list[N, 4], [N]]
    batch_list = []
    for img, bboxes, labels in batch:
        batch_list.append({
            "image": img,
            "bboxes": torch.as_tensor(bboxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64)
        })
    return batch_list

class VOCDataset(Dataset):
    def __init__(self, root: str, split: Literal['train', 'val']):
        super().__init__()
        assert split in ['train', 'val'], 'split must be either "train" or "val"'
        if split == 'train':
            self.transforms = A.Compose([
                # geometric augs
                A.SmallestMaxSize(800, None, cv2.INTER_CUBIC, p=1.),
                A.RandomSizedBBoxSafeCrop(800, 800, 0.2, cv2.INTER_CUBIC, p=0.5),
                A.HorizontalFlip(p=0.5),
                # photometric
                A.ColorJitter((0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.3, 0.3), p=0.5),
                A.GaussNoise(p=0.2),
                A.ToGray(3, p=0.05),
                # default
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.pytorch.transforms.ToTensorV2(p=1.)
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1, label_fields=['labels']))
        else:
            self.transforms = A.Compose([
                A.SmallestMaxSize(800, None, cv2.INTER_CUBIC, p=1.),
                # default
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.pytorch.transforms.ToTensorV2(p=1.)
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.0, label_fields=['labels']))
        self.sample2bbox = defaultdict(list)
        self.sample2class = defaultdict(list)
        self.class2index = {
            "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
            "bottle": 5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
            "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
            "motorbike": 14, "person": 15, "pottedplant": 16, "sheep": 17,
            "sofa": 18, "train": 19, "tvmonitor": 20
            }
        self.samples_paths: List[str] = []
        self.root = Path(root)

        with open(self.root.joinpath(f'ImageSets/Main/{split}.txt'), 'r', encoding='utf-8') as f:
            splits = [str(x).strip() for x in f.readlines()]
        for f in os.listdir(self.root.joinpath('Annotations')):
            xml_f = ET.parse(self.root.joinpath('Annotations', f))
            r = xml_f.getroot() # annotations
            name = r.find('filename').text.replace('.jpg', '').strip()
            if name in splits:
                for o in r.findall('object'):
                    self.sample2bbox[name].append([
                        int(o.find('bndbox/xmin').text), # x1
                        int(o.find('bndbox/ymin').text), # y1
                        int(o.find('bndbox/xmax').text), # x2
                        int(o.find('bndbox/ymax').text), # y2
                    ])
                    cls_name = o.find('name').text.strip() # name of the class
                    self.sample2class[name].append(self.class2index[cls_name])

        for f in os.listdir(self.root.joinpath('JPEGImages')):
            if str(f).replace('.jpg', '').strip() in splits:
                self.samples_paths.append(str(self.root.joinpath('JPEGImages', f)))

    def __len__(self):
        return len(self.samples_paths)

    def __getitem__(self, index: int):
        p = self.samples_paths[index]
        img = np.array(Image.open(p).convert('RGB'))
        name = p.split('/')[-1].replace('.jpg', '').strip()
        bboxes = self.sample2bbox[name]
        labels = self.sample2class[name]
        tf = self.transforms(image=img, bboxes=bboxes, labels=labels)
        return tf['image'], tf['bboxes'], tf['labels']
