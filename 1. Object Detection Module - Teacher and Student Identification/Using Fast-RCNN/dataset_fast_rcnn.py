# File: dataset_fast_rcnn.py

import torch
import cv2
import numpy as np

class FastRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, transforms=None, label_shift=None):
        """
        annotations: list of dicts like:
          {
            'image_path': str,
            'boxes': [[xmin, ymin, xmax, ymax], ...],
            'labels': [class_id, ...]
          }
        transforms: optional data augmentation
        label_shift: int to shift labels if needed. 
                     If using 2 classes (0,1) + background => total classes=3
                     Usually we can keep them as is if we set num_classes=2+1=3 in the model.
                     If your model expects background=0, real classes=1..N, 
                     you'd do label_shift=1. Adjust as needed.
        """
        self.annotations = annotations
        self.transforms = transforms
        self.label_shift = label_shift

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = ann['image_path']
        boxes = ann['boxes']
        labels = ann['labels']

        # Read image
        img = cv2.imread(img_path)
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to float tensor [C,H,W] in [0..1]
        img_tensor = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Convert boxes, labels to torch tensors
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        # If you want to shift labels so that background=0, teacher=1, student=2:
        if self.label_shift is not None:
            labels_tensor = labels_tensor + self.label_shift

        # Build target dict
        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor
        }
        # You can also add 'image_id', 'area', 'iscrowd' if needed.

        # If using transforms (augmentations), apply them here (TorchVision or Albumentations, etc.)

        return img_tensor, target
