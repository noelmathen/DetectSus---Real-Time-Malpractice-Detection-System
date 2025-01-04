# Using EfficientDET/dataset_efficientdet.py

import torch
import cv2
import json
import os

class EfficientDetDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, transforms=None, label_offset=0):
        """
        annotations: list of dicts, each with:
            {
              "image_path": str,
              "boxes": [[xmin, ymin, xmax, ymax], ...],
              "labels": [class_id, ...]
            }
        transforms: optional, for data augmentation
        label_offset: int, if you need to shift labels so that 0 is background, 1=firstClass, etc.
                      For example, if YOLO had (0=teacher, 1=student),
                      and your model expects (1=teacher,2=student), set label_offset=1.
                      But typically if your model is built for [0..num_classes-1], you can keep offset=0
                      as long as you handle background separately in your model config.
        """
        self.annotations = annotations
        self.transforms = transforms
        self.label_offset = label_offset

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = self.annotations[idx]
        image_path = data["image_path"]
        boxes = data["boxes"]
        labels = data["labels"]  # e.g. [0,1] or [1,2] depending on usage

        # Read image with OpenCV
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # make it RGB

        # Convert to float tensor [C,H,W] and scale [0..1]
        img_t = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Convert boxes and labels to tensors
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        # Apply offset if needed
        labels_t = torch.as_tensor([lbl + self.label_offset for lbl in labels], dtype=torch.int64)

        # EfficientDet typically expects:
        #   'bbox': float tensor [N,4]
        #   'cls': float tensor [N] (or int64)
        target = {
            "bbox": boxes_t,
            "cls": labels_t
        }

        # Optionally apply transforms or augmentations if needed
        if self.transforms:
            # e.g. Albumentations or custom transformations
            # Make sure to transform both image + boxes.
            pass

        return img_t, target
