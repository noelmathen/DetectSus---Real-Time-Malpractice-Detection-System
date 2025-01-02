import torch
import cv2
import os

class CustomSSDDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, transforms=None):
        """
        annotations: a list of dicts with:
          {
            'image_path': str,
            'boxes': [[xmin, ymin, xmax, ymax], ...],
            'labels': [class_id, class_id, ...]
          }
        transforms: optional augmentations (random flip, etc.)
        """
        self.annotations = annotations
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = self.annotations[idx]
        image_path = data['image_path']
        boxes = data['boxes']
        labels = [l + 1 for l in data['labels']]
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Load image (BGR -> RGB -> Torch Tensor)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor: shape [C,H,W], scale to [0..1]
        img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        
        # If transforms are needed, apply them here (or inside a transform pipeline).
        
        return img, target
