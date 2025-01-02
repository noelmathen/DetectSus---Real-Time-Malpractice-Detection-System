import torch
import cv2
import os

class CustomFRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, transforms=None):
        """
        annotations: a list of dicts, each dict has:
            {
              'image_path': str,
              'boxes': [[xmin, ymin, xmax, ymax], ...],
              'labels': [class_id, class_id, ...]
            }
        transforms: any data augmentation or normalization transforms
        """
        self.annotations = annotations
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = self.annotations[idx]
        image_path = data['image_path']
        boxes = data['boxes']
        labels = data['labels']
        
        # Read the image with OpenCV (returns a NumPy array)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert the NumPy array to a torch float tensor in [C, H, W] format
        # and scale pixel values from [0..255] to [0..1].
        img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Convert boxes and labels to torch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])

        # Apply your transforms if you have any (now that it's a tensor or PIL)
        # e.g., if self.transforms is a torchvision transform that expects a PIL image,
        # then you'd first reconvert `img` to PIL, apply transforms, and then back to tensor.
        
        return img, target
