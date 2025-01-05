# File: convert_yolo_to_fast_rcnn.py

import os
import cv2
import json

def yolo_to_fast_rcnn(root_dir):
    """
    root_dir: path to 'combined_dataset' which has 'images/train', 'images/val'
              and corresponding 'labels/train', 'labels/val'.

    Returns: (train_annotations, val_annotations)
    Each annotation is a dict:
      {
        'image_path': str,
        'boxes': [[xmin, ymin, xmax, ymax], ...],
        'labels': [class_id, ...]
      }
    """
    images_dir_train = os.path.join(root_dir, 'images', 'train')
    labels_dir_train = os.path.join(root_dir, 'labels', 'train')
    images_dir_val = os.path.join(root_dir, 'images', 'val')
    labels_dir_val = os.path.join(root_dir, 'labels', 'val')
    
    def convert_split(images_dir, labels_dir):
        annotations = []
        image_files = sorted(os.listdir(images_dir))
        for img_file in image_files:
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            image_path = os.path.join(images_dir, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)

            if not os.path.exists(label_path):
                # If no label file, assume no objects
                annotations.append({
                    'image_path': image_path,
                    'boxes': [],
                    'labels': []
                })
                continue

            # Read image to get actual width/height
            img = cv2.imread(image_path)
            if img is None:
                # In case of corrupt or missing image
                annotations.append({
                    'image_path': image_path,
                    'boxes': [],
                    'labels': []
                })
                continue
            height, width = img.shape[:2]

            boxes = []
            labels = []
            with open(label_path, 'r') as f:
                for line in f:
                    # YOLO format: class_id x_center y_center w h (all normalized)
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    w_bbox = float(parts[3]) * width
                    h_bbox = float(parts[4]) * height

                    xmin = x_center - w_bbox / 2
                    ymin = y_center - h_bbox / 2
                    xmax = x_center + w_bbox / 2
                    ymax = y_center + h_bbox / 2

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id)
            
            annotations.append({
                'image_path': image_path,
                'boxes': boxes,
                'labels': labels
            })
        return annotations
    
    train_annotations = convert_split(images_dir_train, labels_dir_train)
    val_annotations = convert_split(images_dir_val, labels_dir_val)

    return train_annotations, val_annotations

if __name__ == "__main__":
    root_dir = "combined_dataset"  # path to your dataset
    train_anns, val_anns = yolo_to_fast_rcnn(root_dir)

    print(f"Train samples: {len(train_anns)}")
    print(f"Val samples: {len(val_anns)}")

    # Save them to JSON for future usage
    with open("train_anns.json", "w") as f:
        json.dump(train_anns, f)
    with open("val_anns.json", "w") as f:
        json.dump(val_anns, f)

    print("Annotations saved to train_anns.json and val_anns.json")
