import os
import cv2

def yolo_to_frcnn(root_dir):
    # root_dir = "combined_dataset"
    images_dir_train = os.path.join(root_dir, 'images', 'train')
    labels_dir_train = os.path.join(root_dir, 'labels', 'train')
    images_dir_val = os.path.join(root_dir, 'images', 'val')
    labels_dir_val = os.path.join(root_dir, 'labels', 'val')
    
    # We’ll create two lists: train_annotations, val_annotations.
    # Each element in these lists will be a dict containing:
    # {
    #   'image_path': str, 
    #   'boxes': [[xmin, ymin, xmax, ymax], ...],
    #   'labels': [class_id, class_id, ...]
    # }
    
    train_annotations = []
    val_annotations = []
    
    def convert_split(images_dir, labels_dir):
        annotations = []
        image_files = sorted(os.listdir(images_dir))
        
        for img_file in image_files:
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            image_path = os.path.join(images_dir, img_file)
            
            # Construct the corresponding label .txt file name
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            # If label file doesn’t exist, skip
            if not os.path.exists(label_path):
                annotations.append({
                    'image_path': image_path,
                    'boxes': [],
                    'labels': []
                })
                continue
            
            # Read image to get width/height
            img = cv2.imread(image_path)
            h, w, _ = img.shape
            
            boxes = []
            labels = []
            
            with open(label_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    class_id = int(data[0])
                    x_center = float(data[1]) * w
                    y_center = float(data[2]) * h
                    bbox_width = float(data[3]) * w
                    bbox_height = float(data[4]) * h
                    
                    xmin = x_center - (bbox_width / 2)
                    ymin = y_center - (bbox_height / 2)
                    xmax = x_center + (bbox_width / 2)
                    ymax = y_center + (bbox_height / 2)
                    
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
    root_dir = 'combined_dataset'
    train_anns, val_anns = yolo_to_frcnn(root_dir)
    print(f"Number of training samples: {len(train_anns)}")
    print(f"Number of validation samples: {len(val_anns)}")

    # You can now save train_anns and val_anns to disk as a .pkl or .json
    # for easier retrieval in your training script, e.g.:
    import json
    with open('train_anns.json', 'w') as f:
        json.dump(train_anns, f)
    with open('val_anns.json', 'w') as f:
        json.dump(val_anns, f)
