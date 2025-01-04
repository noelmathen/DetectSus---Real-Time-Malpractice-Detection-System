import os
import json
import cv2

def yolo_to_coco(root_dir, output_file, class_names):
    """
    Converts YOLO format labels to COCO format.
    Args:
        root_dir: Directory containing `images` and `labels` subdirectories.
        output_file: Path to save the COCO JSON file.
        class_names: List of class names (background not included).
    """
    images_dir = os.path.join(root_dir, 'images', 'train')
    labels_dir = os.path.join(root_dir, 'labels', 'train')

    images = []
    annotations = []
    image_id = 0
    annotation_id = 0

    for img_file in os.listdir(images_dir):
        if not img_file.endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')

        # Read image to get width and height
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        # Add image entry
        images.append({
            "id": image_id,
            "file_name": img_file,
            "height": h,
            "width": w
        })

        # Parse label file
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())
                    x_min = (x_center - bbox_width / 2) * w
                    y_min = (y_center - bbox_height / 2) * h
                    width = bbox_width * w
                    height = bbox_height * h

                    annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(class_id),
                        "bbox": [x_min, y_min, width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    annotation_id += 1

        image_id += 1

    # Create categories
    categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]

    # Save to COCO format
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(output_file, 'w') as f:
        json.dump(coco_data, f)

if __name__ == "__main__":
    root_dir = 'combined_dataset'
    output_file = 'train_coco.json'
    class_names = ["teacher", "student"]  # Modify as per your classes
    yolo_to_coco(root_dir, output_file, class_names)
