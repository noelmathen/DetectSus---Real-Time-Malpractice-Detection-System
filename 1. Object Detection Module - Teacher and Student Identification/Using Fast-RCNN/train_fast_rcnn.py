# File: train_fast_rcnn.py

import torch
import torchvision
from torch.utils.data import DataLoader
import json
import os

from dataset_fast_rcnn import FastRCNNDataset
from get_fast_rcnn_model import get_fast_rcnn_model

def validate_and_fix_bboxes(targets, device):
    """
    Ensure all bounding boxes have positive width and height.
    Invalid boxes are removed from targets.
    """
    for target in targets:
        valid_boxes = []
        valid_labels = []
        for box, label in zip(target["boxes"], target["labels"]):
            x_min, y_min, x_max, y_max = box.tolist()
            if x_max > x_min and y_max > y_min:
                valid_boxes.append([x_min, y_min, x_max, y_max])
                valid_labels.append(label.item())
        
        # Move valid boxes and labels to the correct device
        target["boxes"] = torch.tensor(valid_boxes, dtype=torch.float32).to(device)
        target["labels"] = torch.tensor(valid_labels, dtype=torch.int64).to(device)

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the JSON annotations produced by convert_yolo_to_fast_rcnn.py
    with open("train_anns.json", "r") as f:
        train_anns = json.load(f)
    with open("val_anns.json", "r") as f:
        val_anns = json.load(f)

    # 2. Create dataset objects
    train_dataset = FastRCNNDataset(train_anns, transforms=None, label_shift=1)
    val_dataset = FastRCNNDataset(val_anns, transforms=None, label_shift=1)

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # 4. Build the model (num_classes=3 => background+2)
    model = get_fast_rcnn_model(num_classes=3)
    model.to(device)

    # 5. Set up the optimizer and LR scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        print(f"---- EPOCH {epoch+1}/{num_epochs} ----")
        model.train()
        total_loss = 0.0

        for i, (images, targets) in enumerate(train_loader, 1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Validate and fix bounding boxes
            validate_and_fix_bboxes(targets, device)

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

            # Display progress percentage
            progress = (i / len(train_loader)) * 100
            print(f"\rProgress: {progress:.2f}%", end='')

        print()  # Move to the next line after progress
        epoch_loss = total_loss / len(train_loader)
        print(f"Train Loss: {epoch_loss:.4f}")

        lr_scheduler.step()

        # Optional: Evaluate on val_loader for val loss
        # (Skipping detailed metrics here—will do a separate evaluate function)

    # Save model
    torch.save(model.state_dict(), "fast_rcnn_model.pth")
    print("Training complete. Model saved to fast_rcnn_model.pth")

if __name__ == "__main__":
    main()
