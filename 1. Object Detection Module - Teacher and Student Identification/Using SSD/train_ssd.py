import torch
import json
import numpy as np
from torch.utils.data import DataLoader
import os
from dataset_ssd import CustomSSDDataset
from get_ssd_model import get_ssd_model
from evaluate import evaluate_p_r_f1
import winsound  # For the alarm sound on errors

def collate_fn(batch):
    # To handle images/targets of different shapes
    return tuple(zip(*batch))

def play_alarm():
    """Play an alarm sound."""
    try:
        winsound.Beep(1000, 500)  # Frequency: 1000Hz, Duration: 500ms
        print("Alarm triggered!")
    except Exception as e:
        print(f"Failed to play alarm: {str(e)}")

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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load annotations
    with open("train_anns.json", "r") as f:
        train_anns = json.load(f)
    with open("val_anns.json", "r") as f:
        val_anns = json.load(f)
    
    # 2. Create Dataset and DataLoaders
    train_dataset = CustomSSDDataset(train_anns)
    val_dataset = CustomSSDDataset(val_anns)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # 3. Initialize the model
    model = get_ssd_model(num_classes=3)
    model.to(device)
    
    # 4. Optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            
            try:
                # Validate and fix bounding boxes
                validate_and_fix_bboxes(targets, device)
                targets = [{k: v for k, v in t.items()} for t in targets]
                
                # Compute loss
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
                
                if (i + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i+1}/{len(train_loader)}] Loss: {losses.item():.4f}")
            except Exception as e:
                print(f"Error during training step: {str(e)}")
                play_alarm()  # Trigger alarm on error
                continue
        
        epoch_loss /= len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_loss:.4f}")
        lr_scheduler.step()
    
    # Save the model
    torch.save(model.state_dict(), "ssd_model.pth")
    print("SSD training complete. Model saved to ssd_model.pth.")
    
    # Evaluate precision, recall, and F1 score
    precision, recall, f1 = evaluate_p_r_f1(model, val_loader, device, iou_threshold=0.5, score_threshold=0.5)
    print(f"\n\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n\n")

if __name__ == "__main__":
    main()
