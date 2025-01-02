import torch
import json
import numpy as np
from torch.utils.data import DataLoader
import os
from dataset_ssd import CustomSSDDataset  
from get_ssd_model import get_ssd_model   
from evaluate import evaluate_p_r_f1

def collate_fn(batch):
    # To handle images/targets of different shapes
    return tuple(zip(*batch))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load annotations (same as your Faster R-CNN process)
    with open("train_anns.json", "r") as f:
        train_anns = json.load(f)
    with open("val_anns.json", "r") as f:
        val_anns = json.load(f)
    
    # 2. If needed, shift labels by +1 for SSD
    #    An easy way is to do it inside the dataset class or do it inline here.
    
    # 3. Create Dataset and DataLoaders
    train_dataset = CustomSSDDataset(train_anns)
    val_dataset = CustomSSDDataset(val_anns)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4,   # smaller or bigger depending on GPU memory
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 4. Initialize the model
    # 2 real classes + 1 background = 3
    model = get_ssd_model(num_classes=3)
    
    model.to(device)
    # model.train()
    
    # 5. Optimizer, etc.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i+1}/{len(train_loader)}] Loss: {losses.item():.4f}")
        
        epoch_loss /= len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_loss:.4f}")
        
        lr_scheduler.step()
        
        # Optionally do a validation loop (similar to training) to track val loss
        # but SSD's forward pass with targets expects the same format. 
        # We'll skip for brevity.
    
    # 6. Save model
    torch.save(model.state_dict(), "ssd_model.pth")
    print("SSD training complete. Model saved to ssd_model.pth.")
    
    #7. After finishing all epochs (or inside an epoch loop):from 
    precision, recall, f1 = evaluate_p_r_f1(model, val_loader, device, iou_threshold=0.5, score_threshold=0.5)
    print(f"\n\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n\n")

if __name__ == "__main__":
    main()
