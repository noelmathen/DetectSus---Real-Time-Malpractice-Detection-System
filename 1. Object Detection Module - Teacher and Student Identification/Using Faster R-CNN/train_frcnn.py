import torch
import torchvision
from torch.utils.data import DataLoader
import json
import numpy as np
import os
from dataset_frcnn import CustomFRCNNDataset
from model import get_faster_rcnn_model  # the function we wrote above
from evaluate import evaluate_p_r_f1 

def collate_fn(batch):
    # This is needed for batching images with different box counts
    return tuple(zip(*batch))

def main():
    # 1. Load your annotations
    with open('train_anns.json', 'r') as f:
        train_anns = json.load(f)
    with open('val_anns.json', 'r') as f:
        val_anns = json.load(f)
    
    # 2. Hyperparameters
    num_classes = 3  # Example: if you have 2 real classes + 1 background
    num_epochs = 10
    batch_size = 2
    lr = 0.005
    
    # 3. Create the datasets
    train_dataset = CustomFRCNNDataset(train_anns)
    val_dataset = CustomFRCNNDataset(val_anns)
    
    # 4. Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 5. Initialize the model
    model = get_faster_rcnn_model(num_classes=num_classes)
    
    # 6. Move model to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    # 7. Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    
    # LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # 8. Training loop
    for epoch in range(num_epochs):
        model.train()
        i = 0
        for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            i += 1
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i}/{len(train_loader)}] Loss: {losses.item():.4f}")
        
        lr_scheduler.step()
        
        # 9. Validation loop (simple version, no metrics)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                batch_loss = sum(loss for loss in loss_dict.values())
                val_loss += batch_loss.item()
        
        val_loss /= len(val_loader)
        print(f"Validation Loss after epoch {epoch+1}: {val_loss:.4f}")
    
    # 10. Save the trained model weights
    torch.save(model.state_dict(), "fasterrcnn_model.pth")
    print("Model training complete and saved to fasterrcnn_model.pth")
    
    # after finishing all epochs (or inside an epoch loop):
    precision, recall, f1 = evaluate_p_r_f1(model, val_loader, device, iou_threshold=0.5, score_threshold=0.5)
    print(f"\n\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n\n")


if __name__ == "__main__":
    main()
