import torch
import json
import os
from torch.utils.data import DataLoader

from dataset_efficientdet import EfficientDetDataset
from effdet import create_model

def collate_fn(batch):
    """
    Custom collate function for EfficientDet.
    Each item in `batch` is (img, target). We want a list of images, list of targets.
    """
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load your JSON annotations
    with open("train_anns.json", "r") as f:
        train_anns = json.load(f)
    with open("val_anns.json", "r") as f:
        val_anns = json.load(f)

    # 2. Create Datasets
    #    If your YOLO labels = [0,1], and you want [1,2] => set label_offset=1 in the dataset
    train_dataset = EfficientDetDataset(train_anns, transforms=None, label_offset=1)
    val_dataset = EfficientDetDataset(val_anns, transforms=None, label_offset=1)

    # 3. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # 4. Initialize the model with 2 real classes => total classes = 3 (background + 2)
    #    model_name='tf_efficientdet_d0' is EfficientDet-D0
    model = create_model(
        model_name='tf_efficientdet_d0',
        bench_task='train',
        num_classes=3,          # background + 2 classes => total 3
        pretrained=True,
        image_size=(512, 512)   # pass a tuple instead of a single int
    )
    model.to(device)

    # 5. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 6. Training loop (minimal example)
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for imgs, targets in train_loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass => returns (loss, outputs)
            loss, _ = model(imgs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {epoch_loss:.4f}")

        # (Optional) validation loop with val_loader if desired

    # 7. Save model
    torch.save(model.state_dict(), "efficientdet_d0.pth")
    print("Training finished. Weights saved to efficientdet_d0.pth.")

if __name__ == "__main__":
    main()
