# Suppose in a separate script or inside train_efficientdet.py after training:

import torch
import json
from torch.utils.data import DataLoader
from dataset_efficientdet import EfficientDetDataset
from evaluate_metrics import evaluate_p_r_f1
from train_efficientdet import create_model, collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load model
model = create_model(num_classes=2, compound_coef=0)
model.load_state_dict(torch.load("efficientdet_d0.pth"))
model.to(device)
model.eval()

# 2. Prepare val dataset/loader
with open("val_anns.json","r") as f:
    val_anns = json.load(f)
val_dataset = EfficientDetDataset(val_anns, label_offset=1)
val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn)

# 3. Evaluate
precision, recall, f1 = evaluate_p_r_f1(model, val_loader, device, score_thresh=0.5, iou_thresh=0.5)
print(f"EfficientDet => Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
