# File: evaluate_fast_rcnn.py

import torch
import numpy as np

def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def evaluate_p_r_f1(model, data_loader, device, iou_threshold=0.5, score_threshold=0.5):
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for idx, (images, targets) in enumerate(data_loader):
            print(f"Processing batch {idx + 1} / {len(data_loader)}...")
            # print(f"Batch {idx + 1}: Images = {len(images)}, Targets = {len(targets)}")
            images = [img.to(device) for img in images]
            preds = model(images)

            for pred, target in zip(preds, targets):
                gt_boxes = target['boxes'].numpy()  # shape [N,4]
                gt_labels = target['labels'].numpy()  # shape [N]

                boxes = pred['boxes'].cpu().numpy()
                labels = pred['labels'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()

                # Filter by confidence threshold
                keep_idx = scores >= score_threshold
                boxes = boxes[keep_idx]
                labels = labels[keep_idx]

                # We'll do a simple approach: match predicted boxes -> ground truth
                matched_gt = set()
                for box, label in zip(boxes, labels):
                    iou_best = 0.0
                    best_gt_idx = -1
                    for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                        if gt_label != label:  # class mismatch
                            continue
                        if i in matched_gt:  # already matched
                            continue
                        iou_val = box_iou(box, gt_box)
                        if iou_val > iou_best:
                            iou_best = iou_val
                            best_gt_idx = i
                    if iou_best >= iou_threshold and best_gt_idx != -1:
                        total_tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        total_fp += 1
                # un-matched GT => FN
                num_gt = len(gt_boxes)
                total_fn += (num_gt - len(matched_gt))

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1

if __name__ == "__main__":
    import json
    from torch.utils.data import DataLoader
    from dataset_fast_rcnn import FastRCNNDataset
    from get_fast_rcnn_model import get_fast_rcnn_model
    import torch

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_fast_rcnn_model(num_classes=3)
    model.load_state_dict(torch.load("fast_rcnn_model.pth", map_location=device, weights_only=True))
    model.to(device)

    # Load val set
    with open("val_anns.json", "r") as f:
        val_anns = json.load(f)
    val_dataset = FastRCNNDataset(val_anns, label_shift=1)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Evaluate
    p, r, f = evaluate_p_r_f1(model, val_loader, device, iou_threshold=0.5, score_threshold=0.5)
    print(f"Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}")
