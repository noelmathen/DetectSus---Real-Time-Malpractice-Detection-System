# Using EfficientDET/evaluate_metrics.py

import torch
import numpy as np

def box_iou(box1, box2):
    # box1, box2: [xmin, ymin, xmax, ymax]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0
    return inter / union

def evaluate_p_r_f1(model, data_loader, device, score_thresh=0.5, iou_thresh=0.5):
    model.eval()
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for imgs, targets in data_loader:
            imgs = [img.to(device) for img in imgs]
            preds = model(imgs, torch.jit.annotate(List[Dict[str, torch.Tensor]], []))
            # The output from DetBenchTrain in inference mode is a list of dicts:
            # [ { 'detections': [N,6] } ] => each row: [xmin, ymin, xmax, ymax, score, class]
            
            for pred, target in zip(preds, targets):
                # pred['detections'] shape: [N, 6]
                dets = pred['detections'].cpu().numpy()  # [N,6]
                # columns: [xmin, ymin, xmax, ymax, score, cls]
                gt_boxes = target['bbox'].cpu().numpy()  # shape: [M,4]
                gt_labels = target['cls'].cpu().numpy()  # shape: [M]

                # Filter out low scores
                mask = dets[:,4] >= score_thresh
                dets = dets[mask]
                
                # Sort by score descending if you want (optional)
                # dets = dets[dets[:,4].argsort()[::-1]]
                
                matched_gt = set()

                for d in dets:
                    box_pred = d[:4]
                    cls_pred = int(d[5])
                    
                    # If your classes are 1..N, ignoring 0=background
                    # you may skip background predictions
                    if cls_pred == 0:
                        # background label => skip
                        continue

                    iou_best = 0
                    match_idx = -1
                    for i, (g_box, g_cls) in enumerate(zip(gt_boxes, gt_labels)):
                        # If class doesn't match, skip
                        if g_cls != cls_pred:
                            continue
                        if i in matched_gt:
                            continue
                        
                        iou_val = box_iou(box_pred, g_box)
                        if iou_val > iou_best:
                            iou_best = iou_val
                            match_idx = i
                    
                    if iou_best >= iou_thresh and match_idx != -1:
                        total_tp += 1
                        matched_gt.add(match_idx)
                    else:
                        total_fp += 1
                
                # Now, any ground-truth not matched => FN
                total_fn += (len(gt_boxes) - len(matched_gt))
    
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return precision, recall, f1

if __name__ == "__main__":
    pass
