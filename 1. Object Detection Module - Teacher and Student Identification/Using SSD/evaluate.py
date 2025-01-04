import torch
import numpy as np

def box_iou(box1, box2):
    """
    Compute IoU between box1 and box2.
    box1, box2: [xmin, ymin, xmax, ymax]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    iou = inter_area / union_area
    return iou

def evaluate_p_r_f1(model, data_loader, device, iou_threshold=0.5, score_threshold=0.5):
    """
    Evaluate precision, recall, and F1-score on the given data_loader.
    model: a trained Faster R-CNN model (in eval mode).
    data_loader: DataLoader for the validation set.
    device: 'cuda' or 'cpu'.
    iou_threshold: IoU threshold for a prediction to count as correct.
    score_threshold: confidence threshold to consider a detection “valid”.
    """
    model.eval()
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            # Move images to device
            images = [img.to(device) for img in images]
            
            # predictions is a list of dicts: [{'boxes':..., 'labels':..., 'scores':...}, ...]
            predictions = model(images)
            
            # For each image in the batch
            for pred, target in zip(predictions, targets):
                # Convert target boxes, labels to CPU numpy
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                
                # Filter out predictions below the score threshold
                keep_idx = pred['scores'] >= score_threshold
                pred_boxes = pred['boxes'][keep_idx].cpu().numpy()
                pred_labels = pred['labels'][keep_idx].cpu().numpy()
                
                # Track matches
                matched_gt = set()  # keep track of which gt indexes are matched
                
                for pb, pl in zip(pred_boxes, pred_labels):
                    iou_best = 0
                    gt_match_idx = -1
                    for i, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                        if i in matched_gt or gl != pl:
                            continue
                        iou_val = box_iou(pb, gb)
                        if iou_val > iou_best:
                            iou_best = iou_val
                            gt_match_idx = i
                    
                    if iou_best >= iou_threshold and gt_match_idx != -1:
                        # True positive
                        total_tp += 1
                        matched_gt.add(gt_match_idx)
                    else:
                        # False positive
                        total_fp += 1
                
                # For each ground-truth that was not matched, we have a false negative
                num_gt = len(gt_boxes)
                total_fn += (num_gt - len(matched_gt))
    
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return precision, recall, f1