# File: get_fast_rcnn_model.py

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_fast_rcnn_model(num_classes=3):
    """
    Load a Faster R-CNN with ResNet50-FPN, pretrained on COCO (91 classes),
    then replace the classification head to have `num_classes` outputs.
    Typically, if you have 2 real classes + background => num_classes=3.
    """
    # 1. Load model pretrained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="COCO_V1"
    )

    # 2. Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
