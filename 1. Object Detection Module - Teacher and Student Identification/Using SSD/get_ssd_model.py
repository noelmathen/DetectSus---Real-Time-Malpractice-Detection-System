import torch
import torchvision
from torchvision.models.detection.ssd import SSDClassificationHead

def get_ssd_model(num_classes=3):
    """
    Loads a COCO-pretrained SSD300_VGG16 model (which has 91 classes),
    then replaces its classification head so it predicts `num_classes`
    instead (including background).

    This lets you benefit from the COCO-pretrained backbone
    while customizing the number of classes.
    """
    # 1. Load the COCO-pretrained SSD
    model = torchvision.models.detection.ssd300_vgg16(
        weights="SSD300_VGG16_Weights.COCO_V1"
    )
    # This gives us a model with 91 classes (including background).

    # 2. Extract the current classification head
    old_classification_head = model.head.classification_head  # type: SSDClassificationHead

    # 3. Get the in_channels for each feature map's conv layer from `module_list`
    #    Each item in `module_list` is a conv layer: Conv2d(in_channels=..., out_channels=..., ...)
    in_channels = []
    for conv_layer in old_classification_head.module_list:
        in_channels.append(conv_layer.in_channels)  # read from Conv2d

    # 4. Obtain the number of default anchors (per feature map) from the model's anchor generator
    anchor_generator = model.anchor_generator
    num_anchors_per_location = anchor_generator.num_anchors_per_location()  
    # e.g., something like [4, 6, 6, 6, 6, 6] for the 6 feature maps

    # 5. Create a new SSDClassificationHead with the desired number of classes
    #    (including background). Example: num_classes=3 => [0=background, 1=teacher, 2=student].
    new_classification_head = SSDClassificationHead(
        in_channels=in_channels, 
        num_anchors=num_anchors_per_location,
        num_classes=num_classes
    )

    # 6. Replace the old classification head with the new one
    model.head.classification_head = new_classification_head

    return model
