import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from src.config import GRID_SIZES
class Backbone(nn.Module):
    """
    DenseNet backbone for feature extraction.
    Extracts features at multiple scales for YOLO v3.
    """
    def __init__(self, model_name="darknet53", pretrained=True):
        super(Backbone, self).__init__()
        ### Change here to use Different backbone ###
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True
        )
    def forward(self, x):
        features = self.backbone(x)
        return features[-1], features[-2], features[-3]  # Return feature maps at 3 scales


# ============================================================================
# Neck & Prediction Head
# ============================================================================
class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and LeakyReLU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class YOLOv3Head(nn.Module):
    """
    YOLO v3 detection head with FPN-like neck structure.
    Performs multi-scale predictions.
    """
    def __init__(self, num_classes=20, num_anchors=3):
        super(YOLOv3Head, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.output_channels = num_anchors * (5 + num_classes)
        #################YOUR CODE###################
        # ==== Scale 1: 13x13 (small scale - detects large objects) ====
        # Input from backbone: typically 1024 channels for darknet53
        self.scale1_conv = nn.Sequential(
            ConvBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            ConvBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            ConvBlock(1024, 512, kernel_size=1, stride=1, padding=0),
        )
        # Classifier for scale 1
        self.scale1_detect_conv = nn.Sequential(
            ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, self.output_channels, kernel_size=1, stride=1, padding=0)
        )

        # Upsample for scale 2
        self.scale_13_upsample = nn.Sequential(
            ConvBlock(512, 256, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # ==== Scale 2: 26x26 (medium scale - detects medium objects) ====
        # Input: 256 (from upsample) + 512 (from backbone) = 768 channels
        self.scale2_conv = nn.Sequential(
            ConvBlock(768, 256, kernel_size=1, stride=1, padding=0),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 256, kernel_size=1, stride=1, padding=0),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 256, kernel_size=1, stride=1, padding=0),
        )
        self.scale2_detect_conv = nn.Sequential(
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, self.output_channels, kernel_size=1, stride=1, padding=0)
        )

        # Upsample for scale 3
        self.scale_26_upsample = nn.Sequential(
            ConvBlock(256, 128, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # ==== Scale 3: 52x52 (large scale - detects small objects) ====
        # Input: 128 (from upsample) + 256 (from backbone) = 384 channels
        self.scale3_conv = nn.Sequential(
            ConvBlock(384, 128, kernel_size=1, stride=1, padding=0),
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 128, kernel_size=1, stride=1, padding=0),
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 128, kernel_size=1, stride=1, padding=0),
        )
        self.scale3_detect_conv = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, self.output_channels, kernel_size=1, stride=1, padding=0)
        )
        ################################################
    def forward(self, features):
        """
        Args:
            features: Tuple of (feat_13x13, feat_26x26, feat_52x52)

        Returns:
            Tuple of (pred_13x13, pred_26x26, pred_52x52)
            Each prediction shape: (B, H, W, num_anchors * (5 + num_classes))
        """
        feat_13, feat_26, feat_52 = features
        
        # Scale 1: 13x13
        x1 = self.scale1_conv(feat_13)
        pred_13 = self.scale1_detect_conv(x1)
        # Prepare for scale 2
        x1_up = self.scale_13_upsample(x1)
        # Scale 2: 26x26
        x2 = torch.cat([x1_up, feat_26], dim=1)
        x2 = self.scale2_conv(x2)
        pred_26 = self.scale2_detect_conv(x2)

        # Prepare for scale 3
        x2_up = self.scale_26_upsample(x2)

        # Scale 3: 52x52
        x3 = torch.cat([x2_up, feat_52], dim=1)
        x3 = self.scale3_conv(x3)
        pred_52 = self.scale3_detect_conv(x3)
        
        # Reshape predictions: (B, C, H, W) -> (B, H, W, C)
        pred_13 = pred_13.permute(0, 2, 3, 1).contiguous()
        pred_26 = pred_26.permute(0, 2, 3, 1).contiguous()
        pred_52 = pred_52.permute(0, 2, 3, 1).contiguous()

        return pred_13, pred_26, pred_52
# ============================================================================
# NMS for inference
# ============================================================================
def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    
    Following the original PyTorch-YOLOv3 implementation logic.

    Args:
        prediction: (batch_size, num_boxes, 5 + num_classes)
                   where 5 = (x, y, w, h, objectness)
                   num_boxes = total boxes from all scales
        conf_thres: object confidence threshold
        nms_thres: IOU threshold for NMS

    Returns:
        detections: List of detections for each image in batch
                   Each detection: (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        
        # If none remain process next image
        if not image_pred.size(0):
            continue
        
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5:], 1, keepdim=True)
        
        # Detections ordered as (x, y, w, h, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            # Get detection with highest confidence
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output

def bbox_iou(box1, box2):
    """
    Compute IoU between two sets of boxes.
    Boxes are in (x, y, w, h) format.
    """
    # Get coordinates of bounding boxes
    b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
    b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
    b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2

    # Get intersection rectangle
    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area.unsqueeze(1) + b2_area - inter_area + 1e-16)

    return iou


# ============================================================================
# Object Detection Model
# ============================================================================
from src.config import ANCHORS
class ODModel(nn.Module):
    """
    Complete YOLO v3 Object Detection Model.
    Combines DenseNet backbone with YOLO v3 detection head.
    """
    def __init__(self, num_classes=20, num_anchors=3, pretrained=True, nms_thres=0.4, conf_thres=0.5):
        super(ODModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        #################YOU CAN CHANGE TO ANOTHER BACKBONE########################
        self.backbone = Backbone(pretrained=pretrained, model_name="timm/darknet53.c2ns_in1k")
        ###########################################################################
        
        self.head = YOLOv3Head(num_classes=num_classes, num_anchors=num_anchors)
        self.anchors = ANCHORS
        self.nms_thres = nms_thres
        self.conf_thres = conf_thres
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        # Get predictions at 3 scales
        predictions = self.head(features)

        return predictions
    def inference(self, x, conf_thres=None, nms_thres=None):
        """
        Run inference with NMS.
        Following PyTorch-YOLOv3 Darknet forward logic.

        Args:
            x: Input images tensor [B, 3, H, W]
            conf_thres: Confidence threshold (default: use model's conf_thres)
            nms_thres: NMS IoU threshold (default: use model's nms_thres)

        Returns:
            List of detections per image, each detection: (x, y, w, h, obj_conf, class_conf, class_pred)
        """
        # Use model defaults if not specified
        if conf_thres is None:
            conf_thres = self.conf_thres
        if nms_thres is None:
            nms_thres = self.nms_thres

        self.eval()
        with torch.no_grad():
            features = self.backbone(x)
            predictions = self.head(features)
            # Get raw predictions at 3 scales
            pred_13, pred_26, pred_52 = predictions

            # Transform predictions (apply sigmoid/exp like YOLOLayer)
            pred_13 = self._transform_predictions(pred_13, self.anchors[0])
            pred_26 = self._transform_predictions(pred_26, self.anchors[1])
            pred_52 = self._transform_predictions(pred_52, self.anchors[2])

            # Concatenate all scales along dimension 1 (like Darknet forward)
            all_predictions = torch.cat([pred_13, pred_26, pred_52], dim=1)

            # Apply NMS with specified thresholds
            output = non_max_suppression(all_predictions, conf_thres, nms_thres)
            return output
    def _transform_predictions(self, pred, anchors):
        """
        Transform raw predictions to actual bbox coordinates.
        Following PyTorch-YOLOv3 YOLOLayer logic.

        Args:
            pred: (B, H, W, num_anchors * (5 + num_classes))
            anchors: List of (w, h) tuples

        Returns:
            Transformed predictions with actual coordinates
        """
        batch_size = pred.size(0)
        grid_size = pred.size(1)
        img_size = 416  # Standard YOLO input size
        stride = img_size // grid_size

        # Reshape: (B, H, W, num_anchors * (5 + C)) -> (B, num_anchors, 5 + C, H, W)
        pred = pred.view(batch_size, grid_size, grid_size, self.num_anchors, 5 + self.num_classes)
        pred = pred.permute(0, 3, 4, 1, 2).contiguous()  # (B, num_anchors, 5+C, H, W)
        
        # Reshape to (B, num_anchors, H, W, 5+C)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()

        # Create grid
        grid_x = torch.arange(grid_size, dtype=torch.float, device=pred.device)
        grid_y = torch.arange(grid_size, dtype=torch.float, device=pred.device)
        yv, xv = torch.meshgrid([grid_y, grid_x], indexing='ij')
        grid = torch.stack((xv, yv), 2).view(1, 1, grid_size, grid_size, 2).float()

        # Anchor grid
        anchors_tensor = torch.tensor(anchors, dtype=torch.float, device=pred.device)
        anchor_grid = anchors_tensor.view(1, self.num_anchors, 1, 1, 2)

        # Transform predictions (following YOLOLayer forward in inference mode)
        pred_boxes = pred[..., :4].clone()
        pred_boxes[..., 0:2] = (pred_boxes[..., 0:2].sigmoid() + grid) * stride  # xy
        pred_boxes[..., 2:4] = torch.exp(pred_boxes[..., 2:4]) * anchor_grid * img_size  # wh
        pred_conf = pred[..., 4].sigmoid()
        pred_cls = pred[..., 5:].sigmoid()

        # Reshape to (B, num_boxes, 5+C)
        pred_boxes = pred_boxes.view(batch_size, -1, 4)
        pred_conf = pred_conf.view(batch_size, -1, 1)
        pred_cls = pred_cls.view(batch_size, -1, self.num_classes)

        # Concatenate
        output = torch.cat([pred_boxes, pred_conf, pred_cls], dim=-1)

        return output


def getODmodel(pretrained=True):
    """
    Factory function to create YOLO v3 model with DenseNet backbone.
    Renamed to keep compatibility with existing code.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Returns:
        ODModel: Complete YOLO v3 object detection model
    """
    model = ODModel(num_classes=20, num_anchors=3, pretrained=pretrained)
    return model
