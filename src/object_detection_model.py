"""
Faster R-CNN Object Detection Model.

This module provides a Faster R-CNN model implementation using PyTorch's
torchvision.models.detection module with customizable backbone and parameters.
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.models import resnet50, resnet101, mobilenet_v3_large
from torchvision.ops import MultiScaleRoIAlign
from src.config import Config


def get_backbone(backbone_name='resnet50', trainable_layers=3):
    """
    Get a backbone network for Faster R-CNN.
    
    Args:
        backbone_name (str): Name of the backbone ('resnet50', 'resnet101', 'mobilenet_v3')
        trainable_layers (int): Number of trainable layers (0-5)
        
    Returns:
        nn.Module: Backbone network with FPN
    """
    if backbone_name == 'resnet50':
        backbone = resnet50(weights=None)  # Train from scratch
        backbone_out_channels = 2048
    elif backbone_name == 'resnet101':
        backbone = resnet101(weights=None)
        backbone_out_channels = 2048
    elif backbone_name == 'mobilenet_v3':
        backbone = mobilenet_v3_large(weights=None)
        backbone_out_channels = 960
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    # Remove the classification head
    if 'resnet' in backbone_name:
        backbone = nn.Sequential(*list(backbone.children())[:-2])
    else:  # mobilenet
        backbone = backbone.features
    
    # Freeze layers if needed
    if trainable_layers < 5:
        layers_to_train = []
        if 'resnet' in backbone_name:
            # ResNet has: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
            all_layers = list(backbone.children())
            # Freeze early layers
            for i, layer in enumerate(all_layers[:-trainable_layers]):
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            # For MobileNet, freeze early layers
            for i, layer in enumerate(backbone[:-trainable_layers]):
                for param in layer.parameters():
                    param.requires_grad = False
    
    backbone.out_channels = backbone_out_channels
    
    return backbone


def create_faster_rcnn_model(num_classes, 
                             backbone_name='resnet50',
                             trainable_backbone_layers=3,
                             min_size=800,
                             max_size=1333,
                             **kwargs):
    """
    Create a Faster R-CNN model with custom configuration.
    
    Args:
        num_classes (int): Number of classes (including background)
        backbone_name (str): Backbone network name
        trainable_backbone_layers (int): Number of trainable backbone layers
        min_size (int): Minimum image size
        max_size (int): Maximum image size
        **kwargs: Additional arguments for model configuration
        
    Returns:
        FasterRCNN: Configured Faster R-CNN model
    """
    # Use the built-in Faster R-CNN with ResNet50 FPN as base
    # We'll modify it to train from scratch
    model = fasterrcnn_resnet50_fpn(
        weights=None,  # Train from scratch
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
        **kwargs
    )
    
    # Replace the classifier head with a new one for our number of classes
    # num_classes includes background, so we add 1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    
    return model


def create_faster_rcnn_from_config(num_classes):
    """
    Create a Faster R-CNN model using settings from Config.
    
    Args:
        num_classes (int): Number of object classes (not including background)
        
    Returns:
        FasterRCNN: Configured Faster R-CNN model
    """
    # RPN parameters
    rpn_params = {
        'rpn_pre_nms_top_n_train': Config.OD_RPN_PRE_NMS_TOP_N_TRAIN,
        'rpn_pre_nms_top_n_test': Config.OD_RPN_PRE_NMS_TOP_N_TEST,
        'rpn_post_nms_top_n_train': Config.OD_RPN_POST_NMS_TOP_N_TRAIN,
        'rpn_post_nms_top_n_test': Config.OD_RPN_POST_NMS_TOP_N_TEST,
        'rpn_nms_thresh': Config.OD_RPN_NMS_THRESH,
        'rpn_fg_iou_thresh': Config.OD_RPN_FG_IOU_THRESH,
        'rpn_bg_iou_thresh': Config.OD_RPN_BG_IOU_THRESH,
    }
    
    # Box parameters
    box_params = {
        'box_score_thresh': Config.OD_BOX_SCORE_THRESH,
        'box_nms_thresh': Config.OD_BOX_NMS_THRESH,
        'box_detections_per_img': Config.OD_BOX_DETECTIONS_PER_IMG,
        'box_fg_iou_thresh': Config.OD_BOX_FG_IOU_THRESH,
        'box_bg_iou_thresh': Config.OD_BOX_BG_IOU_THRESH,
    }
    
    # Combine all parameters
    model_params = {**rpn_params, **box_params}
    
    # Create model
    model = create_faster_rcnn_model(
        num_classes=num_classes,
        backbone_name=Config.OD_BACKBONE,
        trainable_backbone_layers=Config.OD_TRAINABLE_BACKBONE_LAYERS,
        min_size=Config.OD_MIN_SIZE,
        max_size=Config.OD_MAX_SIZE,
        **model_params
    )
    
    return model


class FasterRCNNWrapper(nn.Module):
    """
    Wrapper for Faster R-CNN to handle YOLO format inputs and provide utilities.
    """
    
    def __init__(self, num_classes):
        """
        Initialize the Faster R-CNN wrapper.
        
        Args:
            num_classes (int): Number of object classes (not including background)
        """
        super(FasterRCNNWrapper, self).__init__()
        self.num_classes = num_classes
        self.model = create_faster_rcnn_from_config(num_classes)
    
    def forward(self, images, targets=None):
        """
        Forward pass.
        
        Args:
            images: List of tensors or batched tensor [B, C, H, W]
            targets: List of dicts with 'boxes' and 'labels' (during training)
            
        Returns:
            During training: dict with losses
            During inference: list of dicts with 'boxes', 'labels', 'scores'
        """
        # Convert batched tensor to list of tensors if needed
        if isinstance(images, torch.Tensor):
            images = [img for img in images]
        
        # Convert YOLO format targets to xyxy format if needed
        if targets is not None:
            from src.data.object_detection_dataset import yolo_to_xyxy
            
            converted_targets = []
            for target in targets:
                boxes = target['boxes']
                labels = target['labels']
                orig_size = target['orig_size']
                
                # Convert from YOLO format to xyxy
                if len(boxes) > 0:
                    boxes = yolo_to_xyxy(boxes, orig_size[1].item(), orig_size[0].item())
                
                converted_target = {
                    'boxes': boxes,
                    'labels': labels + 1  # Add 1 because class 0 is background in Faster R-CNN
                }
                converted_targets.append(converted_target)
            
            targets = converted_targets
        
        # Forward pass through the model
        return self.model(images, targets)
    
    def predict(self, images, score_threshold=0.5):
        """
        Make predictions on images.
        
        Args:
            images: List of tensors or batched tensor
            score_threshold: Minimum score for predictions
            
        Returns:
            list: List of dicts with filtered predictions
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(images)
        
        # Filter predictions by score threshold
        filtered_predictions = []
        for pred in predictions:
            mask = pred['scores'] > score_threshold
            filtered_pred = {
                'boxes': pred['boxes'][mask],
                'labels': pred['labels'][mask] - 1,  # Subtract 1 to get original class IDs
                'scores': pred['scores'][mask]
            }
            filtered_predictions.append(filtered_pred)
        
        return filtered_predictions
    
    def save(self, path):
        """Save model state dict."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path, device='cpu'):
        """Load model state dict."""
        self.model.load_state_dict(torch.load(path, map_location=device))
    
    def export_onnx(self, path, input_size=(3, 800, 800)):
        """
        Export model to ONNX format.
        
        Args:
            path (str): Path to save ONNX model
            input_size (tuple): Input size (C, H, W)
        """
        self.eval()
        dummy_input = torch.randn(1, *input_size)
        
        torch.onnx.export(
            self.model,
            [dummy_input],
            path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['boxes', 'labels', 'scores'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'boxes': {0: 'batch_size'},
                'labels': {0: 'batch_size'},
                'scores': {0: 'batch_size'}
            }
        )
        print(f"Model exported to ONNX: {path}")
    
    def export_torchscript(self, path):
        """
        Export model to TorchScript format.
        
        Args:
            path (str): Path to save TorchScript model
        """
        self.eval()
        scripted_model = torch.jit.script(self.model)
        scripted_model.save(path)
        print(f"Model exported to TorchScript: {path}")


def load_model(path, num_classes, device='cpu'):
    """
    Load a trained Faster R-CNN model.
    
    Args:
        path (str): Path to model checkpoint
        num_classes (int): Number of classes
        device (str): Device to load model on
        
    Returns:
        FasterRCNNWrapper: Loaded model
    """
    model = FasterRCNNWrapper(num_classes)
    model.load(path, device)
    model.to(device)
    return model

