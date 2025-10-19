"""
Metrics for object detection evaluation.

This module provides functions to calculate:
- IoU (Intersection over Union)
- Precision and Recall
- mAP (mean Average Precision)
"""

import torch
import numpy as np
from collections import defaultdict


def box_iou(boxes1, boxes2):
    """
    Calculate IoU between two sets of boxes.
    
    Args:
        boxes1: Tensor[N, 4] in format [x1, y1, x2, y2]
        boxes2: Tensor[M, 4] in format [x1, y1, x2, y2]
        
    Returns:
        Tensor[N, M]: IoU values between each pair of boxes
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Calculate intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # Calculate union
    union = area1[:, None] + area2 - inter
    
    # Calculate IoU
    iou = inter / union
    
    return iou


def calculate_iou_single(box1, box2):
    """
    Calculate IoU between two single boxes.
    
    Args:
        box1: Tensor[4] or list in format [x1, y1, x2, y2]
        box2: Tensor[4] or list in format [x1, y1, x2, y2]
        
    Returns:
        float: IoU value
    """
    if not isinstance(box1, torch.Tensor):
        box1 = torch.tensor(box1)
    if not isinstance(box2, torch.Tensor):
        box2 = torch.tensor(box2)
    
    # Calculate areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    union = area1 + area2 - intersection
    
    return (intersection / union).item()


def match_predictions_to_ground_truth(pred_boxes, pred_labels, pred_scores,
                                      gt_boxes, gt_labels, iou_threshold=0.5):
    """
    Match predicted boxes to ground truth boxes based on IoU.
    
    Args:
        pred_boxes: Tensor[N, 4] predicted boxes in [x1, y1, x2, y2] format
        pred_labels: Tensor[N] predicted class labels
        pred_scores: Tensor[N] prediction confidence scores
        gt_boxes: Tensor[M, 4] ground truth boxes in [x1, y1, x2, y2] format
        gt_labels: Tensor[M] ground truth class labels
        iou_threshold: float, IoU threshold for matching
        
    Returns:
        dict: Contains 'tp', 'fp', 'scores', 'num_gt' for each class
    """
    results = defaultdict(lambda: {'tp': [], 'fp': [], 'scores': [], 'num_gt': 0})
    
    # Count ground truth boxes per class
    for label in gt_labels:
        results[label.item()]['num_gt'] += 1
    
    if len(pred_boxes) == 0:
        return results
    
    if len(gt_boxes) == 0:
        # All predictions are false positives
        for pred_label, pred_score in zip(pred_labels, pred_scores):
            results[pred_label.item()]['fp'].append(1)
            results[pred_label.item()]['tp'].append(0)
            results[pred_label.item()]['scores'].append(pred_score.item())
        return results
    
    # Calculate IoU between all predictions and ground truths
    ious = box_iou(pred_boxes, gt_boxes)
    
    # Track which ground truth boxes have been matched
    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
    
    # Sort predictions by score (descending)
    sorted_indices = torch.argsort(pred_scores, descending=True)
    
    for idx in sorted_indices:
        pred_label = pred_labels[idx].item()
        pred_score = pred_scores[idx].item()
        
        # Find best matching ground truth box
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx in range(len(gt_boxes)):
            if gt_matched[gt_idx]:
                continue
            
            if gt_labels[gt_idx].item() != pred_label:
                continue
            
            iou = ious[idx, gt_idx].item()
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if match is good enough
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            # True positive
            results[pred_label]['tp'].append(1)
            results[pred_label]['fp'].append(0)
            gt_matched[best_gt_idx] = True
        else:
            # False positive
            results[pred_label]['tp'].append(0)
            results[pred_label]['fp'].append(1)
        
        results[pred_label]['scores'].append(pred_score)
    
    return results


def calculate_precision_recall(tp, fp, num_gt):
    """
    Calculate precision and recall from true positives and false positives.
    
    Args:
        tp: list of true positive indicators (0 or 1)
        fp: list of false positive indicators (0 or 1)
        num_gt: int, total number of ground truth boxes
        
    Returns:
        tuple: (precision, recall) arrays
    """
    if len(tp) == 0:
        return np.array([]), np.array([])
    
    # Cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Calculate precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / max(num_gt, 1)
    
    return precision, recall


def calculate_ap(precision, recall):
    """
    Calculate Average Precision using the 11-point interpolation method.
    
    Args:
        precision: numpy array of precision values
        recall: numpy array of recall values
        
    Returns:
        float: Average Precision
    """
    if len(precision) == 0:
        return 0.0
    
    # Add sentinel values at the beginning and end
    precision = np.concatenate(([0], precision, [0]))
    recall = np.concatenate(([0], recall, [1]))
    
    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    
    return ap


def calculate_map(all_results, num_classes):
    """
    Calculate mean Average Precision across all classes.
    
    Args:
        all_results: list of dicts from match_predictions_to_ground_truth
        num_classes: int, total number of classes
        
    Returns:
        dict: Contains 'mAP', 'AP_per_class', 'precision_per_class', 'recall_per_class'
    """
    # Aggregate results across all images
    aggregated = defaultdict(lambda: {'tp': [], 'fp': [], 'scores': [], 'num_gt': 0})
    
    for results in all_results:
        for class_id, class_results in results.items():
            aggregated[class_id]['tp'].extend(class_results['tp'])
            aggregated[class_id]['fp'].extend(class_results['fp'])
            aggregated[class_id]['scores'].extend(class_results['scores'])
            aggregated[class_id]['num_gt'] += class_results['num_gt']
    
    # Calculate AP for each class
    ap_per_class = {}
    precision_per_class = {}
    recall_per_class = {}
    
    for class_id in range(num_classes):
        if class_id not in aggregated or aggregated[class_id]['num_gt'] == 0:
            ap_per_class[class_id] = 0.0
            precision_per_class[class_id] = 0.0
            recall_per_class[class_id] = 0.0
            continue
        
        # Sort by scores
        class_data = aggregated[class_id]
        sorted_indices = np.argsort(class_data['scores'])[::-1]
        
        tp = np.array(class_data['tp'])[sorted_indices]
        fp = np.array(class_data['fp'])[sorted_indices]
        
        # Calculate precision and recall
        precision, recall = calculate_precision_recall(tp, fp, class_data['num_gt'])
        
        # Calculate AP
        ap = calculate_ap(precision, recall)
        
        ap_per_class[class_id] = ap
        precision_per_class[class_id] = precision[-1] if len(precision) > 0 else 0.0
        recall_per_class[class_id] = recall[-1] if len(recall) > 0 else 0.0
    
    # Calculate mAP
    mAP = np.mean(list(ap_per_class.values()))
    
    return {
        'mAP': mAP,
        'AP_per_class': ap_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class
    }


def evaluate_model(model, dataloader, device, num_classes, iou_threshold=0.5):
    """
    Evaluate object detection model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: torch device
        num_classes: int, number of classes
        iou_threshold: float, IoU threshold for matching
        
    Returns:
        dict: Evaluation metrics including mAP, precision, recall
    """
    model.eval()
    all_results = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            
            # Get predictions
            predictions = model(images)
            
            # Process each image in the batch
            for i, (pred, target) in enumerate(zip(predictions, targets)):
                # Convert YOLO format to xyxy if needed
                from src.data.object_detection_dataset import yolo_to_xyxy
                
                gt_boxes = target['boxes']
                gt_labels = target['labels']
                orig_size = target['orig_size']
                
                if len(gt_boxes) > 0:
                    gt_boxes = yolo_to_xyxy(gt_boxes, orig_size[1].item(), orig_size[0].item())
                
                # Extract predictions
                pred_boxes = pred['boxes']
                pred_labels = pred['labels']
                pred_scores = pred['scores']
                
                # Match predictions to ground truth
                results = match_predictions_to_ground_truth(
                    pred_boxes, pred_labels, pred_scores,
                    gt_boxes, gt_labels, iou_threshold
                )
                
                all_results.append(results)
    
    # Calculate mAP
    metrics = calculate_map(all_results, num_classes)
    
    return metrics

