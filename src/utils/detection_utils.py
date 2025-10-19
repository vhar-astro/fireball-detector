"""
Shared utilities for object detection inference.

This module contains common functions used by detect.py, watcher_object_detection.py,
and scanner_object_detection.py to avoid code duplication.
"""

import os
import shutil
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# Color palette for visualization
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
]


def load_classes(classes_file):
    """
    Load class names from classes.txt file.
    
    Args:
        classes_file (str): Path to classes.txt
        
    Returns:
        list: List of class names
    """
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"Classes file not found: {classes_file}")
    
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    
    return classes


def preprocess_image(image_path):
    """
    Preprocess an image for inference.
    
    Args:
        image_path (str): Path to image
        
    Returns:
        tuple: (preprocessed_tensor, original_image)
    """
    image = Image.open(image_path).convert('RGB')
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image_tensor = transform(image)
    
    return image_tensor, image


def visualize_predictions(image, predictions, class_names, score_threshold=0.5):
    """
    Visualize predictions on an image.
    
    Args:
        image (PIL.Image): Original image
        predictions (dict): Predictions with 'boxes', 'labels', 'scores'
        class_names (list): List of class names
        score_threshold (float): Minimum score to display
        
    Returns:
        PIL.Image: Image with drawn bounding boxes
    """
    # Create a copy of the image
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Draw each prediction
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    
    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue
        
        # Get box coordinates
        x1, y1, x2, y2 = box
        
        # Get color for this class
        color = COLORS[label % len(COLORS)]
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        label_text = f"{class_name}: {score:.2f}"
        
        # Draw text background
        text_bbox = draw.textbbox((x1, y1), label_text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1), label_text, fill=(255, 255, 255), font=font)
    
    return img_draw


def organize_image_by_class(image_path, predictions, class_names, output_dir, action='nothing', logger=None):
    """
    Organize image into class-specific subdirectories based on detections.

    Args:
        image_path (str): Path to the original image
        predictions (dict): Predictions with 'boxes', 'labels', 'scores'
        class_names (list): List of class names
        output_dir (str): Base output directory
        action (str): 'copy', 'move', or 'nothing'
        logger: Optional logger for logging messages

    Returns:
        list: List of classes detected in the image
    """
    if action == 'nothing':
        return []

    # Get unique detected classes
    labels = predictions['labels'].cpu().numpy()
    detected_classes = set()

    for label in labels:
        if label < len(class_names):
            detected_classes.add(class_names[label])

    # If no detections, return empty list
    if not detected_classes:
        return []

    # Organize image into class folders
    filename = os.path.basename(image_path)
    organized_classes = []

    for class_name in detected_classes:
        # Create class subdirectory
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Destination path
        dest_path = os.path.join(class_dir, filename)

        try:
            if action == 'copy':
                shutil.copy2(image_path, dest_path)
                organized_classes.append(class_name)
            elif action == 'move':
                # For move, only move once (to the first class folder)
                # For subsequent classes, copy instead
                if len(organized_classes) == 0:
                    shutil.move(image_path, dest_path)
                else:
                    # Copy from the first destination
                    first_dest = os.path.join(output_dir, organized_classes[0], filename)
                    shutil.copy2(first_dest, dest_path)
                organized_classes.append(class_name)
        except Exception as e:
            msg = f"Warning: Failed to {action} image to {class_dir}: {e}"
            if logger:
                logger.warning(msg)
            else:
                print(f"  {msg}")
            continue

    return organized_classes


def process_single_image(model, image_path, class_names, device, score_threshold=0.5,
                        output_dir='detections', action='nothing', save_visualization=True,
                        logger=None):
    """
    Process a single image: run detection, save visualization, and organize by class.
    
    This is a simplified version of detect_image() for use in watcher and scanner scripts.
    
    Args:
        model: Trained model
        image_path (str): Path to image
        class_names (list): List of class names
        device: Device to run on
        score_threshold (float): Minimum score threshold
        output_dir (str): Base output directory
        action (str): 'copy', 'move', or 'nothing'
        save_visualization (bool): Whether to save visualization image
        logger: Optional logger for logging messages
        
    Returns:
        tuple: (num_detections, detected_classes)
    """
    try:
        # Preprocess image
        image_tensor, original_image = preprocess_image(image_path)
        image_tensor = image_tensor.to(device)
        
        # Run inference
        model.eval()
        with torch.no_grad():
            predictions = model.predict([image_tensor], score_threshold=score_threshold)[0]
        
        # Extract prediction data
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        
        num_detections = len(boxes)
        
        # Save visualization if requested
        if save_visualization and num_detections > 0:
            img_with_boxes = visualize_predictions(original_image, predictions, class_names, score_threshold)
            filename = os.path.basename(image_path)
            save_path = os.path.join(output_dir, f"detected_{filename}")
            img_with_boxes.save(save_path)
        
        # Organize image by detected class
        detected_classes = organize_image_by_class(
            image_path, predictions, class_names, output_dir, action, logger
        )
        
        return num_detections, detected_classes
        
    except Exception as e:
        if logger:
            logger.error(f"Error processing {image_path}: {e}")
        else:
            print(f"Error processing {image_path}: {e}")
        raise


def load_processed_files(state_file):
    """
    Load processed file paths from a state file.
    
    Args:
        state_file (str): Path to the state file
        
    Returns:
        set: Set of processed file paths
    """
    try:
        with open(state_file, 'r') as f:
            return {line.strip() for line in f}
    except FileNotFoundError:
        return set()


def log_processed_file(state_file, file_path):
    """
    Log a processed file path to the state file.
    
    Args:
        state_file (str): Path to the state file
        file_path (str): File path to add to the state file
    """
    with open(state_file, 'a') as f:
        f.write(f"{file_path}\n")
        f.flush()


def is_image_file(filename):
    """
    Check if a file is an image based on extension.
    
    Args:
        filename (str): Filename to check
        
    Returns:
        bool: True if file is an image
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    ext = os.path.splitext(filename)[1].lower()
    return ext in valid_extensions

