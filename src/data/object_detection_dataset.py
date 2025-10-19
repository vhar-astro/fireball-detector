"""
Object Detection Dataset for YOLO format annotations.

This module provides a PyTorch Dataset class for loading images and bounding box
annotations in YOLO format (as created by the image-classification app).

YOLO Format:
- Images in: annotated_images/images/
- Labels in: annotated_images/labels/ (one .txt file per image)
- Classes in: annotated_images/classes.txt
- Format per line: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np


class ObjectDetectionDataset(Dataset):
    """
    PyTorch Dataset for object detection with YOLO format annotations.
    
    Args:
        root_dir (str): Path to the annotated_images directory
        transform (callable, optional): Optional transform to be applied on images
        target_transform (callable, optional): Optional transform for targets
    """
    
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')
        self.classes_file = os.path.join(root_dir, 'classes.txt')
        self.transform = transform
        self.target_transform = target_transform
        
        # Load class names
        self.classes = self._load_classes()
        self.num_classes = len(self.classes)
        
        # Get list of image files
        self.image_files = self._get_image_files()
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
        
        print(f"Loaded {len(self.image_files)} images with {self.num_classes} classes")
        print(f"Classes: {self.classes}")
    
    def _load_classes(self):
        """Load class names from classes.txt file."""
        if not os.path.exists(self.classes_file):
            raise FileNotFoundError(f"Classes file not found: {self.classes_file}")
        
        with open(self.classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        
        return classes
    
    def _get_image_files(self):
        """Get list of all image files in the images directory."""
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        
        for fname in os.listdir(self.images_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_extensions:
                image_files.append(fname)
        
        return sorted(image_files)
    
    def _load_annotations(self, image_filename):
        """
        Load YOLO format annotations for a given image.
        
        Args:
            image_filename (str): Name of the image file
            
        Returns:
            dict: Dictionary containing boxes, labels, and other metadata
        """
        # Get corresponding label file
        base_name = os.path.splitext(image_filename)[0]
        label_file = os.path.join(self.labels_dir, base_name + '.txt')
        
        boxes = []
        labels = []
        
        # If label file exists, parse it
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"Warning: Invalid annotation line in {label_file}: {line}")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert from YOLO format (center, normalized) to corner format
                        # We'll keep normalized coordinates for now
                        boxes.append([x_center, y_center, width, height])
                        labels.append(class_id)
                    except ValueError as e:
                        print(f"Warning: Error parsing annotation in {label_file}: {line} - {e}")
                        continue
        
        return {
            'boxes': boxes,
            'labels': labels
        }
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Returns:
            tuple: (image, target) where target is a dict containing:
                - boxes: FloatTensor[N, 4] in format [x_center, y_center, width, height] (normalized)
                - labels: Int64Tensor[N] class labels
                - image_id: int
        """
        # Load image
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        image = Image.open(img_path).convert('RGB')
        
        # Get original image size
        orig_width, orig_height = image.size
        
        # Load annotations
        annotations = self._load_annotations(img_filename)
        
        # Convert to tensors
        boxes = annotations['boxes']
        labels = annotations['labels']
        
        # Handle images with no annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dict
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'orig_size': torch.tensor([orig_height, orig_width])
        }
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target


def yolo_to_xyxy(boxes, img_width, img_height):
    """
    Convert YOLO format boxes to xyxy format.
    
    Args:
        boxes: Tensor[N, 4] in format [x_center, y_center, width, height] (normalized 0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Tensor[N, 4] in format [x1, y1, x2, y2] (absolute coordinates)
    """
    if len(boxes) == 0:
        return boxes
    
    # Convert from normalized to absolute coordinates
    boxes_abs = boxes.clone()
    boxes_abs[:, 0] = boxes[:, 0] * img_width   # x_center
    boxes_abs[:, 1] = boxes[:, 1] * img_height  # y_center
    boxes_abs[:, 2] = boxes[:, 2] * img_width   # width
    boxes_abs[:, 3] = boxes[:, 3] * img_height  # height
    
    # Convert from center format to corner format
    x1 = boxes_abs[:, 0] - boxes_abs[:, 2] / 2
    y1 = boxes_abs[:, 1] - boxes_abs[:, 3] / 2
    x2 = boxes_abs[:, 0] + boxes_abs[:, 2] / 2
    y2 = boxes_abs[:, 1] + boxes_abs[:, 3] / 2
    
    return torch.stack([x1, y1, x2, y2], dim=1)


def xyxy_to_yolo(boxes, img_width, img_height):
    """
    Convert xyxy format boxes to YOLO format.
    
    Args:
        boxes: Tensor[N, 4] in format [x1, y1, x2, y2] (absolute coordinates)
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Tensor[N, 4] in format [x_center, y_center, width, height] (normalized 0-1)
    """
    if len(boxes) == 0:
        return boxes
    
    # Convert from corner format to center format
    x_center = (boxes[:, 0] + boxes[:, 2]) / 2
    y_center = (boxes[:, 1] + boxes[:, 3]) / 2
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    
    # Normalize
    x_center = x_center / img_width
    y_center = y_center / img_height
    width = width / img_width
    height = height / img_height
    
    return torch.stack([x_center, y_center, width, height], dim=1)


def get_transform(train=True):
    """
    Get image transforms for training or validation.
    
    Args:
        train (bool): Whether this is for training (enables augmentation)
        
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    transform_list = []
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalize using ImageNet stats
    transform_list.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))
    
    return transforms.Compose(transform_list)


def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable number of boxes.
    
    Args:
        batch: List of (image, target) tuples
        
    Returns:
        tuple: (images, targets) where images is a batched tensor and targets is a list
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    
    return images, targets


def get_dataloaders(root_dir, batch_size=4, val_split=0.2, num_workers=4):
    """
    Create training and validation DataLoaders.
    
    Args:
        root_dir (str): Path to annotated_images directory
        batch_size (int): Batch size for training
        val_split (float): Fraction of data to use for validation
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create dataset
    dataset = ObjectDetectionDataset(
        root_dir=root_dir,
        transform=get_transform(train=True)
    )
    
    # Split into train and validation
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Train dataset: {train_size} images")
    print(f"Validation dataset: {val_size} images")
    
    return train_loader, val_loader

