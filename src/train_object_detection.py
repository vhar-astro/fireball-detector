"""
Training script for Faster R-CNN object detection model.

This script trains a Faster R-CNN model on YOLO format dataset created by
the image-classification app.

Usage:
    python -m src.train_object_detection [data_dir] [options]
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from src.config import Config
from src.object_detection_model import FasterRCNNWrapper, load_model
from src.data.object_detection_dataset import get_dataloaders
from src.utils.metrics import evaluate_model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        optimizer: Optimizer
        data_loader: Training data loader
        device: Device to train on
        epoch: Current epoch number
        print_freq: Print frequency
        
    Returns:
        dict: Training statistics
    """
    model.train()
    
    total_loss = 0
    loss_classifier = 0
    loss_box_reg = 0
    loss_objectness = 0
    loss_rpn_box_reg = 0
    
    start_time = time.time()
    
    for i, (images, targets) in enumerate(data_loader):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        
        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += losses.item()
        loss_classifier += loss_dict.get('loss_classifier', 0).item() if 'loss_classifier' in loss_dict else 0
        loss_box_reg += loss_dict.get('loss_box_reg', 0).item() if 'loss_box_reg' in loss_dict else 0
        loss_objectness += loss_dict.get('loss_objectness', 0).item() if 'loss_objectness' in loss_dict else 0
        loss_rpn_box_reg += loss_dict.get('loss_rpn_box_reg', 0).item() if 'loss_rpn_box_reg' in loss_dict else 0
        
        # Print progress
        if (i + 1) % print_freq == 0:
            elapsed = time.time() - start_time
            print(f'Epoch [{epoch}], Step [{i+1}/{len(data_loader)}], '
                  f'Loss: {losses.item():.4f}, '
                  f'Time: {elapsed:.2f}s')
            start_time = time.time()
    
    # Calculate average losses
    num_batches = len(data_loader)
    stats = {
        'total_loss': total_loss / num_batches,
        'loss_classifier': loss_classifier / num_batches,
        'loss_box_reg': loss_box_reg / num_batches,
        'loss_objectness': loss_objectness / num_batches,
        'loss_rpn_box_reg': loss_rpn_box_reg / num_batches,
    }
    
    return stats


def validate(model, data_loader, device, num_classes):
    """
    Validate the model.
    
    Args:
        model: The model to validate
        data_loader: Validation data loader
        device: Device to validate on
        num_classes: Number of classes
        
    Returns:
        dict: Validation metrics
    """
    print("Running validation...")
    metrics = evaluate_model(model, data_loader, device, num_classes, 
                            iou_threshold=Config.OD_IOU_THRESHOLD)
    
    print(f"Validation mAP: {metrics['mAP']:.4f}")
    print("AP per class:")
    for class_id, ap in metrics['AP_per_class'].items():
        print(f"  Class {class_id}: {ap:.4f}")
    
    return metrics


def train_model(data_dir, 
                num_epochs=None,
                batch_size=None,
                learning_rate=None,
                resume_from=None,
                save_dir='checkpoints'):
    """
    Train the Faster R-CNN model.
    
    Args:
        data_dir (str): Path to the annotated_images directory
        num_epochs (int): Number of epochs to train
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        resume_from (str): Path to checkpoint to resume from
        save_dir (str): Directory to save checkpoints
        
    Returns:
        FasterRCNNWrapper: Trained model
    """
    # Use config defaults if not specified
    num_epochs = num_epochs or Config.OD_NUM_EPOCHS
    batch_size = batch_size or Config.OD_BATCH_SIZE
    learning_rate = learning_rate or Config.OD_LEARNING_RATE
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    train_loader, val_loader = get_dataloaders(
        root_dir=data_dir,
        batch_size=batch_size,
        val_split=Config.OD_VAL_SPLIT,
        num_workers=Config.OD_NUM_WORKERS
    )
    
    # Get number of classes from dataset
    num_classes = train_loader.dataset.dataset.num_classes
    print(f"Number of classes: {num_classes}")
    
    # Create model
    print("Creating model...")
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        model = load_model(resume_from, num_classes, device)
        start_epoch = 1  # You can save epoch info in checkpoint if needed
    else:
        model = FasterRCNNWrapper(num_classes)
        model.to(device)
        start_epoch = 1
    
    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=learning_rate,
        momentum=Config.OD_MOMENTUM,
        weight_decay=Config.OD_WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    lr_scheduler = StepLR(
        optimizer,
        step_size=Config.OD_LR_STEP_SIZE,
        gamma=Config.OD_LR_GAMMA
    )
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    print("-" * 80)
    
    best_map = 0.0
    
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 80)
        
        # Train for one epoch
        train_stats = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            print_freq=Config.OD_PRINT_FREQ
        )
        
        # Print training statistics
        print(f"\nEpoch {epoch} Training Summary:")
        print(f"  Total Loss: {train_stats['total_loss']:.4f}")
        print(f"  Classifier Loss: {train_stats['loss_classifier']:.4f}")
        print(f"  Box Reg Loss: {train_stats['loss_box_reg']:.4f}")
        print(f"  Objectness Loss: {train_stats['loss_objectness']:.4f}")
        print(f"  RPN Box Reg Loss: {train_stats['loss_rpn_box_reg']:.4f}")
        
        # Update learning rate
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Validate
        val_metrics = validate(model, val_loader, device, num_classes)
        
        # Save checkpoint
        if epoch % Config.OD_SAVE_FREQ == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            model.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_metrics['mAP'] > best_map:
            best_map = val_metrics['mAP']
            best_model_path = os.path.join(save_dir, Config.OD_BEST_MODEL_PATH)
            model.save(best_model_path)
            print(f"New best model saved: {best_model_path} (mAP: {best_map:.4f})")
    
    # Save final model
    final_model_path = os.path.join(save_dir, Config.OD_MODEL_PATH)
    model.save(final_model_path)
    print(f"\nTraining completed!")
    print(f"Final model saved: {final_model_path}")
    print(f"Best mAP: {best_map:.4f}")
    
    return model


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Train Faster R-CNN object detection model on YOLO format dataset.'
    )
    
    parser.add_argument(
        'data_dir',
        type=str,
        nargs='?',
        default=Config.OD_DATASET_PATH,
        help=f'Path to annotated_images directory (default: {Config.OD_DATASET_PATH})'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=Config.OD_NUM_EPOCHS,
        help=f'Number of training epochs (default: {Config.OD_NUM_EPOCHS})'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=Config.OD_BATCH_SIZE,
        help=f'Batch size (default: {Config.OD_BATCH_SIZE})'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=Config.OD_LEARNING_RATE,
        help=f'Learning rate (default: {Config.OD_LEARNING_RATE})'
    )
    parser.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=Config.OD_CHECKPOINT_DIR,
        help=f'Directory to save checkpoints (default: {Config.OD_CHECKPOINT_DIR})'
    )
    
    args = parser.parse_args()
    
    # Verify data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        print(f"Please ensure the annotated_images directory exists with:")
        print(f"  - images/ subdirectory")
        print(f"  - labels/ subdirectory")
        print(f"  - classes.txt file")
        sys.exit(1)
    
    # Train model
    train_model(
        data_dir=args.data_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume_from=args.resume_from,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()

