"""
Object detection inference script.

This script runs inference on images using a trained Faster R-CNN model,
visualizes predictions, and supports model export to ONNX and TorchScript.

Usage:
    # Detect objects in a single image
    python -m src.detect --image path/to/image.jpg --model checkpoints/best_model.pth
    
    # Detect objects in a directory
    python -m src.detect --input_dir path/to/images/ --model checkpoints/best_model.pth
    
    # Export model
    python -m src.detect --export --model checkpoints/best_model.pth
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from src.config import Config
from src.object_detection_model import FasterRCNNWrapper, load_model


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


def detect_image(model, image_path, class_names, device, score_threshold=0.5,
                save_path=None, show=False, verbose=False):
    """
    Run detection on a single image.

    Args:
        model: Trained model
        image_path (str): Path to image
        class_names (list): List of class names
        device: Device to run on
        score_threshold (float): Minimum score threshold
        save_path (str): Path to save visualization
        show (bool): Whether to display the image
        verbose (bool): Show detailed diagnostic information

    Returns:
        dict: Predictions
    """
    # Preprocess image
    image_tensor, original_image = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    if verbose:
        print(f"\n{'='*80}")
        print(f"DIAGNOSTIC INFORMATION FOR: {os.path.basename(image_path)}")
        print(f"{'='*80}")
        print(f"Image size: {original_image.size}")
        print(f"Tensor shape: {image_tensor.shape}")
        print(f"Device: {device}")
        print(f"Score threshold: {score_threshold}")

    # Run inference - get ALL predictions (no filtering)
    model.eval()
    with torch.no_grad():
        # Get raw predictions without filtering
        raw_predictions = model.forward([image_tensor])[0]
        # Get filtered predictions
        filtered_predictions = model.predict([image_tensor], score_threshold=score_threshold)[0]

    # Extract raw prediction data
    raw_boxes = raw_predictions['boxes'].cpu().numpy()
    raw_labels = raw_predictions['labels'].cpu().numpy() - 1  # Subtract 1 to match class IDs
    raw_scores = raw_predictions['scores'].cpu().numpy()

    # Extract filtered prediction data
    filtered_boxes = filtered_predictions['boxes'].cpu().numpy()
    filtered_labels = filtered_predictions['labels'].cpu().numpy()
    filtered_scores = filtered_predictions['scores'].cpu().numpy()

    # Print diagnostic information
    if verbose:
        print(f"\n{'='*80}")
        print(f"RAW MODEL OUTPUT (before filtering)")
        print(f"{'='*80}")
        print(f"Total predictions: {len(raw_scores)}")

        if len(raw_scores) > 0:
            print(f"Score range: [{raw_scores.min():.4f}, {raw_scores.max():.4f}]")
            print(f"Mean score: {raw_scores.mean():.4f}")
            print(f"Median score: {np.median(raw_scores):.4f}")

            # Show score distribution
            print(f"\nScore distribution:")
            thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for thresh in thresholds:
                count = (raw_scores >= thresh).sum()
                print(f"  >= {thresh:.2f}: {count:4d} predictions")

            # Show per-class statistics
            print(f"\nPredictions per class (all scores):")
            for class_id in range(len(class_names)):
                class_mask = raw_labels == class_id
                class_count = class_mask.sum()
                if class_count > 0:
                    class_scores = raw_scores[class_mask]
                    print(f"  {class_names[class_id]:15s}: {class_count:4d} predictions "
                          f"(max score: {class_scores.max():.4f}, mean: {class_scores.mean():.4f})")

            # Show top predictions (even if below threshold)
            print(f"\nTop 20 predictions (regardless of threshold):")
            top_indices = np.argsort(raw_scores)[::-1][:20]
            for i, idx in enumerate(top_indices, 1):
                box = raw_boxes[idx]
                label = raw_labels[idx]
                score = raw_scores[idx]
                class_name = class_names[label] if label < len(class_names) else f"Class {label}"
                above_thresh = "✓" if score >= score_threshold else "✗"
                print(f"  {i:2d}. [{above_thresh}] {class_name:15s}: {score:.4f} "
                      f"at [{box[0]:6.1f}, {box[1]:6.1f}, {box[2]:6.1f}, {box[3]:6.1f}]")
        else:
            print("WARNING: Model produced 0 predictions!")
            print("This may indicate:")
            print("  - Model not trained properly")
            print("  - Input image preprocessing issue")
            print("  - Model architecture mismatch")

    # Print standard output
    print(f"\n{'='*80}")
    print(f"DETECTIONS FOR: {os.path.basename(image_path)}")
    print(f"{'='*80}")
    print(f"Found {len(filtered_boxes)} objects (score >= {score_threshold})")

    if len(filtered_boxes) > 0:
        for i, (box, label, score) in enumerate(zip(filtered_boxes, filtered_labels, filtered_scores)):
            class_name = class_names[label] if label < len(class_names) else f"Class {label}"
            print(f"  {i+1}. {class_name}: {score:.3f} at [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    else:
        if verbose:
            print(f"\nNo detections above threshold {score_threshold}")
            if len(raw_scores) > 0:
                print(f"However, model made {len(raw_scores)} predictions total.")
                print(f"Highest confidence: {raw_scores.max():.4f}")
                print(f"\nSuggestions:")
                print(f"  1. Lower the threshold: --score_threshold 0.1")
                print(f"  2. Train for more epochs (current model may be undertrained)")
                print(f"  3. Check if image is similar to training data")
        else:
            print(f"\nTip: Use --verbose flag to see all predictions and diagnostic info")

    # Visualize
    if save_path or show:
        img_with_boxes = visualize_predictions(original_image, filtered_predictions, class_names, score_threshold)

        if save_path:
            img_with_boxes.save(save_path)
            print(f"\nSaved visualization to: {save_path}")

        if show:
            img_with_boxes.show()

    return filtered_predictions


def detect_directory(model, input_dir, output_dir, class_names, device,
                     score_threshold=0.5, verbose=False):
    """
    Run detection on all images in a directory.

    Args:
        model: Trained model
        input_dir (str): Input directory
        output_dir (str): Output directory for visualizations
        class_names (list): List of class names
        device: Device to run on
        score_threshold (float): Minimum score threshold
        verbose (bool): Show detailed diagnostic information
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []

    for fname in os.listdir(input_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_extensions:
            image_files.append(fname)

    print(f"Found {len(image_files)} images in {input_dir}")

    # Process each image
    for i, fname in enumerate(image_files, 1):
        print(f"\nProcessing {i}/{len(image_files)}: {fname}")

        image_path = os.path.join(input_dir, fname)
        save_path = os.path.join(output_dir, f"detected_{fname}")

        try:
            detect_image(model, image_path, class_names, device,
                        score_threshold, save_path=save_path, verbose=verbose)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue

    print(f"\nProcessing complete! Results saved to: {output_dir}")


def export_model(model, num_classes, output_dir='exports'):
    """
    Export model to ONNX and TorchScript formats.
    
    Args:
        model: Trained model
        num_classes (int): Number of classes
        output_dir (str): Output directory for exports
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Export to ONNX
    onnx_path = os.path.join(output_dir, Config.OD_ONNX_PATH)
    try:
        print(f"Exporting to ONNX...")
        model.export_onnx(onnx_path)
        print(f"✓ ONNX export successful: {onnx_path}")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
    
    # Export to TorchScript
    torchscript_path = os.path.join(output_dir, Config.OD_TORCHSCRIPT_PATH)
    try:
        print(f"Exporting to TorchScript...")
        model.export_torchscript(torchscript_path)
        print(f"✓ TorchScript export successful: {torchscript_path}")
    except Exception as e:
        print(f"✗ TorchScript export failed: {e}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Run object detection inference on images.'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--classes',
        type=str,
        default=Config.OD_CLASSES_FILE,
        help=f'Path to classes.txt file (default: {Config.OD_CLASSES_FILE})'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to a single image for detection'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Path to directory of images for detection'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='detections',
        help='Output directory for visualizations (default: detections)'
    )
    parser.add_argument(
        '--score_threshold',
        type=float,
        default=0.5,
        help='Minimum score threshold for detections (default: 0.5)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image with detections'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed diagnostic information including all predictions'
    )
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export model to ONNX and TorchScript formats'
    )
    parser.add_argument(
        '--export_dir',
        type=str,
        default='exports',
        help='Directory for model exports (default: exports)'
    )
    
    args = parser.parse_args()
    
    # Verify model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    
    # Load class names
    print(f"Loading classes from {args.classes}...")
    class_names = load_classes(args.classes)
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, num_classes, device)
    print("Model loaded successfully!")
    
    # Export model if requested
    if args.export:
        export_model(model, num_classes, args.export_dir)
        return
    
    # Run detection
    if args.image:
        # Single image detection
        save_path = os.path.join(args.output_dir, f"detected_{os.path.basename(args.image)}")
        os.makedirs(args.output_dir, exist_ok=True)
        detect_image(model, args.image, class_names, device,
                    args.score_threshold, save_path, args.show, args.verbose)

    elif args.input_dir:
        # Directory detection
        detect_directory(model, args.input_dir, args.output_dir, class_names,
                        device, args.score_threshold, args.verbose)

    else:
        print("Error: Please specify either --image or --input_dir or --export")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

