"""
Object Detection Scanner Script

This script recursively scans a directory tree for images and runs object detection
on all found images using a trained Faster R-CNN model. It supports automatic organization
of detected images into class-specific subdirectories.

Usage:
    # Recursively scan directory and organize images (copy mode)
    python -m src.scanner_object_detection \
        --model checkpoints/faster_rcnn_bs4_ne50.pth \
        --input_dir dataset/ \
        --output_dir organized/ \
        --action copy \
        --score_threshold 0.5 \
        --recursive
    
    # Scan directory with move action
    python -m src.scanner_object_detection \
        --model checkpoints/faster_rcnn_bs4_ne50.pth \
        --input_dir dataset/ \
        --output_dir organized/ \
        --action move \
        --score_threshold 0.7 \
        --recursive
    
    # Scan directory without organizing (only save visualizations)
    python -m src.scanner_object_detection \
        --model checkpoints/faster_rcnn_bs4_ne50.pth \
        --input_dir dataset/ \
        --output_dir detections/ \
        --recursive

Features:
    - Recursive directory scanning for images
    - Automatic object detection on all found images
    - Support for copy/move/nothing action modes
    - Multi-class detection support (images appear in all relevant class folders)
    - Configurable score threshold
    - State tracking to avoid reprocessing images
    - Progress reporting and batch statistics
    - Comprehensive summary report
    - File pattern filtering
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import torch

from src.config import Config
from src.object_detection_model import FasterRCNNWrapper
from src.utils.detection_utils import (
    load_classes,
    process_single_image,
    load_processed_files,
    log_processed_file,
    is_image_file
)


# --- State File ---
STATE_FILE = "processed_files_object_detection_scanner.log"

# --- Logging Configuration ---
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "scanner_object_detection.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)


def scan_for_images(base_path, recursive=True, pattern='*'):
    """
    Scan a directory for image files.
    
    Args:
        base_path (Path): Base directory to scan
        recursive (bool): Whether to scan recursively
        pattern (str): Glob pattern for filtering files
        
    Returns:
        list: List of Path objects for found image files
    """
    image_files = []
    
    if recursive:
        # Recursively find all files
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(base_path.rglob(f'{pattern}{ext}'))
            image_files.extend(base_path.rglob(f'{pattern}{ext.upper()}'))
    else:
        # Only scan the base directory
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(base_path.glob(f'{pattern}{ext}'))
            image_files.extend(base_path.glob(f'{pattern}{ext.upper()}'))
    
    return sorted(image_files)


def main():
    """Main function to run the scanner."""
    parser = argparse.ArgumentParser(
        description="Recursively scan directories for images and run object detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recursively scan and organize images by class (copy mode)
  python -m src.scanner_object_detection --model checkpoints/model.pth --input_dir dataset/ --action copy --recursive
  
  # Scan with move action and custom threshold
  python -m src.scanner_object_detection --model checkpoints/model.pth --input_dir dataset/ --action move --score_threshold 0.7 --recursive
  
  # Scan without organizing (only save visualizations)
  python -m src.scanner_object_detection --model checkpoints/model.pth --input_dir dataset/ --recursive
  
  # Scan with file pattern filter
  python -m src.scanner_object_detection --model checkpoints/model.pth --input_dir dataset/ --pattern "meteor_*" --recursive
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory to scan for images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='detections',
        help='Output directory for visualizations and organized images (default: detections)'
    )
    parser.add_argument(
        '--classes',
        type=str,
        default=Config.OD_CLASSES_FILE,
        help=f'Path to classes file (default: {Config.OD_CLASSES_FILE})'
    )
    parser.add_argument(
        '--score_threshold',
        type=float,
        default=0.5,
        help='Minimum score threshold for detections (default: 0.5)'
    )
    parser.add_argument(
        '--action',
        type=str,
        choices=['copy', 'move', 'nothing'],
        default='nothing',
        help='Organize images by detected class: copy (copy to class folders), '
             'move (move to class folders), nothing (only save visualizations, default)'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recursively scan subdirectories'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*',
        help='File pattern filter (default: *)'
    )
    parser.add_argument(
        '--skip_processed',
        action='store_true',
        help='Skip images that have been processed before (uses state file)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        logging.error(f"Error: Input directory not found: '{args.input_dir}'")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Ensure the state directory exists
    state_dir = Path(STATE_FILE).parent
    os.makedirs(state_dir, exist_ok=True)
    
    # Load processed files if skip_processed is enabled
    processed_files = set()
    if args.skip_processed:
        processed_files = load_processed_files(STATE_FILE)
        logging.info(f"Loaded {len(processed_files)} processed files from state.")
    
    # Load classes
    logging.info(f"Loading classes from {args.classes}...")
    class_names = load_classes(args.classes)
    logging.info(f"Classes: {class_names}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load model
    logging.info(f"Loading model from {args.model}...")
    model = FasterRCNNWrapper(num_classes=len(class_names))
    model.load(args.model, device)
    model.to(device)
    model.eval()
    logging.info("Model loaded successfully!")
    
    # Scan for images
    logging.info(f"\nScanning for images in {args.input_dir}...")
    base_path = Path(args.input_dir)
    image_files = scan_for_images(base_path, recursive=args.recursive, pattern=args.pattern)
    logging.info(f"Found {len(image_files)} image files.")
    
    if len(image_files) == 0:
        logging.info("No images found. Exiting.")
        return
    
    # Print scan configuration
    logging.info(f"\n{'='*80}")
    logging.info(f"OBJECT DETECTION SCANNER")
    logging.info(f"{'='*80}")
    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Recursive: {args.recursive}")
    logging.info(f"Pattern: {args.pattern}")
    logging.info(f"Score threshold: {args.score_threshold}")
    logging.info(f"Action: {args.action}")
    logging.info(f"Skip processed: {args.skip_processed}")
    logging.info(f"Total images found: {len(image_files)}")
    logging.info(f"{'='*80}\n")
    
    # Statistics tracking
    stats = {
        'total_found': len(image_files),
        'total_processed': 0,
        'total_skipped': 0,
        'total_with_detections': 0,
        'total_without_detections': 0,
        'class_counts': defaultdict(int),
        'errors': []
    }
    
    # Process each image
    for i, image_file in enumerate(image_files, 1):
        image_path = str(image_file)
        
        # Skip if already processed
        if args.skip_processed and image_path in processed_files:
            logging.info(f"[{i}/{len(image_files)}] Skipping already processed: {image_file.name}")
            stats['total_skipped'] += 1
            continue
        
        logging.info(f"\n{'='*80}")
        logging.info(f"Processing {i}/{len(image_files)}: {image_file.name}")
        logging.info(f"{'='*80}")
        
        try:
            # Process the image
            num_detections, detected_classes = process_single_image(
                model=model,
                image_path=image_path,
                class_names=class_names,
                device=device,
                score_threshold=args.score_threshold,
                output_dir=args.output_dir,
                action=args.action,
                save_visualization=True,
                logger=logging.getLogger(__name__)
            )
            
            stats['total_processed'] += 1
            
            # Log results
            if num_detections > 0:
                stats['total_with_detections'] += 1
                classes_str = ', '.join(detected_classes) if detected_classes else 'none'
                logging.info(f"Found {num_detections} detections, classes: {classes_str}")
                
                for class_name in detected_classes:
                    stats['class_counts'][class_name] += 1
                
                if args.action == 'copy':
                    logging.info(f"Copied to class folders: {classes_str}")
                elif args.action == 'move':
                    logging.info(f"Moved to class folders: {classes_str}")
            else:
                stats['total_without_detections'] += 1
                logging.info(f"No detections above threshold {args.score_threshold}")
            
            # Mark as processed
            if args.skip_processed:
                log_processed_file(STATE_FILE, image_path)
                processed_files.add(image_path)
            
        except Exception as e:
            logging.error(f"Error processing {image_file.name}: {e}")
            stats['errors'].append((image_file.name, str(e)))
            continue
    
    # Print summary report
    logging.info(f"\n{'='*80}")
    logging.info(f"SCANNER SUMMARY")
    logging.info(f"{'='*80}")
    logging.info(f"Total images found: {stats['total_found']}")
    logging.info(f"Images processed: {stats['total_processed']}")
    if args.skip_processed:
        logging.info(f"Images skipped (already processed): {stats['total_skipped']}")
    logging.info(f"Images with detections: {stats['total_with_detections']}")
    logging.info(f"Images without detections: {stats['total_without_detections']}")
    
    if stats['class_counts']:
        logging.info(f"\nDetections per class:")
        for class_name in sorted(stats['class_counts'].keys()):
            count = stats['class_counts'][class_name]
            logging.info(f"  {class_name}: {count} images")
    
    if stats['errors']:
        logging.info(f"\nErrors encountered: {len(stats['errors'])}")
        for filename, error in stats['errors']:
            logging.info(f"  {filename}: {error}")
    
    logging.info(f"\nResults saved to: {args.output_dir}")
    
    if args.action != 'nothing' and stats['total_with_detections'] > 0:
        logging.info(f"\nOrganized images by class:")
        for class_name in sorted(stats['class_counts'].keys()):
            class_dir = os.path.join(args.output_dir, class_name)
            logging.info(f"  {class_dir}/")
    
    logging.info(f"{'='*80}")
    logging.info("Scanner finished.")


if __name__ == "__main__":
    main()

