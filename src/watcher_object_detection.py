"""
Object Detection Watcher Script

This script monitors a directory for new images and automatically runs object detection
on them using a trained Faster R-CNN model. It supports automatic organization of detected
images into class-specific subdirectories.

Usage:
    # Watch directory and organize images by detected class (copy mode)
    python -m src.watcher_object_detection \
        --model checkpoints/faster_rcnn_bs4_ne50.pth \
        --watch_dir incoming/ \
        --output_dir processed/ \
        --action copy \
        --score_threshold 0.5
    
    # Watch directory and move images to class folders
    python -m src.watcher_object_detection \
        --model checkpoints/faster_rcnn_bs4_ne50.pth \
        --watch_dir incoming/ \
        --output_dir organized/ \
        --action move \
        --score_threshold 0.7
    
    # Watch directory without organizing (only save visualizations)
    python -m src.watcher_object_detection \
        --model checkpoints/faster_rcnn_bs4_ne50.pth \
        --watch_dir incoming/ \
        --output_dir detections/

Features:
    - Real-time monitoring of directory for new images
    - Automatic object detection on newly added images
    - Support for copy/move/nothing action modes
    - Multi-class detection support (images appear in all relevant class folders)
    - Configurable score threshold
    - State tracking to avoid reprocessing images
    - Graceful shutdown (Ctrl+C)
    - Comprehensive logging
"""

import sys
import time
import os
import argparse
import logging
from pathlib import Path

import torch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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
STATE_FILE = "processed_files_object_detection.log"

# --- Logging Configuration ---
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "watcher_object_detection.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)


class NewImageHandler(FileSystemEventHandler):
    """Handles the event when a new image file is created."""
    
    def __init__(self, model, class_names, device, processed_files, output_dir,
                 action='nothing', score_threshold=0.5):
        """
        Initialize the event handler.
        
        Args:
            model: Trained Faster R-CNN model
            class_names (list): List of class names
            device: Device to run inference on
            processed_files (set): Set of already processed file paths
            output_dir (str): Output directory for visualizations and organized images
            action (str): 'copy', 'move', or 'nothing'
            score_threshold (float): Minimum score threshold for detections
        """
        self.model = model
        self.class_names = class_names
        self.device = device
        self.processed_files = processed_files
        self.output_dir = output_dir
        self.action = action
        self.score_threshold = score_threshold
        self.logger = logging.getLogger(__name__)
    
    def on_created(self, event):
        """Handle file creation event."""
        # Skip directories
        if event.is_directory:
            return
        
        # Check if it's an image file
        if not is_image_file(event.src_path):
            return
        
        # Skip if already processed
        if event.src_path in self.processed_files:
            self.logger.info(f"Skipping already processed file: {event.src_path}")
            return
        
        self.logger.info(f"New image detected: {os.path.basename(event.src_path)}")
        
        # Wait a moment for the file to be fully written
        time.sleep(1)
        
        try:
            # Process the image
            num_detections, detected_classes = process_single_image(
                model=self.model,
                image_path=event.src_path,
                class_names=self.class_names,
                device=self.device,
                score_threshold=self.score_threshold,
                output_dir=self.output_dir,
                action=self.action,
                save_visualization=True,
                logger=self.logger
            )
            
            # Log results
            if num_detections > 0:
                classes_str = ', '.join(detected_classes) if detected_classes else 'none'
                self.logger.info(
                    f"{os.path.basename(event.src_path)}: "
                    f"{num_detections} detections, classes: {classes_str}"
                )
                
                if self.action == 'copy':
                    self.logger.info(f"Copied to class folders: {classes_str}")
                elif self.action == 'move':
                    self.logger.info(f"Moved to class folders: {classes_str}")
            else:
                self.logger.info(f"{os.path.basename(event.src_path)}: No detections above threshold {self.score_threshold}")
            
            # Mark as processed
            log_processed_file(STATE_FILE, event.src_path)
            self.processed_files.add(event.src_path)
            
        except Exception as e:
            self.logger.error(f"Error processing {event.src_path}: {e}")


def main():
    """Main function to start the watcher service."""
    parser = argparse.ArgumentParser(
        description="Watch a directory and run object detection on new images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Watch directory with copy action
  python -m src.watcher_object_detection --model checkpoints/model.pth --watch_dir incoming/ --action copy
  
  # Watch directory with move action and custom threshold
  python -m src.watcher_object_detection --model checkpoints/model.pth --watch_dir incoming/ --action move --score_threshold 0.7
  
  # Watch directory without organizing (only save visualizations)
  python -m src.watcher_object_detection --model checkpoints/model.pth --watch_dir incoming/
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--watch_dir',
        type=str,
        required=True,
        help='Directory to watch for new images'
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
    
    args = parser.parse_args()
    
    # Validate watch directory
    if not os.path.isdir(args.watch_dir):
        logging.error(f"Error: Watch directory not found: '{args.watch_dir}'")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Ensure the state directory exists
    state_dir = Path(STATE_FILE).parent
    os.makedirs(state_dir, exist_ok=True)
    
    # Load processed files
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
    
    # Create event handler and observer
    event_handler = NewImageHandler(
        model=model,
        class_names=class_names,
        device=device,
        processed_files=processed_files,
        output_dir=args.output_dir,
        action=args.action,
        score_threshold=args.score_threshold
    )
    
    observer = Observer()
    observer.schedule(event_handler, args.watch_dir, recursive=False)
    
    logging.info(f"\n{'='*80}")
    logging.info(f"OBJECT DETECTION WATCHER STARTED")
    logging.info(f"{'='*80}")
    logging.info(f"Watch directory: {args.watch_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Score threshold: {args.score_threshold}")
    logging.info(f"Action: {args.action}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Classes: {', '.join(class_names)}")
    logging.info(f"{'='*80}")
    logging.info("Watching for new images... Press Ctrl+C to stop.")
    
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("\nShutting down watcher...")
        observer.stop()
    observer.join()
    
    logging.info("Watcher stopped.")


if __name__ == "__main__":
    main()

