# Object Detection Watcher & Scanner Guide

**Date**: 2025-10-19  
**Feature**: Automated object detection workflows  
**Status**: ✅ **COMPLETE**

---

## Overview

Two new scripts have been added to automate object detection workflows:

1. **`watcher_object_detection.py`** - Real-time monitoring and detection
2. **`scanner_object_detection.py`** - Recursive batch processing

Both scripts extend the functionality of `detect.py` with automation capabilities, making it easy to process large numbers of images automatically.

---

## Feature Comparison

| Feature | detect.py | watcher_object_detection.py | scanner_object_detection.py |
|---------|-----------|----------------------------|----------------------------|
| **Single image** | ✅ Yes | ❌ No | ❌ No |
| **Batch processing** | ✅ Yes (non-recursive) | ❌ No | ✅ Yes (recursive) |
| **Real-time monitoring** | ❌ No | ✅ Yes | ❌ No |
| **Recursive scanning** | ❌ No | ❌ No | ✅ Yes |
| **State tracking** | ❌ No | ✅ Yes | ✅ Yes (optional) |
| **Auto-organization** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Verbose mode** | ✅ Yes | ❌ No | ❌ No |
| **File pattern filtering** | ❌ No | ❌ No | ✅ Yes |
| **Progress reporting** | ✅ Yes | ✅ Yes (real-time) | ✅ Yes |
| **Summary statistics** | ✅ Yes | ❌ No | ✅ Yes |

---

## 1. Object Detection Watcher

### Purpose

Monitors a directory for new images and automatically runs object detection on them in real-time. Perfect for:
- **Production pipelines**: Automatically process incoming images
- **Surveillance systems**: Real-time detection on camera feeds
- **Automated workflows**: Continuous processing without manual intervention

### Key Features

- ✅ **Real-time monitoring**: Detects new images as they arrive
- ✅ **Automatic processing**: Runs detection immediately on new files
- ✅ **State tracking**: Avoids reprocessing the same image
- ✅ **Graceful shutdown**: Clean exit with Ctrl+C
- ✅ **Comprehensive logging**: File and console logging
- ✅ **Auto-organization**: Supports copy/move/nothing modes
- ✅ **Multi-class support**: Images appear in all relevant class folders

### Installation Requirements

The watcher requires the `watchdog` library:

```bash
pip install watchdog
```

### Usage

#### Basic Usage

```bash
cd fireball-detector
python3 -m src.watcher_object_detection \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --watch_dir incoming/
```

#### With Auto-Organization (Copy Mode)

```bash
python3 -m src.watcher_object_detection \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --watch_dir incoming/ \
    --output_dir processed/ \
    --action copy \
    --score_threshold 0.5
```

#### With Auto-Organization (Move Mode)

```bash
python3 -m src.watcher_object_detection \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --watch_dir incoming/ \
    --output_dir organized/ \
    --action move \
    --score_threshold 0.7
```

### Command-Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--model` | str | ✅ Yes | - | Path to trained model checkpoint |
| `--watch_dir` | str | ✅ Yes | - | Directory to watch for new images |
| `--output_dir` | str | ❌ No | `detections` | Output directory for visualizations |
| `--classes` | str | ❌ No | `Config.OD_CLASSES_FILE` | Path to classes file |
| `--score_threshold` | float | ❌ No | `0.5` | Minimum confidence threshold |
| `--action` | str | ❌ No | `nothing` | Image organization mode: `copy`, `move`, or `nothing` |

### Example Output

```
2025-10-19 14:30:15,123 - INFO - Loaded 0 processed files from state.
2025-10-19 14:30:15,124 - INFO - Loading classes from annotated_images/classes.txt...
2025-10-19 14:30:15,125 - INFO - Classes: ['fireball', 'brightmeteor']
2025-10-19 14:30:15,126 - INFO - Using device: cuda
2025-10-19 14:30:15,127 - INFO - Loading model from checkpoints/faster_rcnn_bs4_ne50.pth...
2025-10-19 14:30:16,234 - INFO - Model loaded successfully!

================================================================================
OBJECT DETECTION WATCHER STARTED
================================================================================
Watch directory: incoming/
Output directory: processed/
Score threshold: 0.5
Action: copy
Model: checkpoints/faster_rcnn_bs4_ne50.pth
Classes: fireball, brightmeteor
================================================================================
Watching for new images... Press Ctrl+C to stop.

2025-10-19 14:31:22,456 - INFO - New image detected: meteor001.jpg
2025-10-19 14:31:23,789 - INFO - meteor001.jpg: 3 detections, classes: fireball, brightmeteor
2025-10-19 14:31:23,790 - INFO - Copied to class folders: fireball, brightmeteor

2025-10-19 14:32:15,123 - INFO - New image detected: meteor002.jpg
2025-10-19 14:32:16,456 - INFO - meteor002.jpg: 1 detections, classes: fireball
2025-10-19 14:32:16,457 - INFO - Copied to class folders: fireball

^C
2025-10-19 14:33:00,000 - INFO - Shutting down watcher...
2025-10-19 14:33:00,100 - INFO - Watcher stopped.
```

### State File

The watcher maintains a state file (`processed_files_object_detection.log`) to track processed images:

```
/path/to/incoming/meteor001.jpg
/path/to/incoming/meteor002.jpg
/path/to/incoming/meteor003.jpg
```

This prevents reprocessing images if the watcher is restarted.

### Log Files

Logs are saved to `logs/watcher_object_detection.log` with both file and console output.

### Workflow Example

**Production Pipeline**:

```bash
# Terminal 1: Start the watcher
cd fireball-detector
python3 -m src.watcher_object_detection \
    --model checkpoints/faster_rcnn_best.pth \
    --watch_dir /data/incoming/ \
    --output_dir /data/processed/ \
    --action move \
    --score_threshold 0.8

# Terminal 2: Copy new images to the watched directory
cp /camera/captures/*.jpg /data/incoming/

# The watcher automatically processes each image as it arrives
# Images with detections are moved to class folders
# Visualizations are saved to /data/processed/
```

---

## 2. Object Detection Scanner

### Purpose

Recursively scans a directory tree for images and runs object detection on all found images. Perfect for:
- **Dataset processing**: Process entire directory trees
- **Batch analysis**: Analyze large collections of images
- **Data curation**: Organize existing image collections by detected objects

### Key Features

- ✅ **Recursive scanning**: Processes entire directory trees
- ✅ **File pattern filtering**: Filter by filename patterns
- ✅ **State tracking**: Optional skip of already processed images
- ✅ **Progress reporting**: Real-time progress updates
- ✅ **Summary statistics**: Comprehensive batch statistics
- ✅ **Auto-organization**: Supports copy/move/nothing modes
- ✅ **Error handling**: Continues processing on errors
- ✅ **Multi-class support**: Images appear in all relevant class folders

### Usage

#### Basic Recursive Scan

```bash
cd fireball-detector
python3 -m src.scanner_object_detection \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --input_dir dataset/ \
    --recursive
```

#### With Auto-Organization (Copy Mode)

```bash
python3 -m src.scanner_object_detection \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --input_dir dataset/ \
    --output_dir organized/ \
    --action copy \
    --score_threshold 0.5 \
    --recursive
```

#### With File Pattern Filter

```bash
python3 -m src.scanner_object_detection \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --input_dir dataset/ \
    --output_dir organized/ \
    --action copy \
    --pattern "meteor_*" \
    --recursive
```

#### With State Tracking (Skip Processed)

```bash
python3 -m src.scanner_object_detection \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --input_dir dataset/ \
    --output_dir organized/ \
    --action copy \
    --recursive \
    --skip_processed
```

### Command-Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--model` | str | ✅ Yes | - | Path to trained model checkpoint |
| `--input_dir` | str | ✅ Yes | - | Input directory to scan |
| `--output_dir` | str | ❌ No | `detections` | Output directory for visualizations |
| `--classes` | str | ❌ No | `Config.OD_CLASSES_FILE` | Path to classes file |
| `--score_threshold` | float | ❌ No | `0.5` | Minimum confidence threshold |
| `--action` | str | ❌ No | `nothing` | Image organization mode: `copy`, `move`, or `nothing` |
| `--recursive` | flag | ❌ No | False | Recursively scan subdirectories |
| `--pattern` | str | ❌ No | `*` | File pattern filter (e.g., `meteor_*`) |
| `--skip_processed` | flag | ❌ No | False | Skip images that have been processed before |

### Example Output

```
2025-10-19 14:45:00,123 - INFO - Loading classes from annotated_images/classes.txt...
2025-10-19 14:45:00,124 - INFO - Classes: ['fireball', 'brightmeteor']
2025-10-19 14:45:00,125 - INFO - Using device: cuda
2025-10-19 14:45:00,126 - INFO - Loading model from checkpoints/faster_rcnn_bs4_ne50.pth...
2025-10-19 14:45:01,234 - INFO - Model loaded successfully!

2025-10-19 14:45:01,235 - INFO - Scanning for images in dataset/...
2025-10-19 14:45:02,456 - INFO - Found 150 image files.

================================================================================
OBJECT DETECTION SCANNER
================================================================================
Input directory: dataset/
Output directory: organized/
Recursive: True
Pattern: *
Score threshold: 0.5
Action: copy
Skip processed: False
Total images found: 150
================================================================================

================================================================================
Processing 1/150: meteor001.jpg
================================================================================
2025-10-19 14:45:03,123 - INFO - Found 3 detections, classes: fireball, brightmeteor
2025-10-19 14:45:03,124 - INFO - Copied to class folders: fireball, brightmeteor

================================================================================
Processing 2/150: meteor002.jpg
================================================================================
2025-10-19 14:45:04,456 - INFO - Found 1 detections, classes: fireball
2025-10-19 14:45:04,457 - INFO - Copied to class folders: fireball

...

================================================================================
Processing 150/150: meteor150.jpg
================================================================================
2025-10-19 14:50:00,123 - INFO - Found 2 detections, classes: brightmeteor
2025-10-19 14:50:00,124 - INFO - Copied to class folders: brightmeteor

================================================================================
SCANNER SUMMARY
================================================================================
Total images found: 150
Images processed: 150
Images with detections: 142
Images without detections: 8

Detections per class:
  brightmeteor: 85 images
  fireball: 120 images

Results saved to: organized/

Organized images by class:
  organized/brightmeteor/
  organized/fireball/
================================================================================
2025-10-19 14:50:00,200 - INFO - Scanner finished.
```

### State File

When `--skip_processed` is used, the scanner maintains a state file (`processed_files_object_detection_scanner.log`):

```
/path/to/dataset/subdir1/meteor001.jpg
/path/to/dataset/subdir1/meteor002.jpg
/path/to/dataset/subdir2/meteor003.jpg
```

### Log Files

Logs are saved to `logs/scanner_object_detection.log` with both file and console output.

### Workflow Example

**Dataset Organization**:

```bash
# Step 1: Recursively scan and organize dataset
cd fireball-detector
python3 -m src.scanner_object_detection \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --input_dir /data/raw_dataset/ \
    --output_dir /data/organized_dataset/ \
    --action copy \
    --score_threshold 0.7 \
    --recursive

# Step 2: Review organized images
ls /data/organized_dataset/fireball/
ls /data/organized_dataset/brightmeteor/

# Step 3: Process additional images (skip already processed)
python3 -m src.scanner_object_detection \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --input_dir /data/raw_dataset/ \
    --output_dir /data/organized_dataset/ \
    --action copy \
    --score_threshold 0.7 \
    --recursive \
    --skip_processed
```

---

## Shared Utilities

Both scripts use shared utility functions from `src/utils/detection_utils.py`:

- `load_classes()` - Load class names from classes.txt
- `preprocess_image()` - Preprocess images for inference
- `visualize_predictions()` - Draw bounding boxes on images
- `organize_image_by_class()` - Organize images into class folders
- `process_single_image()` - Complete processing pipeline for one image
- `load_processed_files()` - Load state from file
- `log_processed_file()` - Save state to file
- `is_image_file()` - Check if file is an image

This ensures consistency across all detection scripts and reduces code duplication.

---

## Comparison with detect.py

### When to Use Each Script

**Use `detect.py` when**:
- Processing a single image
- Need verbose diagnostic output
- Want to see detailed predictions
- Testing model performance
- Debugging detection issues

**Use `watcher_object_detection.py` when**:
- Need real-time processing
- Images arrive continuously
- Running a production pipeline
- Want automatic processing without manual intervention
- Need to monitor a camera feed or data source

**Use `scanner_object_detection.py` when**:
- Processing existing directory trees
- Need recursive scanning
- Want to organize large datasets
- Processing historical data
- Need file pattern filtering

### Feature Matrix

| Feature | detect.py | watcher | scanner |
|---------|-----------|---------|---------|
| **Best for** | Testing & debugging | Real-time processing | Batch processing |
| **Input** | Single image or directory | Monitored directory | Directory tree |
| **Processing** | On-demand | Automatic (new files) | Batch (all files) |
| **Recursion** | No | No | Yes (optional) |
| **State tracking** | No | Yes | Yes (optional) |
| **Verbose mode** | Yes | No | No |
| **Pattern filtering** | No | No | Yes |

---

## Best Practices

### 1. **Use Appropriate Thresholds**

- **High threshold (0.7-0.9)**: Production use, fewer false positives
- **Medium threshold (0.3-0.6)**: General use, balanced precision/recall
- **Low threshold (0.1-0.2)**: Testing, see all predictions

### 2. **Monitor Disk Space**

When using `--action copy`, ensure sufficient disk space:

```bash
# Check disk space before processing
df -h /data/organized_dataset/

# Use move instead of copy to save space
--action move
```

### 3. **Use State Tracking**

Enable state tracking to avoid reprocessing:

```bash
# Scanner with state tracking
--skip_processed

# Watcher automatically uses state tracking
```

### 4. **Review Logs**

Check logs for errors and statistics:

```bash
# Watcher logs
tail -f logs/watcher_object_detection.log

# Scanner logs
tail -f logs/scanner_object_detection.log
```

### 5. **Test Before Production**

Test with a small dataset first:

```bash
# Test with copy mode (preserves originals)
--action copy

# Then switch to move mode for production
--action move
```

---

## Troubleshooting

### Issue: Watcher Not Detecting New Files

**Solution**: Ensure the watch directory exists and is writable:

```bash
ls -la /path/to/watch_dir/
chmod 755 /path/to/watch_dir/
```

### Issue: Scanner Finds No Images

**Solution**: Check the pattern and recursive flag:

```bash
# Enable recursive scanning
--recursive

# Check pattern matches your files
--pattern "*.jpg"
```

### Issue: Out of Memory

**Solution**: Process images in smaller batches or reduce batch size in model config.

### Issue: Permission Denied

**Solution**: Check file and directory permissions:

```bash
chmod 644 /path/to/images/*.jpg
chmod 755 /path/to/output_dir/
```

---

## Summary

✅ **Watcher**: Real-time monitoring and automatic processing  
✅ **Scanner**: Recursive batch processing with filtering  
✅ **Shared utilities**: Consistent behavior across all scripts  
✅ **State tracking**: Avoid reprocessing images  
✅ **Auto-organization**: Automatic image organization by class  
✅ **Comprehensive logging**: File and console logging  
✅ **Error handling**: Graceful error handling and reporting  

Both scripts extend the functionality of `detect.py` with automation capabilities, making it easy to process large numbers of images automatically in production environments.

---

**Feature completed**: 2025-10-19  
**Status**: Production Ready ✅  
**Documentation**: Complete ✅

