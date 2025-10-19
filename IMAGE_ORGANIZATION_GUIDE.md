# Image Organization Guide - Object Detection

**Date**: 2025-10-19  
**Feature**: Automatic image organization by detected class  
**Status**: ✅ **COMPLETE**

---

## Overview

The object detection inference script (`src/detect.py`) now supports automatic organization of images into class-specific subdirectories based on detection results. This feature works similarly to the image classification workflow and is useful for:

- **Dataset curation**: Automatically organize detected objects by class
- **Quality control**: Review detections by class
- **Data preparation**: Prepare images for further processing or annotation
- **Archiving**: Organize processed images systematically

---

## Features

### 1. **Automatic Image Organization** (`--action` flag)

Three modes available:

| Mode | Description | Use Case |
|------|-------------|----------|
| `nothing` | Default - only save visualizations | Standard detection workflow |
| `copy` | Copy original images to class folders | Keep originals in place |
| `move` | Move original images to class folders | Organize and clean up source directory |

### 2. **Multi-Class Detection Support**

- Images with multiple detected classes are copied/moved to **all relevant class folders**
- Example: Image with both "fireball" and "brightmeteor" appears in both folders
- Preserves original filenames

### 3. **Batch Processing with Statistics**

Enhanced batch processing provides:
- ✅ Progress reporting (e.g., "Processing 15/100 images...")
- ✅ Summary report with detection statistics
- ✅ Per-class image counts
- ✅ Error tracking and reporting
- ✅ Non-recursive directory scanning (only processes files in specified directory)

---

## Usage Examples

### Example 1: Single Image with Copy Action

```bash
cd fireball-detector
python3 -m src.detect \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --image test.jpg \
    --action copy \
    --output_dir organized/ \
    --score_threshold 0.1
```

**Output**:
```
================================================================================
DETECTIONS FOR: test.jpg
================================================================================
Found 17 objects (score >= 0.1)
  1. fireball: 0.414 at [678.3, 79.3, 886.6, 663.5]
  ...
  8. brightmeteor: 0.154 at [625.7, 217.0, 950.3, 337.6]
  ...

Saved visualization to: organized/detected_test.jpg
Copied image to class folders: fireball, brightmeteor
```

**Directory structure**:
```
organized/
├── fireball/
│   └── test.jpg (original image)
├── brightmeteor/
│   └── test.jpg (original image)
└── detected_test.jpg (visualization with bounding boxes)
```

### Example 2: Batch Processing with Copy Action

```bash
cd fireball-detector
python3 -m src.detect \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --input_dir raw_images/ \
    --action copy \
    --output_dir organized/ \
    --score_threshold 0.5
```

**Output**:
```
================================================================================
BATCH PROCESSING
================================================================================
Input directory: raw_images/
Output directory: organized/
Found 100 images
Score threshold: 0.5
Action: copy

================================================================================
Processing 1/100: image001.jpg
================================================================================
Found 3 objects (score >= 0.5)
  1. fireball: 0.856 at [...]
  ...
Copied image to class folders: fireball

================================================================================
Processing 2/100: image002.jpg
================================================================================
Found 2 objects (score >= 0.5)
  1. brightmeteor: 0.723 at [...]
  ...
Copied image to class folders: brightmeteor

...

================================================================================
BATCH PROCESSING SUMMARY
================================================================================
Total images processed: 100/100
Images with detections: 87
Images without detections: 13

Detections per class:
  brightmeteor: 42 images
  fireball: 58 images

Results saved to: organized/

Organized images by class:
  organized/brightmeteor/
  organized/fireball/
================================================================================
```

**Directory structure**:
```
organized/
├── fireball/
│   ├── image001.jpg
│   ├── image003.jpg
│   ├── image005.jpg
│   └── ... (58 images total)
├── brightmeteor/
│   ├── image002.jpg
│   ├── image004.jpg
│   └── ... (42 images total)
├── detected_image001.jpg (visualizations)
├── detected_image002.jpg
├── detected_image003.jpg
└── ... (100 visualizations)
```

### Example 3: Batch Processing with Move Action

```bash
cd fireball-detector
python3 -m src.detect \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --input_dir raw_images/ \
    --action move \
    --output_dir organized/ \
    --score_threshold 0.3
```

**Behavior**:
- Original images are **moved** from `raw_images/` to class folders in `organized/`
- Images with multiple classes are moved to the first class folder, then copied to others
- `raw_images/` will be empty after processing (only images with detections are moved)
- Images without detections remain in `raw_images/`

### Example 4: Default Behavior (Backward Compatible)

```bash
cd fireball-detector
python3 -m src.detect \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --input_dir raw_images/ \
    --output_dir detections/
```

**Behavior**:
- Default `--action nothing` is used
- Only visualizations are saved to `detections/`
- Original images remain untouched in `raw_images/`
- No class subdirectories are created
- **Fully backward compatible** with previous behavior

### Example 5: Verbose Mode with Organization

```bash
cd fireball-detector
python3 -m src.detect \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --input_dir raw_images/ \
    --action copy \
    --output_dir organized/ \
    --verbose \
    --score_threshold 0.1
```

**Behavior**:
- Shows detailed diagnostic information for each image
- Displays all predictions (not just filtered ones)
- Organizes images by detected class
- Useful for debugging and understanding model behavior

---

## Command-Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--model` | Path to trained model checkpoint | `checkpoints/faster_rcnn_bs4_ne50.pth` |

### Input Arguments (choose one)

| Argument | Description | Example |
|----------|-------------|---------|
| `--image` | Single image for detection | `test.jpg` |
| `--input_dir` | Directory of images (non-recursive) | `raw_images/` |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--classes` | str | `annotated_images/classes.txt` | Path to classes file |
| `--output_dir` | str | `detections` | Output directory |
| `--score_threshold` | float | `0.5` | Minimum confidence threshold (0.0-1.0) |
| `--action` | str | `nothing` | Image organization mode: `copy`, `move`, or `nothing` |
| `--show` | flag | False | Display image with detections |
| `--verbose` | flag | False | Show detailed diagnostic information |
| `--export` | flag | False | Export model to ONNX/TorchScript |
| `--export_dir` | str | `exports` | Directory for model exports |

---

## Batch Processing Details

### Non-Recursive Scanning

The `--input_dir` argument processes **only files in the specified directory**, not subdirectories.

**Example**:
```
raw_images/
├── image1.jpg          ← Processed
├── image2.jpg          ← Processed
├── image3.png          ← Processed
└── subdir/
    └── image4.jpg      ← NOT processed
```

To process subdirectories, run the script multiple times or move images to a flat directory.

### Supported Image Formats

- `.jpg` / `.jpeg`
- `.png`
- `.bmp`

### Progress Reporting

During batch processing, you'll see:
```
Processing 1/100: image001.jpg
Processing 2/100: image002.jpg
...
Processing 100/100: image100.jpg
```

### Summary Report

After batch processing completes, a summary is displayed:

```
================================================================================
BATCH PROCESSING SUMMARY
================================================================================
Total images processed: 100/100
Images with detections: 87
Images without detections: 13

Detections per class:
  brightmeteor: 42 images
  fireball: 58 images

Results saved to: organized/

Organized images by class:
  organized/brightmeteor/
  organized/fireball/
================================================================================
```

---

## Multi-Class Detection Behavior

### Images with Multiple Classes

When an image contains multiple detected classes:

**Copy Mode** (`--action copy`):
- Original image is **copied** to all relevant class folders
- Original remains in source directory

**Example**:
```
Input: raw_images/meteor.jpg (contains fireball + brightmeteor)

Output:
organized/
├── fireball/
│   └── meteor.jpg (copy)
├── brightmeteor/
│   └── meteor.jpg (copy)
└── detected_meteor.jpg (visualization)

raw_images/
└── meteor.jpg (original still here)
```

**Move Mode** (`--action move`):
- Original image is **moved** to the first detected class folder
- Then **copied** to other class folders
- Original is removed from source directory

**Example**:
```
Input: raw_images/meteor.jpg (contains fireball + brightmeteor)

Output:
organized/
├── fireball/
│   └── meteor.jpg (moved here first)
├── brightmeteor/
│   └── meteor.jpg (copied from fireball/)
└── detected_meteor.jpg (visualization)

raw_images/
(empty - meteor.jpg was moved)
```

### Images with No Detections

**Copy/Move Mode**:
- Images with no detections (score < threshold) are **not organized**
- They remain in the source directory
- Only visualization is saved (showing no bounding boxes)

**Example**:
```
Input: raw_images/empty_sky.jpg (no detections)

Output:
organized/
└── detected_empty_sky.jpg (visualization with no boxes)

raw_images/
└── empty_sky.jpg (still here - not moved/copied)
```

---

## Error Handling

### File Operation Errors

The script handles common file operation errors gracefully:

**Permission Errors**:
```
Warning: Failed to copy image to organized/fireball/: Permission denied
```

**Disk Space Errors**:
```
Warning: Failed to copy image to organized/fireball/: No space left on device
```

**Duplicate Filenames**:
- Files with the same name are **overwritten** in the destination
- Use unique filenames to avoid data loss

### Processing Errors

If an image fails to process:
```
✗ Error processing image001.jpg: Invalid image format
```

The script continues processing remaining images and reports errors in the summary:
```
Errors encountered: 2
  image001.jpg: Invalid image format
  image050.jpg: Corrupted file
```

---

## Workflow Examples

### Workflow 1: Dataset Curation

**Goal**: Organize a large collection of images by detected objects

```bash
# Step 1: Run detection with copy action
cd fireball-detector
python3 -m src.detect \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --input_dir raw_dataset/ \
    --action copy \
    --output_dir curated_dataset/ \
    --score_threshold 0.7

# Step 2: Review organized images
ls curated_dataset/fireball/        # Review fireball detections
ls curated_dataset/brightmeteor/    # Review brightmeteor detections

# Step 3: Manual quality control
# Remove false positives from class folders
# Add missed detections manually
```

### Workflow 2: Production Pipeline

**Goal**: Process incoming images and archive by class

```bash
# Step 1: Run detection with move action
cd fireball-detector
python3 -m src.detect \
    --model checkpoints/faster_rcnn_best_bs4_ne50.pth \
    --input_dir incoming/ \
    --action move \
    --output_dir archive/ \
    --score_threshold 0.8

# Step 2: Images are automatically organized
# incoming/ is now empty (all processed images moved)
# archive/fireball/ contains fireball detections
# archive/brightmeteor/ contains brightmeteor detections
```

### Workflow 3: Debugging Low Confidence

**Goal**: Find images where model has low confidence

```bash
# Step 1: Run with low threshold and verbose mode
cd fireball-detector
python3 -m src.detect \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --input_dir test_images/ \
    --action copy \
    --output_dir debug/ \
    --score_threshold 0.1 \
    --verbose

# Step 2: Review diagnostic output
# Check which images have low confidence scores
# Identify patterns in failed detections

# Step 3: Retrain model or adjust threshold
```

---

## Best Practices

### 1. **Use Copy Mode for Safety**

When first testing, use `--action copy` to preserve originals:
```bash
--action copy
```

### 2. **Set Appropriate Threshold**

- **High threshold (0.7-0.9)**: Production use, fewer false positives
- **Medium threshold (0.3-0.6)**: General use, balanced precision/recall
- **Low threshold (0.1-0.2)**: Debugging, see all predictions

### 3. **Backup Before Move**

Before using `--action move`, backup your data:
```bash
cp -r raw_images/ raw_images_backup/
```

### 4. **Use Verbose Mode for Debugging**

When troubleshooting, add `--verbose`:
```bash
--verbose --score_threshold 0.1
```

### 5. **Check Summary Report**

Always review the batch processing summary to:
- Verify expected number of detections
- Check for errors
- Validate class distribution

### 6. **Handle Duplicate Filenames**

Ensure unique filenames in source directory to avoid overwriting:
```bash
# Rename files with timestamps
for f in *.jpg; do mv "$f" "$(date +%s)_$f"; done
```

---

## Comparison with Image Classification

| Feature | Object Detection | Image Classification |
|---------|------------------|---------------------|
| **Multi-class per image** | ✅ Yes (image can be in multiple class folders) | ❌ No (single class per image) |
| **Bounding boxes** | ✅ Yes (visualizations show boxes) | ❌ No |
| **Confidence threshold** | ✅ Configurable per detection | ✅ Configurable per image |
| **Batch processing** | ✅ Yes with statistics | ✅ Yes with statistics |
| **Action modes** | ✅ copy, move, nothing | ✅ copy, move, nothing |
| **Verbose diagnostics** | ✅ Yes (shows all predictions) | ✅ Yes (shows probabilities) |

---

## Troubleshooting

### Issue: No Images Organized

**Symptoms**:
```
Images with detections: 0
Images without detections: 100
```

**Solutions**:
1. Lower the threshold: `--score_threshold 0.1`
2. Use verbose mode to see predictions: `--verbose`
3. Check if model is trained properly
4. Verify images are similar to training data

### Issue: Too Many False Positives

**Symptoms**: Images organized into wrong classes

**Solutions**:
1. Raise the threshold: `--score_threshold 0.7`
2. Train model for more epochs
3. Review and clean training data

### Issue: Permission Denied

**Symptoms**:
```
Warning: Failed to copy image: Permission denied
```

**Solutions**:
1. Check directory permissions: `ls -la output_dir/`
2. Run with appropriate permissions
3. Choose a different output directory

### Issue: Disk Space

**Symptoms**:
```
Warning: Failed to copy image: No space left on device
```

**Solutions**:
1. Check disk space: `df -h`
2. Use `--action move` instead of `copy` to save space
3. Clean up old files
4. Use a different output directory with more space

---

## Summary

✅ **Automatic image organization** by detected class  
✅ **Three action modes**: copy, move, nothing  
✅ **Multi-class support**: Images appear in all relevant class folders  
✅ **Batch processing** with progress reporting  
✅ **Comprehensive statistics** and error reporting  
✅ **Backward compatible**: Default behavior unchanged  
✅ **Non-recursive scanning**: Only processes specified directory  
✅ **Error handling**: Graceful handling of file operation errors  

The enhanced detection script provides a complete workflow for organizing detected objects, making it easy to curate datasets, review detections, and prepare images for further processing.

---

**Feature completed**: 2025-10-19  
**Status**: Production Ready ✅  
**Tested**: Successfully tested with copy and move actions ✅

