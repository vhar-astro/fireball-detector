# Object Detection Diagnostics Guide

**Date**: 2025-10-19  
**Enhancement**: Added verbose diagnostic mode to detection script  
**Status**: ✅ **COMPLETE**

---

## Overview

The `src/detect.py` script has been enhanced with comprehensive diagnostic capabilities to help troubleshoot detection issues and understand model behavior.

---

## New Features

### 1. **Verbose Mode** (`--verbose` flag)

Shows detailed diagnostic information including:
- ✅ All predictions (not just those above threshold)
- ✅ Confidence scores for all predictions
- ✅ Score distribution across different thresholds
- ✅ Per-class prediction statistics
- ✅ Top 20 predictions regardless of threshold
- ✅ Diagnostic suggestions when no detections found

### 2. **Configurable Confidence Threshold** (`--score_threshold`)

- Default: 0.5 (50% confidence)
- Can be lowered to see more predictions: `--score_threshold 0.1`
- Useful for debugging undertrained models

---

## Usage Examples

### Basic Detection (Standard Mode)

```bash
cd fireball-detector
python3 -m src.detect \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --image test.jpg \
    --show
```

**Output**:
```
Found 0 objects (score >= 0.5)

Tip: Use --verbose flag to see all predictions and diagnostic info
```

### Verbose Detection (Diagnostic Mode)

```bash
cd fireball-detector
python3 -m src.detect \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --image test.jpg \
    --verbose \
    --score_threshold 0.1
```

**Output**:
```
================================================================================
DIAGNOSTIC INFORMATION FOR: test.jpg
================================================================================
Image size: (1936, 1216)
Tensor shape: torch.Size([3, 1216, 1936])
Device: cuda
Score threshold: 0.1

================================================================================
RAW MODEL OUTPUT (before filtering)
================================================================================
Total predictions: 55
Score range: [0.0502, 0.4140]
Mean score: 0.1082
Median score: 0.0805

Score distribution:
  >= 0.01:   55 predictions
  >= 0.05:   55 predictions
  >= 0.10:   17 predictions
  >= 0.20:    4 predictions
  >= 0.30:    4 predictions
  >= 0.40:    1 predictions
  >= 0.50:    0 predictions  ← No predictions above default threshold!

Predictions per class (all scores):
  fireball       :   31 predictions (max score: 0.4140, mean: 0.1273)
  brightmeteor   :   24 predictions (max score: 0.1537, mean: 0.0836)

Top 20 predictions (regardless of threshold):
   1. [✓] fireball       : 0.4140 at [ 678.3,   79.3,  886.6,  663.5]
   2. [✓] fireball       : 0.3657 at [ 678.8,  201.7, 1044.7,  593.8]
   3. [✓] fireball       : 0.3528 at [ 599.5,  141.1,  943.1,  549.2]
   ...
```

### Lower Threshold Detection

```bash
cd fireball-detector
python3 -m src.detect \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --image test.jpg \
    --score_threshold 0.05 \
    --show
```

### Batch Processing with Diagnostics

```bash
cd fireball-detector
python3 -m src.detect \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --input_dir annotated_images/images/ \
    --output_dir detections/ \
    --verbose \
    --score_threshold 0.1
```

---

## Understanding the Output

### 1. Diagnostic Information Section

```
================================================================================
DIAGNOSTIC INFORMATION FOR: test.jpg
================================================================================
Image size: (1936, 1216)          ← Original image dimensions
Tensor shape: torch.Size([3, 1216, 1936])  ← Preprocessed tensor shape
Device: cuda                       ← GPU/CPU being used
Score threshold: 0.1               ← Current confidence threshold
```

**What to check**:
- Image size matches expected dimensions
- Tensor shape is [3, H, W] (RGB channels)
- Device is cuda (GPU) for faster inference

### 2. Raw Model Output Section

```
================================================================================
RAW MODEL OUTPUT (before filtering)
================================================================================
Total predictions: 55              ← Total predictions made by model
Score range: [0.0502, 0.4140]     ← Min and max confidence scores
Mean score: 0.1082                 ← Average confidence
Median score: 0.0805               ← Median confidence
```

**What this tells you**:
- **Total predictions > 0**: Model is working, making predictions
- **Score range**: Shows confidence range (0.0-1.0)
- **Mean/Median**: Indicates overall model confidence
  - Low mean (< 0.3): Model may need more training
  - High mean (> 0.7): Model is confident in predictions

### 3. Score Distribution

```
Score distribution:
  >= 0.01:   55 predictions
  >= 0.05:   55 predictions
  >= 0.10:   17 predictions  ← 17 predictions above 10% confidence
  >= 0.20:    4 predictions
  >= 0.30:    4 predictions
  >= 0.40:    1 predictions
  >= 0.50:    0 predictions  ← 0 predictions above 50% confidence
```

**How to interpret**:
- Shows how many predictions exceed each threshold
- Helps you choose appropriate threshold
- If `>= 0.50: 0`, model needs more training or lower threshold

**Example interpretation**:
```
>= 0.50:    0 predictions  → No high-confidence detections
>= 0.10:   17 predictions  → 17 low-confidence detections
```
**Conclusion**: Model is making predictions but with low confidence. Try:
1. Lower threshold to 0.1-0.3
2. Train for more epochs
3. Verify training data quality

### 4. Per-Class Statistics

```
Predictions per class (all scores):
  fireball       :   31 predictions (max score: 0.4140, mean: 0.1273)
  brightmeteor   :   24 predictions (max score: 0.1537, mean: 0.0836)
```

**What this tells you**:
- Distribution of predictions across classes
- Max score per class (highest confidence)
- Mean score per class (average confidence)

**Example interpretation**:
- Fireball: 31 predictions, max 0.414 → Model detects fireballs better
- Brightmeteor: 24 predictions, max 0.154 → Lower confidence for brightmeteors

**Possible reasons**:
- More fireball examples in training data
- Fireball features are more distinctive
- Brightmeteor class needs more training data

### 5. Top Predictions

```
Top 20 predictions (regardless of threshold):
   1. [✓] fireball       : 0.4140 at [ 678.3,   79.3,  886.6,  663.5]
   2. [✓] fireball       : 0.3657 at [ 678.8,  201.7, 1044.7,  593.8]
   ...
  18. [✗] fireball       : 0.0990 at [ 670.7,  430.2,  987.4,  547.5]
```

**Symbols**:
- `[✓]`: Above current threshold (will be shown in final detections)
- `[✗]`: Below current threshold (filtered out)

**Bounding box format**: `[x1, y1, x2, y2]`
- (x1, y1): Top-left corner
- (x2, y2): Bottom-right corner

### 6. Final Detections

```
================================================================================
DETECTIONS FOR: test.jpg
================================================================================
Found 17 objects (score >= 0.1)
  1. fireball: 0.414 at [678.3, 79.3, 886.6, 663.5]
  2. fireball: 0.366 at [678.8, 201.7, 1044.7, 593.8]
  ...
```

**What this shows**:
- Only predictions above the threshold
- These are drawn on the output image
- Saved to `detections/detected_<filename>`

---

## Diagnostic Scenarios

### Scenario 1: "Found 0 objects" (No Verbose)

**Output**:
```
Found 0 objects (score >= 0.5)

Tip: Use --verbose flag to see all predictions and diagnostic info
```

**What to do**:
1. Add `--verbose` flag to see all predictions
2. Lower threshold: `--score_threshold 0.1`

### Scenario 2: Model Makes Predictions But Low Confidence

**Verbose Output**:
```
Total predictions: 55
Score range: [0.0502, 0.4140]
  >= 0.50:    0 predictions
  >= 0.10:   17 predictions
```

**Diagnosis**: Model is working but undertrained

**Solutions**:
1. **Lower threshold** for current model:
   ```bash
   --score_threshold 0.1
   ```

2. **Train for more epochs**:
   ```bash
   cd fireball-detector
   python3 -m src.train_object_detection --num_epochs 100
   ```

3. **Check training progress**:
   - Look at training loss (should decrease)
   - Look at validation mAP (should increase)

### Scenario 3: Model Produces 0 Predictions

**Verbose Output**:
```
Total predictions: 0
WARNING: Model produced 0 predictions!
```

**Diagnosis**: Serious issue with model or input

**Possible causes**:
1. **Model not trained**: Checkpoint is from epoch 0
2. **Input preprocessing issue**: Image not normalized correctly
3. **Model architecture mismatch**: Wrong number of classes

**Solutions**:
1. **Verify model is trained**:
   ```bash
   # Check if checkpoint exists and is not empty
   ls -lh checkpoints/faster_rcnn_bs4_ne50.pth
   ```

2. **Check training logs**: Verify training completed successfully

3. **Try with training image**:
   ```bash
   python3 -m src.detect \
       --model checkpoints/faster_rcnn_bs4_ne50.pth \
       --image annotated_images/images/train_001.jpg \
       --verbose
   ```

### Scenario 4: High Confidence Detections

**Verbose Output**:
```
Total predictions: 45
Score range: [0.5123, 0.9876]
  >= 0.50:   42 predictions
  >= 0.90:    5 predictions
```

**Diagnosis**: Model is well-trained and confident

**What to do**:
- Use default threshold (0.5) or higher (0.7)
- Model is ready for production use

### Scenario 5: Class Imbalance in Predictions

**Verbose Output**:
```
Predictions per class (all scores):
  fireball       :   45 predictions (max score: 0.8140, mean: 0.6273)
  brightmeteor   :    2 predictions (max score: 0.1537, mean: 0.0836)
```

**Diagnosis**: Model biased toward one class

**Possible causes**:
1. Training data imbalance (more fireball examples)
2. One class has more distinctive features
3. Insufficient training for minority class

**Solutions**:
1. **Collect more data** for underrepresented class
2. **Use data augmentation** to balance classes
3. **Train for more epochs** to improve minority class performance

---

## Command-Line Arguments Reference

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--model` | Path to trained model checkpoint | `checkpoints/faster_rcnn_bs4_ne50.pth` |

### Input Arguments (choose one)

| Argument | Description | Example |
|----------|-------------|---------|
| `--image` | Single image for detection | `test.jpg` |
| `--input_dir` | Directory of images | `annotated_images/images/` |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--classes` | str | `annotated_images/classes.txt` | Path to classes file |
| `--output_dir` | str | `detections` | Output directory for visualizations |
| `--score_threshold` | float | `0.5` | Minimum confidence threshold (0.0-1.0) |
| `--show` | flag | False | Display image with detections |
| `--verbose` | flag | False | Show detailed diagnostic information |
| `--export` | flag | False | Export model to ONNX/TorchScript |
| `--export_dir` | str | `exports` | Directory for model exports |

---

## Recommended Workflows

### 1. Initial Model Testing

```bash
# Test with verbose mode and low threshold
cd fireball-detector
python3 -m src.detect \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --image test.jpg \
    --verbose \
    --score_threshold 0.1 \
    --show
```

**What to look for**:
- Total predictions > 0 (model is working)
- Score distribution (choose appropriate threshold)
- Per-class statistics (check for imbalance)

### 2. Finding Optimal Threshold

```bash
# Try different thresholds
for thresh in 0.1 0.2 0.3 0.4 0.5; do
    echo "Testing threshold: $thresh"
    python3 -m src.detect \
        --model checkpoints/faster_rcnn_bs4_ne50.pth \
        --image test.jpg \
        --score_threshold $thresh
done
```

### 3. Batch Evaluation

```bash
# Process all validation images
cd fireball-detector
python3 -m src.detect \
    --model checkpoints/faster_rcnn_bs4_ne50.pth \
    --input_dir annotated_images/images/ \
    --output_dir detections/ \
    --score_threshold 0.3
```

### 4. Production Deployment

```bash
# Use higher threshold for production
cd fireball-detector
python3 -m src.detect \
    --model checkpoints/faster_rcnn_best_bs4_ne50.pth \
    --image new_image.jpg \
    --score_threshold 0.7 \
    --show
```

---

## Troubleshooting Guide

### Issue: "Found 0 objects"

**Step 1**: Add verbose mode
```bash
--verbose
```

**Step 2**: Check if model makes any predictions
- If "Total predictions: 0" → Model/input issue (see Scenario 3)
- If "Total predictions: > 0" → Threshold too high (see Scenario 2)

**Step 3**: Lower threshold
```bash
--score_threshold 0.1
```

**Step 4**: If still no detections, train longer
```bash
cd fireball-detector
python3 -m src.train_object_detection --num_epochs 100
```

### Issue: Too Many False Positives

**Solution**: Raise threshold
```bash
--score_threshold 0.7
```

### Issue: Missing True Positives

**Solution**: Lower threshold
```bash
--score_threshold 0.2
```

### Issue: Low Confidence Scores

**Diagnosis**: Model needs more training

**Solution**:
```bash
cd fireball-detector
python3 -m src.train_object_detection --num_epochs 100
```

---

## Understanding Model Confidence

### Confidence Score Ranges

| Range | Interpretation | Action |
|-------|----------------|--------|
| 0.9 - 1.0 | Very high confidence | Use threshold 0.9 |
| 0.7 - 0.9 | High confidence | Use threshold 0.7 (production) |
| 0.5 - 0.7 | Moderate confidence | Use threshold 0.5 (default) |
| 0.3 - 0.5 | Low confidence | Use threshold 0.3 (testing) |
| 0.1 - 0.3 | Very low confidence | Use threshold 0.1 (debugging) |
| < 0.1 | Extremely low | Model needs training |

### Expected Confidence After Training

| Training Epochs | Expected Max Score | Expected Mean Score |
|-----------------|-------------------|---------------------|
| 2 (test run) | 0.2 - 0.4 | 0.1 - 0.2 |
| 10 | 0.4 - 0.6 | 0.2 - 0.3 |
| 50 | 0.6 - 0.8 | 0.4 - 0.6 |
| 100+ | 0.8 - 0.95 | 0.6 - 0.8 |

**Note**: These are rough estimates. Actual scores depend on:
- Dataset quality and size
- Model architecture
- Training hyperparameters
- Task difficulty

---

## Summary

✅ **Verbose mode added** to detection script  
✅ **Configurable threshold** via command-line  
✅ **Comprehensive diagnostics** for troubleshooting  
✅ **Score distribution analysis** to choose threshold  
✅ **Per-class statistics** to identify imbalances  
✅ **Top predictions view** regardless of threshold  
✅ **Helpful suggestions** when no detections found  

The enhanced detection script provides all the diagnostic information needed to understand model behavior and troubleshoot detection issues.

---

**Enhancement completed**: 2025-10-19  
**Status**: Production Ready ✅  
**Tested**: Successfully tested on test.jpg ✅

