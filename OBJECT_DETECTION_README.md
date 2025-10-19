# Object Detection Training Guide

This guide explains how to train a Faster R-CNN object detection model using datasets prepared with the `image-classification` annotation tool.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Model Export](#model-export)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)

## Overview

This object detection system uses:
- **Model**: Faster R-CNN with ResNet50 backbone (trained from scratch)
- **Dataset Format**: YOLO format annotations (created by image-classification app)
- **Framework**: PyTorch with torchvision
- **Metrics**: mAP, IoU, Precision, Recall

## Prerequisites

### 1. Install Dependencies

```bash
# For GPU training
pip install -r torch_requirements.txt

# For CPU training
pip install -r torch_requirements_cpu.txt
```

### 2. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
```

## Dataset Preparation

### Using the Image Classification App

1. **Navigate to the image-classification directory**:
   ```bash
   cd ../image-classification
   ```

2. **Build and run the annotation tool**:
   ```bash
   ./build.sh
   ./run_app.sh
   ```

3. **Annotate your images**:
   - Select "Object Detection" mode
   - Click "Open Folder" and select your images
   - Create labels (e.g., "fireball", "meteor", "satellite")
   - Draw bounding boxes around objects
   - Assign labels to each box
   - Click "Save & Next" to save and move to next image

4. **Output structure**:
   ```
   annotated_images/
   ├── images/           # Copied original images
   │   ├── image001.jpg
   │   ├── image002.jpg
   │   └── ...
   ├── labels/           # YOLO format annotations
   │   ├── image001.txt
   │   ├── image002.txt
   │   └── ...
   └── classes.txt       # Class names (one per line)
   ```

5. **YOLO Format** (in each .txt file):
   ```
   <class_id> <x_center> <y_center> <width> <height>
   ```
   All coordinates are normalized (0-1).

### Move Dataset to Training Directory

```bash
# Copy annotated_images to fireball-detector directory
cp -r annotated_images ../fireball-detector/
```

## Training

### Basic Training

```bash
cd fireball-detector

# Train with default settings
python -m src.train_object_detection

# Or specify the dataset path
python -m src.train_object_detection annotated_images
```

### Advanced Training Options

```bash
python -m src.train_object_detection \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 0.001 \
    --save_dir checkpoints
```

### Resume Training

```bash
python -m src.train_object_detection \
    --resume_from checkpoints/checkpoint_epoch_20.pth
```

### Training Output

The training script will:
- Create a `checkpoints/` directory
- Save checkpoints every N epochs (configurable)
- Save the best model based on validation mAP
- Print training statistics and validation metrics

Example output:
```
Epoch 1/50
--------------------------------------------------------------------------------
Epoch [1], Step [10/100], Loss: 2.3456, Time: 5.23s
...
Epoch 1 Training Summary:
  Total Loss: 2.1234
  Classifier Loss: 0.5432
  Box Reg Loss: 0.3210
  Objectness Loss: 0.8765
  RPN Box Reg Loss: 0.3827
  Learning Rate: 0.001000

Running validation...
Validation mAP: 0.4523
AP per class:
  Class 0: 0.4321
  Class 1: 0.4725

New best model saved: checkpoints/faster_rcnn_best_bs4_ne50.pth (mAP: 0.4523)
```

## Inference

### Detect Objects in a Single Image

```bash
python -m src.detect \
    --model checkpoints/faster_rcnn_best_bs4_ne50.pth \
    --image path/to/test_image.jpg \
    --show
```

### Detect Objects in a Directory

```bash
python -m src.detect \
    --model checkpoints/faster_rcnn_best_bs4_ne50.pth \
    --input_dir path/to/test_images/ \
    --output_dir detections/
```

### Adjust Detection Threshold

```bash
python -m src.detect \
    --model checkpoints/faster_rcnn_best_bs4_ne50.pth \
    --image test.jpg \
    --score_threshold 0.7
```

### Detection Output

The script will:
- Print detected objects with confidence scores
- Save visualizations with bounding boxes and labels
- Display images if `--show` flag is used

Example output:
```
Detections for test_image.jpg:
Found 3 objects
  1. fireball: 0.923 at [120.5, 85.3, 245.7, 198.2]
  2. meteor: 0.856 at [350.1, 120.8, 420.3, 180.5]
  3. satellite: 0.734 at [500.2, 300.1, 550.8, 340.6]
Saved visualization to: detections/detected_test_image.jpg
```

## Model Export

### Export to ONNX and TorchScript

```bash
python -m src.detect \
    --model checkpoints/faster_rcnn_best_bs4_ne50.pth \
    --export \
    --export_dir exports/
```

This creates:
- `exports/faster_rcnn_model.onnx` - ONNX format
- `exports/faster_rcnn_model.pt` - TorchScript format

### Use Exported Models

**ONNX**:
```python
import onnxruntime as ort

session = ort.InferenceSession('exports/faster_rcnn_model.onnx')
outputs = session.run(None, {'input': image_tensor})
```

**TorchScript**:
```python
import torch

model = torch.jit.load('exports/faster_rcnn_model.pt')
outputs = model(image_tensor)
```

## Configuration

All configuration parameters are in `src/config.py`. Key settings:

### Training Hyperparameters

```python
OD_NUM_EPOCHS = 50              # Number of training epochs
OD_BATCH_SIZE = 4               # Batch size (reduce if out of memory)
OD_LEARNING_RATE = 0.001        # Initial learning rate
OD_MOMENTUM = 0.9               # SGD momentum
OD_WEIGHT_DECAY = 0.0005        # Weight decay for regularization
```

### Model Settings

```python
OD_BACKBONE = 'resnet50'        # Backbone: resnet50, resnet101, mobilenet_v3
OD_TRAINABLE_BACKBONE_LAYERS = 3  # Number of trainable layers (0-5)
OD_MIN_SIZE = 800               # Min image size
OD_MAX_SIZE = 1333              # Max image size
```

### Detection Settings

```python
OD_BOX_SCORE_THRESH = 0.05      # Score threshold for predictions
OD_BOX_NMS_THRESH = 0.5         # NMS threshold
OD_BOX_DETECTIONS_PER_IMG = 100 # Max detections per image
```

### Evaluation Settings

```python
OD_IOU_THRESHOLD = 0.5          # IoU threshold for metrics
OD_VAL_SPLIT = 0.2              # Validation split ratio
```

## Evaluation Metrics

### mAP (mean Average Precision)

The primary metric for object detection. Measures how well the model detects and classifies objects.

- **Range**: 0.0 to 1.0 (higher is better)
- **Good mAP**: > 0.5 for most applications
- **Excellent mAP**: > 0.7

### IoU (Intersection over Union)

Measures overlap between predicted and ground truth boxes.

- **Formula**: IoU = (Area of Overlap) / (Area of Union)
- **Threshold**: Typically 0.5 (configurable)

### Precision and Recall

- **Precision**: What fraction of detections are correct?
- **Recall**: What fraction of ground truth objects are detected?

### Per-Class Metrics

The validation output shows metrics for each class:

```
AP per class:
  Class 0 (fireball): 0.8234
  Class 1 (meteor): 0.7123
  Class 2 (satellite): 0.6543
```

## Troubleshooting

### Out of Memory Error

Reduce batch size in config:
```python
OD_BATCH_SIZE = 2  # or even 1
```

### Low mAP

1. **More training epochs**: Increase `OD_NUM_EPOCHS`
2. **More data**: Annotate more images
3. **Better annotations**: Ensure bounding boxes are accurate
4. **Adjust learning rate**: Try `OD_LEARNING_RATE = 0.0005`
5. **More trainable layers**: Increase `OD_TRAINABLE_BACKBONE_LAYERS`

### Slow Training

1. **Use GPU**: Ensure CUDA is available
2. **Reduce image size**: Decrease `OD_MIN_SIZE` and `OD_MAX_SIZE`
3. **Fewer workers**: Reduce `OD_NUM_WORKERS`

### No Detections

1. **Lower threshold**: Use `--score_threshold 0.3` or lower
2. **Check model**: Ensure model is trained properly
3. **Verify classes**: Ensure classes.txt matches training data

## Example Workflow

### Complete End-to-End Example

```bash
# 1. Prepare dataset with annotation tool
cd image-classification
./run_app.sh
# Annotate images in Object Detection mode

# 2. Copy dataset
cp -r annotated_images ../fireball-detector/

# 3. Train model
cd ../fireball-detector
python -m src.train_object_detection \
    --num_epochs 50 \
    --batch_size 4

# 4. Run inference
python -m src.detect \
    --model checkpoints/faster_rcnn_best_bs4_ne50.pth \
    --input_dir test_images/ \
    --output_dir results/

# 5. Export model
python -m src.detect \
    --model checkpoints/faster_rcnn_best_bs4_ne50.pth \
    --export
```

## Additional Resources

- **PyTorch Object Detection Tutorial**: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- **Faster R-CNN Paper**: https://arxiv.org/abs/1506.01497
- **YOLO Format**: https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the configuration in `src/config.py`
3. Examine training logs for errors
4. Verify dataset format and annotations

