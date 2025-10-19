# Object Detection Quick Start Guide

Get started with object detection training in 5 minutes!

## 📋 Prerequisites

```bash
# Install dependencies
pip install -r torch_requirements.txt
```

## 🎯 Step 1: Prepare Dataset

### Option A: Use the Annotation Tool

```bash
cd ../image-classification
./build.sh
./run_app.sh
```

1. Select "Object Detection" mode
2. Open your images folder
3. Create labels (e.g., "fireball", "meteor")
4. Draw bounding boxes and assign labels
5. Save annotations

### Option B: Use Existing Dataset

Ensure your dataset follows this structure:
```
annotated_images/
├── images/       # Your images
├── labels/       # YOLO format .txt files
└── classes.txt   # Class names (one per line)
```

## 🚀 Step 2: Train Model

```bash
cd fireball-detector

# Basic training (uses default settings)
python -m src.train_object_detection

# Custom settings
python -m src.train_object_detection \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 0.001
```

**Training will:**
- Create `checkpoints/` directory
- Save best model based on validation mAP
- Print progress and metrics

**Expected output:**
```
Loading dataset from annotated_images...
Loaded 150 images with 3 classes
Classes: ['fireball', 'meteor', 'satellite']
Train dataset: 120 images
Validation dataset: 30 images

Creating model...
Using device: cuda

Starting training for 50 epochs...
Batch size: 4, Learning rate: 0.001
--------------------------------------------------------------------------------

Epoch 1/50
...
Validation mAP: 0.4523
New best model saved: checkpoints/faster_rcnn_best_bs4_ne50.pth
```

## 🔍 Step 3: Run Detection

### Single Image

```bash
python -m src.detect \
    --model checkpoints/faster_rcnn_best_bs4_ne50.pth \
    --image test_image.jpg \
    --show
```

### Batch Processing

```bash
python -m src.detect \
    --model checkpoints/faster_rcnn_best_bs4_ne50.pth \
    --input_dir test_images/ \
    --output_dir results/
```

**Detection output:**
```
Detections for test_image.jpg:
Found 2 objects
  1. fireball: 0.923 at [120.5, 85.3, 245.7, 198.2]
  2. meteor: 0.856 at [350.1, 120.8, 420.3, 180.5]
Saved visualization to: results/detected_test_image.jpg
```

## 📦 Step 4: Export Model (Optional)

```bash
python -m src.detect \
    --model checkpoints/faster_rcnn_best_bs4_ne50.pth \
    --export
```

Creates:
- `exports/faster_rcnn_model.onnx` (ONNX format)
- `exports/faster_rcnn_model.pt` (TorchScript format)

## ⚙️ Common Configurations

### Low Memory (GPU/CPU)

Edit `src/config.py`:
```python
OD_BATCH_SIZE = 2  # or even 1
OD_MIN_SIZE = 600  # reduce from 800
OD_MAX_SIZE = 1000 # reduce from 1333
```

### Faster Training (Less Accurate)

```python
OD_NUM_EPOCHS = 20
OD_TRAINABLE_BACKBONE_LAYERS = 1
```

### Better Accuracy (Slower)

```python
OD_NUM_EPOCHS = 100
OD_TRAINABLE_BACKBONE_LAYERS = 5
OD_BACKBONE = 'resnet101'
```

## 🐛 Troubleshooting

### "Out of memory" error
→ Reduce `OD_BATCH_SIZE` to 2 or 1

### "No images found"
→ Check dataset path and structure

### Low mAP (< 0.3)
→ Train longer, add more data, or check annotations

### No detections
→ Lower `--score_threshold` to 0.3 or 0.2

## 📚 Full Documentation

See [OBJECT_DETECTION_README.md](OBJECT_DETECTION_README.md) for:
- Detailed configuration options
- Evaluation metrics explanation
- Advanced training techniques
- Complete API reference

## 🎓 Example Workflow

```bash
# 1. Annotate images
cd image-classification
./run_app.sh
# (Annotate in Object Detection mode)

# 2. Copy dataset
cp -r annotated_images ../fireball-detector/

# 3. Train
cd ../fireball-detector
python -m src.train_object_detection --num_epochs 50

# 4. Test
python -m src.detect \
    --model checkpoints/faster_rcnn_best_bs4_ne50.pth \
    --image test.jpg \
    --show

# 5. Export
python -m src.detect \
    --model checkpoints/faster_rcnn_best_bs4_ne50.pth \
    --export
```

## 📊 Understanding Results

### Training Metrics

- **Total Loss**: Should decrease over time (< 1.0 is good)
- **mAP**: Mean Average Precision (> 0.5 is good, > 0.7 is excellent)
- **AP per class**: Shows performance for each object type

### Detection Confidence

- **> 0.9**: Very confident detection
- **0.7-0.9**: Good detection
- **0.5-0.7**: Moderate confidence
- **< 0.5**: Low confidence (may be false positive)

## 🔧 File Structure

```
fireball-detector/
├── src/
│   ├── config.py                      # Configuration
│   ├── train_object_detection.py      # Training script
│   ├── detect.py                      # Inference script
│   ├── object_detection_model.py      # Model definition
│   ├── data/
│   │   └── object_detection_dataset.py # Dataset loader
│   └── utils/
│       └── metrics.py                  # Evaluation metrics
├── annotated_images/                   # Your dataset
├── checkpoints/                        # Saved models
├── detections/                         # Detection results
└── exports/                            # Exported models
```

## 💡 Tips

1. **Start small**: Train on 50-100 images first
2. **Check annotations**: Verify bounding boxes are accurate
3. **Monitor mAP**: Should improve each epoch
4. **Save checkpoints**: Don't lose progress
5. **Test early**: Run detection after 10-20 epochs

## 🚀 Next Steps

- Fine-tune hyperparameters in `src/config.py`
- Add more training data for better accuracy
- Experiment with different backbones
- Deploy model in production

Happy detecting! 🎯

