# RTX 2070 Quick Reference Card

**GPU**: NVIDIA GeForce RTX 2070 (8GB VRAM)  
**Optimization Date**: 2025-10-19  
**Status**: ✅ Production Ready

---

## 📊 Optimized Configuration

```python
# In fireball-detector/src/config.py

OD_BATCH_SIZE = 4                    # 64.4% GPU utilization
OD_MIN_SIZE = 800                    # Standard COCO size
OD_MAX_SIZE = 1333                   # Standard COCO size
OD_TRAINABLE_BACKBONE_LAYERS = 3     # Balanced training
OD_NUM_WORKERS = 6                   # Efficient data loading
```

---

## 🎯 Performance Metrics

| Metric | Value |
|--------|-------|
| **GPU Memory Usage** | 5273 MB (64.4%) |
| **Headroom** | 2919 MB (35.6%) |
| **Risk of OOM** | Very Low |
| **Training Speed** | 8-12 images/sec |
| **Stability** | High |

---

## ✅ Quick Verification

```bash
# Verify configuration
python3 verify_gpu_config.py

# Expected output:
# ✅ Configuration is correctly set for RTX 2070!
```

---

## 🚀 Start Training

```bash
# Terminal 1: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 2: Start training
cd fireball-detector
python3 -m src.train_object_detection
```

**Expected GPU Usage**: ~5000-5500 MB during training

---

## 📈 Expected nvidia-smi Output

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx    CUDA Version: 12.x   |
|-------------------------------+----------------------+----------------------+
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| 45%   65C    P2   150W / 175W |   5273MiB /  8192MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
```

**What to check**:
- ✅ Memory: ~5000-5500 MB
- ✅ GPU Util: 90-100%
- ✅ Temp: < 80°C
- ✅ Power: Near max

---

## 🔧 Troubleshooting

### Out of Memory Error

```python
# Reduce batch size
OD_BATCH_SIZE = 2

# Or reduce image size
OD_MIN_SIZE = 600
OD_MAX_SIZE = 1000
```

### Underutilization (< 50%)

```python
# Increase batch size (with smaller images)
OD_BATCH_SIZE = 6
OD_MIN_SIZE = 600
OD_MAX_SIZE = 1000
```

### Slow Training

```bash
# Check data loading bottleneck
# Increase workers if CPU has more cores
OD_NUM_WORKERS = 8
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| `GPU_OPTIMIZATION_RTX2070.md` | Detailed profiling results |
| `OPTIMIZATION_SUMMARY.md` | Executive summary |
| `QUICKSTART_OBJECT_DETECTION.md` | Training guide |
| `RTX2070_QUICK_REFERENCE.md` | This card |

---

## 🧪 Profiling Data

**Tested Configurations**: 9  
**Optimal Configurations Found**: 2  
**Selected**: Batch=4, Size=(800,1333), Layers=3

| Batch | Size | Layers | Memory | Util % | Status |
|-------|------|--------|--------|--------|--------|
| 2 | 600x1000 | 2 | 2743 MB | 33.5% | ⚠️ Low |
| 4 | 600x1000 | 3 | 3763 MB | 45.9% | ⚠️ Low |
| **4** | **800x1333** | **3** | **5273 MB** | **64.4%** | **✅ Optimal** |
| 6 | 600x1000 | 3 | 4829 MB | 58.9% | ✅ Optimal |
| 8 | 600x1000 | 3 | 7655 MB | 93.4% | ⚠️ High |

---

## 💡 Key Insights

1. **Batch Size 4** provides optimal balance
2. **Standard COCO sizes** (800, 1333) maintain accuracy
3. **3 trainable layers** balance speed and capacity
4. **6 workers** prevent data loading bottleneck
5. **64.4% utilization** leaves safe headroom

---

## 🎓 Alternative Configurations

### Maximum Accuracy (Slower)
```python
OD_BATCH_SIZE = 2
OD_MIN_SIZE = 1000
OD_MAX_SIZE = 1600
OD_TRAINABLE_BACKBONE_LAYERS = 5
```

### Maximum Speed (Lower Accuracy)
```python
OD_BATCH_SIZE = 8
OD_MIN_SIZE = 600
OD_MAX_SIZE = 1000
OD_TRAINABLE_BACKBONE_LAYERS = 2
```

### Balanced Alternative
```python
OD_BATCH_SIZE = 6
OD_MIN_SIZE = 600
OD_MAX_SIZE = 1000
OD_TRAINABLE_BACKBONE_LAYERS = 3
```

---

## 📞 Quick Commands

```bash
# Verify config
python3 verify_gpu_config.py

# Run profiling
python3 test_gpu_memory.py

# Monitor GPU
watch -n 1 nvidia-smi

# Start training
cd fireball-detector && python3 -m src.train_object_detection

# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

**Last Updated**: 2025-10-19  
**Optimization Status**: ✅ Complete  
**Production Ready**: Yes

