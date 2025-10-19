"""
GPU Memory Profiling Script for Object Detection Training

This script tests different batch sizes and image sizes to find optimal
configuration for RTX 2070 (8GB VRAM) targeting 50-70% utilization (4-6GB).

Usage:
    python3 test_gpu_memory.py
"""

import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import subprocess
import time

# Add fireball-detector to path
sys.path.insert(0, 'fireball-detector')

from src.config import Config
from src.object_detection_model import create_faster_rcnn_from_config


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            used, total = map(int, result.stdout.strip().split(','))
            return used, total
        return None, None
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return None, None


def create_dummy_batch(batch_size, image_size, num_classes=3):
    """Create dummy batch for testing."""
    images = []
    targets = []
    
    for _ in range(batch_size):
        # Create random image
        img = torch.rand(3, image_size, image_size)
        images.append(img)
        
        # Create random targets (5 boxes per image)
        num_boxes = 5
        boxes = torch.rand(num_boxes, 4) * image_size
        # Ensure x2 > x1 and y2 > y1
        boxes[:, 2:] = boxes[:, :2] + torch.abs(boxes[:, 2:])
        
        labels = torch.randint(1, num_classes + 1, (num_boxes,))
        
        targets.append({
            'boxes': boxes,
            'labels': labels
        })
    
    return images, targets


def test_configuration(batch_size, min_size, max_size, trainable_layers, num_classes=3):
    """Test a specific configuration and measure GPU memory usage."""
    print(f"\n{'='*80}")
    print(f"Testing Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Min Size: {min_size}, Max Size: {max_size}")
    print(f"  Trainable Backbone Layers: {trainable_layers}")
    print(f"{'='*80}")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    time.sleep(1)
    
    # Get baseline memory
    baseline_used, total_mem = get_gpu_memory_usage()
    if baseline_used is None:
        print("❌ Could not get GPU memory info")
        return None
    
    print(f"Baseline GPU Memory: {baseline_used} MB / {total_mem} MB ({baseline_used/total_mem*100:.1f}%)")
    
    try:
        # Create model
        device = torch.device('cuda')
        
        # Temporarily update config
        original_min = Config.OD_MIN_SIZE
        original_max = Config.OD_MAX_SIZE
        original_trainable = Config.OD_TRAINABLE_BACKBONE_LAYERS
        
        Config.OD_MIN_SIZE = min_size
        Config.OD_MAX_SIZE = max_size
        Config.OD_TRAINABLE_BACKBONE_LAYERS = trainable_layers
        
        model = create_faster_rcnn_from_config(num_classes)
        model = model.to(device)
        model.train()
        
        # Restore config
        Config.OD_MIN_SIZE = original_min
        Config.OD_MAX_SIZE = original_max
        Config.OD_TRAINABLE_BACKBONE_LAYERS = original_trainable
        
        # Get memory after model loading
        after_model_used, _ = get_gpu_memory_usage()
        model_memory = after_model_used - baseline_used
        print(f"Model Memory: {model_memory} MB")
        
        # Create dummy batch
        images, targets = create_dummy_batch(batch_size, min_size, num_classes)
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        print("Running forward pass...")
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Get memory after forward pass
        after_forward_used, _ = get_gpu_memory_usage()
        forward_memory = after_forward_used - baseline_used
        print(f"Memory after Forward: {forward_memory} MB ({after_forward_used/total_mem*100:.1f}%)")
        
        # Backward pass
        print("Running backward pass...")
        losses.backward()
        
        # Get peak memory
        peak_used, _ = get_gpu_memory_usage()
        peak_memory = peak_used - baseline_used
        utilization = peak_used / total_mem * 100
        
        print(f"\n📊 Results:")
        print(f"  Peak GPU Memory: {peak_used} MB / {total_mem} MB")
        print(f"  Memory Used by Training: {peak_memory} MB")
        print(f"  GPU Utilization: {utilization:.1f}%")
        
        # Check if within target range (50-70% = 4000-5600 MB for 8GB)
        target_min = total_mem * 0.50
        target_max = total_mem * 0.70
        
        if target_min <= peak_used <= target_max:
            print(f"  ✅ OPTIMAL - Within target range (50-70%)")
            status = "optimal"
        elif peak_used < target_min:
            print(f"  ⚠️  UNDERUTILIZED - Below 50% ({peak_used} < {target_min:.0f} MB)")
            status = "underutilized"
        else:
            print(f"  ⚠️  HIGH - Above 70% ({peak_used} > {target_max:.0f} MB)")
            status = "high"
        
        # Cleanup
        del model, images, targets, loss_dict, losses
        torch.cuda.empty_cache()
        
        return {
            'batch_size': batch_size,
            'min_size': min_size,
            'max_size': max_size,
            'trainable_layers': trainable_layers,
            'peak_memory_mb': peak_used,
            'utilization_pct': utilization,
            'status': status
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ OUT OF MEMORY ERROR")
            torch.cuda.empty_cache()
            return {
                'batch_size': batch_size,
                'min_size': min_size,
                'max_size': max_size,
                'trainable_layers': trainable_layers,
                'peak_memory_mb': None,
                'utilization_pct': None,
                'status': 'oom'
            }
        else:
            print(f"❌ Error: {e}")
            torch.cuda.empty_cache()
            return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        return None


def main():
    """Run GPU memory profiling tests."""
    print("="*80)
    print("GPU Memory Profiling for RTX 2070 (8GB VRAM)")
    print("Target: 50-70% utilization (4000-5600 MB)")
    print("="*80)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. This script requires a GPU.")
        sys.exit(1)
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    
    # Get total GPU memory
    _, total_mem = get_gpu_memory_usage()
    print(f"Total GPU Memory: {total_mem} MB")
    print(f"Target Range: {total_mem*0.5:.0f} - {total_mem*0.7:.0f} MB (50-70%)")
    
    # Test configurations
    # Format: (batch_size, min_size, max_size, trainable_layers)
    configs = [
        # Conservative configurations
        (2, 600, 1000, 2),
        (2, 800, 1333, 3),
        (4, 600, 1000, 2),
        (4, 600, 1000, 3),
        
        # Moderate configurations
        (4, 800, 1333, 3),  # Current default
        (6, 600, 1000, 3),
        (8, 600, 1000, 2),
        
        # Aggressive configurations
        (8, 600, 1000, 3),
        (8, 800, 1333, 3),
    ]
    
    results = []
    
    for batch_size, min_size, max_size, trainable_layers in configs:
        result = test_configuration(batch_size, min_size, max_size, trainable_layers)
        if result:
            results.append(result)
        time.sleep(2)  # Cool down between tests
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL CONFIGURATIONS")
    print("="*80)
    print(f"{'Batch':>5} {'MinSize':>7} {'MaxSize':>7} {'Layers':>6} {'Memory(MB)':>11} {'Util%':>6} {'Status':>15}")
    print("-"*80)
    
    optimal_configs = []
    
    for r in results:
        if r['peak_memory_mb'] is not None:
            status_symbol = {
                'optimal': '✅',
                'underutilized': '⚠️ ',
                'high': '⚠️ ',
                'oom': '❌'
            }.get(r['status'], '?')
            
            print(f"{r['batch_size']:>5} {r['min_size']:>7} {r['max_size']:>7} "
                  f"{r['trainable_layers']:>6} {r['peak_memory_mb']:>11} "
                  f"{r['utilization_pct']:>6.1f} {status_symbol} {r['status']:>13}")
            
            if r['status'] == 'optimal':
                optimal_configs.append(r)
        else:
            print(f"{r['batch_size']:>5} {r['min_size']:>7} {r['max_size']:>7} "
                  f"{r['trainable_layers']:>6} {'OOM':>11} {'N/A':>6} {'❌ OOM':>15}")
    
    # Recommend best configuration
    if optimal_configs:
        print("\n" + "="*80)
        print("RECOMMENDED CONFIGURATIONS (50-70% utilization)")
        print("="*80)
        
        # Sort by utilization (prefer higher within range)
        optimal_configs.sort(key=lambda x: x['utilization_pct'], reverse=True)
        
        for i, config in enumerate(optimal_configs[:3], 1):
            print(f"\nOption {i}:")
            print(f"  OD_BATCH_SIZE = {config['batch_size']}")
            print(f"  OD_MIN_SIZE = {config['min_size']}")
            print(f"  OD_MAX_SIZE = {config['max_size']}")
            print(f"  OD_TRAINABLE_BACKBONE_LAYERS = {config['trainable_layers']}")
            print(f"  Expected GPU Usage: {config['peak_memory_mb']} MB ({config['utilization_pct']:.1f}%)")
    else:
        print("\n⚠️  No optimal configurations found in target range (50-70%)")
        print("Consider using the configuration closest to 50-70% range.")


if __name__ == '__main__':
    main()

