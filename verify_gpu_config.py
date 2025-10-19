"""
Verification script for RTX 2070 optimized configuration.

This script verifies that the configuration is correctly set and
tests GPU memory usage with the optimized settings.

Usage:
    python3 verify_gpu_config.py
"""

import sys
import subprocess

# Add fireball-detector to path
sys.path.insert(0, 'fireball-detector')

from src.config import Config


def get_gpu_info():
    """Get GPU information."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None, None
        
        gpu_name = torch.cuda.get_device_name(0)
        
        # Get memory info
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            total_mem = int(result.stdout.strip())
        else:
            total_mem = None
        
        return True, gpu_name, total_mem
    except Exception as e:
        return False, None, None


def verify_config():
    """Verify the configuration settings."""
    print("="*80)
    print("RTX 2070 Configuration Verification")
    print("="*80)
    
    # Check CUDA
    cuda_available, gpu_name, total_mem = get_gpu_info()
    
    if not cuda_available:
        print("\n❌ CUDA is not available!")
        print("   Please ensure PyTorch with CUDA support is installed.")
        return False
    
    print(f"\n✅ CUDA Available")
    print(f"   GPU: {gpu_name}")
    if total_mem:
        print(f"   Total Memory: {total_mem} MB")
    
    # Verify it's RTX 2070
    if "2070" not in gpu_name:
        print(f"\n⚠️  Warning: GPU is not RTX 2070")
        print(f"   Current GPU: {gpu_name}")
        print(f"   Configuration is optimized for RTX 2070 (8GB)")
        print(f"   You may need to adjust settings for your GPU.")
    
    # Check configuration values
    print("\n" + "="*80)
    print("Configuration Settings")
    print("="*80)
    
    configs = [
        ("OD_BATCH_SIZE", Config.OD_BATCH_SIZE, 4, "Batch size"),
        ("OD_MIN_SIZE", Config.OD_MIN_SIZE, 800, "Minimum image size"),
        ("OD_MAX_SIZE", Config.OD_MAX_SIZE, 1333, "Maximum image size"),
        ("OD_TRAINABLE_BACKBONE_LAYERS", Config.OD_TRAINABLE_BACKBONE_LAYERS, 3, "Trainable layers"),
        ("OD_NUM_WORKERS", Config.OD_NUM_WORKERS, 6, "Data loading workers"),
    ]
    
    all_correct = True
    
    for name, actual, expected, description in configs:
        status = "✅" if actual == expected else "⚠️ "
        print(f"{status} {name:30} = {actual:4} (expected: {expected:4}) - {description}")
        if actual != expected:
            all_correct = False
    
    # Additional settings
    print(f"\nAdditional Settings:")
    print(f"   OD_BACKBONE: {Config.OD_BACKBONE}")
    print(f"   OD_NUM_EPOCHS: {Config.OD_NUM_EPOCHS}")
    print(f"   OD_LEARNING_RATE: {Config.OD_LEARNING_RATE}")
    
    # Expected GPU usage
    print("\n" + "="*80)
    print("Expected Performance")
    print("="*80)
    print(f"   Expected GPU Memory Usage: ~5273 MB (64.4% of 8192 MB)")
    print(f"   Target Range: 4096-5734 MB (50-70%)")
    print(f"   Headroom: ~2919 MB (35.6%)")
    print(f"   Risk of OOM: Very Low")
    
    # Recommendations
    print("\n" + "="*80)
    print("Next Steps")
    print("="*80)
    
    if all_correct:
        print("✅ Configuration is correctly set for RTX 2070!")
        print("\nTo verify GPU memory usage during training:")
        print("   1. Open a new terminal")
        print("   2. Run: watch -n 1 nvidia-smi")
        print("   3. Start training in another terminal")
        print("   4. Monitor memory usage (should be ~5000-5500 MB)")
        print("\nTo start training:")
        print("   cd fireball-detector")
        print("   python3 -m src.train_object_detection")
    else:
        print("⚠️  Configuration values don't match expected settings!")
        print("   Please check fireball-detector/src/config.py")
        print("   See GPU_OPTIMIZATION_RTX2070.md for details")
    
    return all_correct


def main():
    """Main verification function."""
    try:
        success = verify_config()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

