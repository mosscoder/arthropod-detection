#!/usr/bin/env python3
"""
05_yolo_train.py - YOLOv8 Training for Arthropod Detection

Trains YOLOv8 model using prepared dataset from 04_yolo_prep.py:
- Standard YOLOv8 training with 1280px image resizing
- Optimizes hyperparameters for small object detection
- Trains with validation monitoring and early stopping
- Saves best model and training metrics

Key Features:
- Simple image resizing to 1280px (no custom cropping)
- Standard YOLOv8 data loading and augmentation
- Optimized for single-class arthropod detection
- Scientific imagery augmentation settings

Prerequisites:
- Run 04_yolo_prep.py first to prepare the dataset
- Requires data/yolo_dataset/ with train/val/test structure

Output:
- models/yolov8_arthropod_best.pt - Best trained model
- results/training_metrics.png - Training visualization
- runs/detect/arthropod_detection/ - Training logs and checkpoints
"""

import os
import json
import shutil
from pathlib import Path
from ultralytics import YOLO

# Config
YOLO_DATASET_DIR = "data/yolo_dataset"
MODELS_DIR = "models"
RESULTS_DIR = "results"
IMAGE_SIZE = 1280 * 2

def check_dataset():
    """Check if dataset is properly prepared."""
    required_files = [
        f"{YOLO_DATASET_DIR}/data.yaml",
        f"{YOLO_DATASET_DIR}/dataset_info.json"
    ]
    
    required_dirs = [
        f"{YOLO_DATASET_DIR}/train/images",
        f"{YOLO_DATASET_DIR}/train/labels",
        f"{YOLO_DATASET_DIR}/val/images",
        f"{YOLO_DATASET_DIR}/val/labels",
        f"{YOLO_DATASET_DIR}/test/images",
        f"{YOLO_DATASET_DIR}/test/labels"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Missing required file: {file_path}")
            return False
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Missing required directory: {dir_path}")
            return False
    
    return True

def load_dataset_info():
    """Load dataset metadata from preparation step."""
    info_path = os.path.join(YOLO_DATASET_DIR, 'dataset_info.json')
    
    with open(info_path, 'r') as f:
        dataset_info = json.load(f)
    
    return dataset_info


def setup_output_directories():
    """Create output directories for models and results."""
    dirs = [MODELS_DIR, RESULTS_DIR]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def train_model(data_config_path, dataset_info):
    """Train YOLOv8 model with standard configuration."""
    print("🤖 Initializing YOLOv8 model...")
    
    # Initialize model (can be changed to yolov8s.pt, yolov8m.pt, etc.)
    model = YOLO('models/yolov8n.pt')
    
    print("🚀 Starting training...")
    print(f"   📊 Dataset: {dataset_info['total_annotations']} annotations, {dataset_info['total_images']} images")
    print(f"   🏷️  Classes: {dataset_info['classes']['count']}")
    print(f"   🖼️  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"   📊 Train/Val/Test: {dataset_info['splits']['train']['images']}/{dataset_info['splits']['val']['images']}/{dataset_info['splits']['test']['images']} images")
    print()
    
    # Train model with standard YOLOv8 configuration
    results = model.train(
        data=data_config_path,
        epochs=20,           # Training epochs
        imgsz=IMAGE_SIZE,     # Image size for resizing
        batch=8,              # Batch size (adjust based on GPU memory)
        name='arthropod_detection',
        patience=10,          # Early stopping patience
        save_period=10,       # Save checkpoint every N epochs
        device='cpu',        # Automatically select device (GPU/CPU)
        workers=4,            # Number of worker threads for data loading
        project='runs/detect', # Project directory
        exist_ok=True,        # Allow overwriting existing project
        pretrained=True,      # Use pretrained weights
        optimizer='auto',     # Optimizer selection
        verbose=True,         # Verbose output
        seed=42,              # Random seed for reproducibility
        deterministic=True,   # Use deterministic algorithms
        single_cls=True,      # Single-class training (arthropod)
        rect=False,           # Rectangular training
        cos_lr=True,          # Cosine learning rate scheduler
        close_mosaic=10,      # Epochs to close mosaic
        resume=False,         # Resume training from checkpoint
        amp=True,             # Automatic Mixed Precision training
        fraction=1.0,         # Dataset fraction to use
        profile=False,        # Profile ONNX and TensorRT speeds
        freeze=None,          # Freeze layers
        # Learning rate parameters
        lr0=0.01,             # Initial learning rate
        lrf=0.001,            # Final learning rate
        momentum=0.937,       # SGD momentum
        weight_decay=0.0005,  # Optimizer weight decay
        warmup_epochs=3.0,    # Warmup epochs
        warmup_momentum=0.8,  # Warmup initial momentum
        warmup_bias_lr=0.1,   # Warmup initial bias lr
        # Loss parameters
        box=0.05,             # Box loss gain
        cls=0.5,              # Class loss gain
        dfl=1.5,              # DFL loss gain
        # Augmentation parameters - optimized for scientific imagery
        hsv_h=0.010,          # HSV-Hue augmentation (conservative)
        hsv_s=0.5,            # HSV-Saturation augmentation
        hsv_v=0.3,            # HSV-Value augmentation
        degrees=45.0,         # Image rotation
        translate=0.1,        # Image translation
        scale=0.5,            # Image scale
        shear=0.0,            # Image shear (disabled for scientific accuracy)
        perspective=0.0,      # Image perspective (disabled for scientific accuracy)
        flipud=0.5,           # Image flip up-down (disabled for scientific orientation)
        fliplr=0.5,           # Image flip left-right (safe for arthropods)
        mosaic=0.0,           # Image mosaic (disabled for scientific imagery)
        mixup=0.5,            # Image mixup (disabled for scientific imagery)
        copy_paste=0.0,       # Segment copy-paste (disabled)
        # Other parameters
        plots=True,           # Save plots
        val=True              # Validate during training
    )
    
    return results

def save_model_and_results(results, dataset_info):
    """Save the best model and training results."""
    print("💾 Saving model and results...")
    
    # Save best model
    best_model_path = os.path.join(MODELS_DIR, 'yolov8_arthropod_best.pt')
    best_weights_path = results.save_dir / 'weights' / 'best.pt'
    
    if best_weights_path.exists():
        shutil.copy(best_weights_path, best_model_path)
        print(f"✅ Best model saved to: {best_model_path}")
    else:
        print("❌ Best model weights not found")
    
    # Save training visualization
    results_png_path = results.save_dir / 'results.png'
    if results_png_path.exists():
        results_viz_path = os.path.join(RESULTS_DIR, 'training_metrics.png')
        shutil.copy(results_png_path, results_viz_path)
        print(f"📊 Training metrics saved to: {results_viz_path}")
    
    # Save validation results
    val_batch_path = results.save_dir / 'val_batch0_labels.jpg'
    if val_batch_path.exists():
        val_viz_path = os.path.join(RESULTS_DIR, 'validation_examples.jpg')
        shutil.copy(val_batch_path, val_viz_path)
        print(f"🔍 Validation examples saved to: {val_viz_path}")
    
    # Save confusion matrix
    confusion_path = results.save_dir / 'confusion_matrix.png'
    if confusion_path.exists():
        confusion_viz_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
        shutil.copy(confusion_path, confusion_viz_path)
        print(f"📋 Confusion matrix saved to: {confusion_viz_path}")
    
    return best_model_path

def main():
    """Main function to train YOLOv8 model."""
    print("🚀 Starting YOLOv8 training for arthropod detection...")
    print(f"📁 Dataset: {YOLO_DATASET_DIR}")
    print(f"🤖 Models output: {MODELS_DIR}")
    print(f"📊 Results output: {RESULTS_DIR}")
    print()
    
    # Check if dataset is prepared
    if not check_dataset():
        print("❌ Dataset not properly prepared. Please run 04_yolo_prep.py first.")
        return 1
    
    # Load dataset information
    try:
        dataset_info = load_dataset_info()
    except Exception as e:
        print(f"❌ Error loading dataset info: {e}")
        return 1
    
    # Setup output directories
    setup_output_directories()
    
    # Get dataset configuration path
    data_config_path = os.path.join(YOLO_DATASET_DIR, 'data.yaml')
    
    print("✅ Pre-training setup complete!")
    print(f"   📄 Data config: {data_config_path}")
    print()
    
    # Train the model
    try:
        results = train_model(data_config_path, dataset_info)
        
        # Save model and results
        best_model_path = save_model_and_results(results, dataset_info)
        
        # Run validation on best model
        print("🔍 Running final validation...")
        model = YOLO(best_model_path)
        val_results = model.val(data=data_config_path, imgsz=IMAGE_SIZE, batch=8)
        
        print()
        print("🎉 Training complete!")
        print(f"   📊 Total annotations: {dataset_info['total_annotations']}")
        print(f"   🏷️  Classes: {dataset_info['classes']['count']}")
        print(f"   🖼️  Image size: {IMAGE_SIZE}x{IMAGE_SIZE} (resized)")
        print(f"   📊 Train/Val/Test: {dataset_info['splits']['train']['images']}/{dataset_info['splits']['val']['images']}/{dataset_info['splits']['test']['images']} images")
        print(f"   🤖 Best model: {best_model_path}")
        print(f"   📈 Training logs: {results.save_dir}")
        print(f"   💎 mAP50: {val_results.box.map50:.3f}")
        print(f"   💎 mAP50-95: {val_results.box.map:.3f}")
        print()
        print("🔮 Next steps:")
        print("   • Test the model on new images")
        print("   • Analyze results and confusion matrix")
        print("   • Fine-tune hyperparameters if needed")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())