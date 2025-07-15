#!/usr/bin/env python3
"""
05_yolo_train.py - Standard YOLO Training for Arthropod Detection

Trains YOLO model using the standard tiled dataset from 04_yolo_prep.py:
- Uses standard YOLO training without custom trainers
- Works with 1280x1280 tiles directly
- Training data: 50% overlap tiles for diversity
- Validation data: 10% overlap tiles for evaluation
- Default YOLO augmentations and data loading

Prerequisites:
- Run 04_yolo_prep.py first to prepare the tiled dataset
- Requires data/yolo_dataset/ with prepared tiles

Output:
- models/yolo_arthropod_best.pt - Best trained model
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
    """Train YOLO model with standard settings."""
    print("🤖 Initializing YOLO model...")
    
    # Initialize model
    model = YOLO('yolo11m.pt')
    
    print("🚀 Starting standard YOLO training...")
    print(f"   📊 Dataset: {dataset_info['total_original_annotations']} original annotations")
    print(f"   🚂 Training tiles: {dataset_info['tiles']['train']} (50% overlap)")
    print(f"   🔍 Validation tiles: {dataset_info['tiles']['val']} (10% overlap)")
    print(f"   🏷️  Classes: {dataset_info['classes']['count']}")
    print()
    
    # Train model with standard settings
    results = model.train(
        data=data_config_path,
        epochs=100,           # Training epochs
        imgsz=1280,           # Image size matches tile size
        batch=16,             # Batch size
        cache=True,           # Cache images for faster training
        name='arthropod_detection_standard',
        patience=25,          # Early stopping patience
        save_period=10,       # Save checkpoint every N epochs
        device='cuda',        # Automatically select device (GPU/CPU)
        workers=4,            # Number of worker threads for data loading
        project='results/detection', # Project directory
        exist_ok=True,        # Allow overwriting existing project
        pretrained=True,      # Use pretrained weights
        optimizer='AdamW',    # Optimizer selection
        verbose=True,         # Verbose output
        seed=42,              # Random seed for reproducibility
        deterministic=True,   # Use deterministic algorithms
        single_cls=True,      # Single-class training (arthropod)
        rect=False,           # Rectangular training (disabled)
        cos_lr=True,         # Cosine learning rate scheduler
        close_mosaic=10,      # Epochs to close mosaic
        resume=False,         # Resume training from checkpoint
        amp=True,             # Automatic Mixed Precision training
        fraction=1.0,         # Dataset fraction to use
        profile=False,        # Profile ONNX and TensorRT speeds
        # Learning rate parameters
        lr0=1e-3,             # Initial learning rate
        lrf=1e-2,             # Final learning rate
        momentum=0.937,       # SGD momentum
        weight_decay=0.0005,  # Optimizer weight decay
        warmup_epochs=5.0,    # Warmup epochs
        warmup_momentum=0.8,  # Warmup initial momentum
        warmup_bias_lr=1e-4,  # Warmup initial bias lr
        # Loss parameters
        box=0.05,             # Box loss gain
        cls=2.0,              # Class loss gain
        dfl=1.5,              # DFL loss gain
        # Augmentation parameters (standard YOLO augmentations)
        hsv_h=0.015,          # HSV-Hue augmentation
        hsv_s=0.7,            # HSV-Saturation augmentation
        hsv_v=0.4,            # HSV-Value augmentation
        degrees=15.0,          # Image rotation (disabled)
        translate=0.1,        # Image translation
        scale=0.2,            # Image scale
        shear=0.0,            # Image shear (disabled)
        perspective=0.0,      # Image perspective (disabled)
        flipud=0.5,           # Image flip up-down (disabled)
        fliplr=0.5,           # Image flip left-right
        mosaic=1.0,           # Image mosaic
        mixup=0.2,            # Image mixup (disabled)
        copy_paste=0.2,       # Segment copy-paste (disabled)
        # Other parameters
        plots=True,           # Save plots
        val=True,             # Enable validation
    )
    
    return results

def save_model_and_results(results, dataset_info):
    """Save the best model and training results."""
    print("💾 Saving model and results...")
    
    # Save best model
    best_model_path = os.path.join(MODELS_DIR, 'yolo_arthropod_best.pt')
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
    val_batch_path = results.save_dir / 'val_batch0_pred.jpg'
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
    print("📐 Using standard YOLO training with 1280x1280 tiles")
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
        
        # Get final metrics from results
        final_metrics = results.results_dict
        
        print()
        print("🎉 Training complete!")
        print(f"   📊 Original annotations: {dataset_info['total_original_annotations']}")
        print(f"   🚂 Training tiles: {dataset_info['tiles']['train']}")
        print(f"   🔍 Validation tiles: {dataset_info['tiles']['val']}")
        print(f"   🏷️  Classes: {dataset_info['classes']['count']}")
        print(f"   🤖 Best model: {best_model_path}")
        print(f"   📈 Training logs: {results.save_dir}")
        
        # Print final validation metrics if available
        if 'metrics/mAP50(B)' in final_metrics:
            print(f"   💎 mAP50: {final_metrics['metrics/mAP50(B)']:.3f}")
            print(f"   💎 mAP50-95: {final_metrics['metrics/mAP50-95(B)']:.3f}")
        
        print()
        print("🔮 Next steps:")
        print("   • Test the model on new full-size images")
        print("   • Analyze validation results and confusion matrix")
        print("   • Fine-tune hyperparameters if needed")
        print("   • Consider using SAHI for inference on large images")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())