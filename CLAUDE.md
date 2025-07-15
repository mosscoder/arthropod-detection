# Arthropod Detection Pipeline

This repository contains a data processing pipeline for arthropod detection and analysis using annotated petri dish images.

## Pipeline Overview

The pipeline consists of five main scripts that should be run in sequence:

### 00_prep_env.py - Data Preparation
**Purpose:** Downloads arthropod annotation data and images from Google Drive to set up the local environment.

**What it does:**
- Downloads JSON annotation file containing bounding box annotations with labels and rotation data
- Downloads petri dish images from Google Drive folder
- Creates directory structure: `data/tabular/` and `data/raster/petri_dish_src/`
- Requires Google Service Account credentials via environment variable

**Output:**
- `data/tabular/annotations.json` - Annotation data with bounding boxes, labels, and rotation information
- `data/raster/petri_dish_src/` - Directory containing source images

### 01_plot_batch.py - Batch Visualization
**Purpose:** Creates visualization batches to inspect the dataset and verify annotation quality.

**What it does:**
- Loads annotations and finds exact filename matches between annotations and actual image files
- Extracts crops from images based on annotations (with rotation support)
- Creates batches of 12 images with bounding box overlays showing labels
- Supports rotated bounding box visualization using affine transformations
- Uses GDAL for efficient subregion reading from large images

**Features:**
- Exact filename matching to ensure data integrity
- Rotation-aware bounding box visualization
- Configurable batch size and crop dimensions
- Color-coded labels for different arthropod classes

**Output:**
- `results/example_batch.png` - Visualization showing 12 annotated image crops

### 02_establish_data_splits.py - Data Splits
**Purpose:** Creates train/val/test splits of the dataset.

**What it does:**
- Loads annotations and creates train/val/test splits
- For all images with labels, splits data into train/val/test sets (70/20/10 by images)
- Stratifies by label: if a class has fewer than three images, assigns to training set; if exactly three, assigns one to each split
- Saves a data/tabular/split_map.json file that maps each image to its split
- This split map is used by downstream scripts (04_yolo_prep.py, DINOv2 training) to ensure consistent data splits

**Output:**
- `data/tabular/annotations_split.json` - Annotation data with split labels
- `data/tabular/split_map.json` - Mapping of image filenames to their assigned splits

### 03_process_crops.py - Crop Extraction Pipeline
**Purpose:** Extracts individual rectangular thumbnails for each annotation with proper rotation handling.

**What it does:**
- Processes all annotations to extract individual crops for each bounding box
- Handles rotation by calculating envelope sizes and applying proper affine transformations
- Produces rectangular crops (not square) that show annotations upright without black corners
- Uses multiprocessing with n-1 cores for efficient parallel processing
- Generates unique IDs for each crop and maintains metadata

**Rotation Handling:**
- Calculates envelope size needed to contain rotated bounding boxes
- Extracts larger areas from source images to avoid information loss
- Rotates crops to make annotations appear upright
- Crops to original bounding box dimensions for clean rectangular output

**Output:**
- `data/raster/crops/{id}.png` - Individual crop images (rectangular, no rescaling)
- `data/tabular/arthropod_crops_metadata.csv` - Metadata with crop IDs, parent images, labels, coordinates, rotation, and dimensions

### 04_yolo_prep.py - Standard Tiled YOLOv8 Dataset Preparation
**Purpose:** Prepares standard tiled YOLOv8 dataset from arthropod annotation data with different overlap ratios for training and validation.

**What it does:**
- Loads annotations following 02_establish_data_splits.py conventions
- Creates single-class mapping for arthropod detection (taxonomically agnostic)
- Uses pre-established splits from split_map.json to ensure consistency
- **Training data**: Tiles images into 1280x1280 squares with 50% overlap for diversity
- **Validation data**: Tiles images into 1280x1280 squares with 10% overlap
- **Test data**: Saves resized images to 5120px (no tiling) with proper label scaling
- Saves tiles directly as 1280x1280 images (no canvas)
- Labels are normalized relative to each tile
- Handles annotations that span multiple tiles by calculating intersection ratios
- Converts rotated bounding boxes to axis-aligned format for YOLO compatibility
- Uses multiprocessing with n-1 CPU cores for efficient parallel tile generation

**Key Features:**
- **Standard tiling**: Direct tile extraction without canvas complications
- **Different overlaps**: 50% overlap for training (more diversity), 10% for validation
- **Direct saving**: Tiles saved as standard 1280x1280 images
- **Consistent test preprocessing**: Test images resized to 5120px to match label coordinates
- Smart annotation splitting: includes bounding boxes in tiles where at least 10% of the box area is visible
- Parallel processing: uses multiprocessing to speed up tile generation
- Single-class detection: all arthropod types mapped to class 0 for taxonomically agnostic detection

**Output:**
- `data/yolo_dataset/train/images/` - Training tile images (1280x1280, 50% overlap)
- `data/yolo_dataset/train/labels/` - YOLO format training labels
- `data/yolo_dataset/val/images/` - Validation tile images (1280x1280, 10% overlap)
- `data/yolo_dataset/val/labels/` - YOLO format validation labels
- `data/yolo_dataset/test/images/` - Test images (resized to 5120px, no tiling)
- `data/yolo_dataset/test/labels/` - YOLO format test labels
- `data/yolo_dataset/data.yaml` - YOLO configuration with dataset paths
- `data/yolo_dataset/dataset_info.json` - Dataset metadata
- `data/yolo_dataset/tile_mapping.json` - Mapping of tiles back to source images

### 05_yolo_train.py - Standard YOLOv8 Training
**Purpose:** Trains YOLOv8 model using the standard tiled dataset from 04_yolo_prep.py with default YOLOv8 training.

**What it does:**
- Loads dataset configuration from preparation step
- Uses standard YOLOv8 training without custom trainers
- Works directly with 1280x1280 tiles
- Training data has 50% overlap for diversity
- Validation data has 10% overlap for evaluation
- Uses default YOLOv8 augmentations and data loading
- Single-class arthropod detection (taxonomically agnostic)
- Saves best model weights and comprehensive training metrics

**Key Features:**
- **Standard training**: No custom trainers or complex modifications
- **Direct tile usage**: Works with 1280x1280 tiles directly
- **Default augmentations**: Uses proven YOLOv8 augmentations
- **Simple pipeline**: Straightforward training without complications
- Single-class focus: all arthropods treated as one class for detection
- Deterministic training: uses fixed seed for reproducibility

**Training Configuration:**
- Image size: 1280x1280 (matches tile size)
- Batch size: 16
- Epochs: 100 with early stopping
- Optimizer: AdamW
- Augmentations: Standard YOLOv8 augmentations (mosaic, flip, HSV, scale, translate)
- Learning rate: 1e-3 with cosine annealing

**Output:**
- `models/yolov8_arthropod_best.pt` - Best trained model weights
- `results/training_metrics.png` - Training loss curves
- `results/confusion_matrix.png` - Class confusion matrix
- `results/validation_examples.jpg` - Sample validation predictions
- `runs/detect/arthropod_detection_standard/` - Full training logs, checkpoints, and detailed metrics

### 06_inference.py - SAHI Inference with Comprehensive Evaluation
**Purpose:** Performs SAHI (Slicing Aided Hyper Inference) on test images with comprehensive evaluation and visualization, optimized for high recall.

**What it does:**
- Loads trained YOLOv8 model and processes test images with SAHI sliced inference
- Generates side-by-side comparisons: ground truth vs predictions
- Calculates detailed metrics: precision, recall, F1-score per image and overall
- Saves results in organized output structure with JSON exports
- Creates HTML summary report for easy analysis
- **Optimized for recall**: Configuration prioritizes detecting all arthropods over precision

**Key Features:**
- **SAHI sliced inference**: Optimized for large petri dish images with 1280x1280 tiles
- **Recall-focused configuration**: Lower thresholds and increased overlap to minimize false negatives
- **IoU-based matching**: Matches predictions to ground truth using lenient IoU threshold
- **Comprehensive metrics**: Both image-level and dataset-wide performance metrics
- **Visual comparisons**: Side-by-side ground truth vs prediction images
- **JSON export**: All detection results and metrics saved for further analysis
- **HTML reporting**: Human-readable summary with performance breakdown

**SAHI Configuration (Optimized for Recall):**
- Slice size: 1280x1280 (matching training tile size)
- Overlap ratio: 30% for better boundary coverage and detection merging
- Confidence threshold: 0.15 (lowered to capture more potential detections)
- IoU threshold: 0.35 for more lenient NMS (reduced false negatives)
- IoU match threshold: 0.3 for more lenient ground truth matching

**Output:**
- `results/sahi_inference/comparisons/` - Side-by-side truth vs predictions
- `results/sahi_inference/predictions/` - JSON detection results per image
- `results/sahi_inference/metrics/image_metrics.json` - Per-image performance metrics
- `results/sahi_inference/metrics/overall_metrics.json` - Dataset-wide performance metrics
- `results/sahi_inference/summary_report.html` - Human-readable summary report

**Metrics Calculated:**
- **Per-image**: Precision, recall, F1-score, true/false positives/negatives, confidence statistics
- **Overall**: Dataset-wide precision, recall, F1-score, detection counts, confidence distribution
- **Visualization**: Color-coded performance indicators and comparison images

## Training Approach

### Standard Tiling Strategy
The pipeline uses a straightforward tiling approach for efficient training:

1. **Tile Generation**: Large petri dish images are tiled into 1280x1280 squares
2. **Different Overlaps**: 
   - Training: 50% overlap for maximum data diversity
   - Validation: 10% overlap for standard evaluation
3. **Direct Processing**: Tiles are saved as standard 1280x1280 images
4. **Coordinate Normalization**: Bounding boxes are normalized relative to each tile
5. **Standard Training**: Uses default YOLOv8 training with proven augmentations

This approach ensures efficient training on manageable tile sizes while maintaining good coverage of the original images.

## Recent Major Update: Simplified Tiling (January 2025)

### Simplified Standard Tiling Approach
- **Standard tiles**: Direct 1280x1280 tile extraction without canvas complications
- **Different overlaps**: 50% overlap for training diversity, 10% for validation
- **Standard training**: Uses default YOLOv8 training without custom trainers
- **Direct coordinates**: Labels normalized to tile dimensions

### Key Improvements
- **Simpler pipeline**: Removed complex canvas-based approach
- **Different overlap ratios**: Training uses 50% overlap for more diversity
- **Standard YOLOv8**: No custom trainers or complex augmentation pipelines
- **Direct tile saving**: Tiles saved as regular 1280x1280 images
- **Proven approach**: Uses well-tested YOLOv8 defaults

### Technical Benefits
- **Straightforward**: Simple tiling without coordinate transformation issues
- **Training diversity**: 50% overlap provides many training samples
- **Standard validation**: Uses default YOLOv8 validation metrics
- **Simpler codebase**: Removed custom trainer classes and Albumentations
- **Better compatibility**: Works with any YOLOv8 version out of the box

## Usage

Run scripts in sequence:

```bash
# 1. Download data (requires GOOGLE_SERVICE_ACCOUNT_FILE env var)
python 00_prep_env.py

# 2. Visualize batch to verify data quality
python 01_plot_batch.py

# 3. Create train/val/test splits
python 02_establish_data_splits.py

# 4. Extract all crops for downstream analysis (optional for non-YOLO workflows)
python 03_process_crops.py

# 5. Prepare standard tiled YOLOv8 dataset
python 04_yolo_prep.py

# 6. Train YOLOv8 model with standard training
python 05_yolo_train.py

# 7. Run SAHI inference on test images with comprehensive evaluation
python 06_inference.py
```

Note: The tiling approach uses 50% overlap for training data (more diversity) and 10% overlap for validation data. All tiles are saved as standard 1280x1280 images with labels normalized to tile dimensions.

## Dependencies

- GDAL/OGR (for efficient image reading)
- PIL/Pillow (for image processing and rotation)
- Google Drive API (for data download)
- NumPy, Matplotlib (for data processing and visualization)
- Multiprocessing (for parallel crop extraction)
- ultralytics (for YOLOv8 training and inference)
- SAHI (for slicing aided hyper inference)
- PyYAML (for configuration files)
- scikit-learn (for data splitting)
- OpenCV (for fast image processing)

## Data Structure

The annotation format includes:
- Bounding box coordinates (as percentages)
- Rotation angles for oriented annotations
- Arthropod class labels (Hymenoptera, Araneae, Coleoptera, etc.)
- Original image dimensions for coordinate conversion

### Bounding Box Definition

**Coordinate System:**
- `x`, `y`: Top-left corner of the bounding box (as percentage of image dimensions)
- `width`, `height`: Dimensions of the bounding box (as percentage of image dimensions)
- `rotation`: Rotation angle in degrees (optional, defaults to 0.0)

**Rotation Specification:**
- **Rotation Point**: All rotations are applied around the **top-left corner** of the bounding box
- **Rotation Direction**: Positive angles rotate clockwise, negative angles rotate counter-clockwise
- **Coordinate Persistence**: The `x`, `y` coordinates always refer to the top-left corner of the original (unrotated) bounding box
- **Rotation Reference**: The rotation transforms the rectangular bounding box around its top-left corner, not its center

**Example:**
```
bbox = {
    "x": 25.0,          # Top-left X at 25% of image width
    "y": 30.0,          # Top-left Y at 30% of image height  
    "width": 10.0,      # Width is 10% of image width
    "height": 15.0,     # Height is 15% of image height
    "rotation": 30.0    # Rotated 30° clockwise around (25%, 30%) point
}
```

This rotation specification ensures consistent interpretation across visualization (Script 01) and crop extraction (Script 02), where both scripts rotate around the same reference point for accurate object localization.

The pipeline handles various arthropod classes and supports rotated annotations for accurate object detection training data preparation.

## Memories

### Project Purpose
- This analysis aims to develop a robust pipeline for extracting and preparing annotated arthropod images from petri dish photographs
- The goal is to create a high-quality dataset for training machine learning models in arthropod classification and detection
- Supports multiple arthropod classes with advanced annotation handling (including rotated bounding boxes)
- Provides scalable, efficient processing of large image collections with metadata preservation

### @hdbscan_clustering.py
- Script for applying HDBSCAN clustering algorithm to extracted arthropod crop features
- Likely used for unsupervised grouping of arthropod crops based on visual similarities
- Helps in identifying potential natural groupings or clusters within the arthropod dataset
- Can assist in understanding morphological variations or potential undocumented species relationships

### Simplified Tiling Update (January 2025)
- Simplified to standard tiling approach without canvas complications
- Training uses 50% overlap for diversity, validation uses 10% overlap
- Removed custom trainer classes and Albumentations integration
- Uses standard YOLOv8 training with default augmentations
- Direct tile saving as 1280x1280 images with normalized labels

### RESIZE_SIZE Configuration
- I've updated RESIZE_SIZE to 5120