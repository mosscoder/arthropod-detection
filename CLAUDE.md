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
- For all image swith labels, splits data into train/val/test sets (70/20/10 by images)
- Stratify by label, if a class has fewer than three images, assign to the training set. If has exactly three, assign one to each split. 
- Saves a data/tabular/split_map.json file that maps each image to its split. This is used for YOLOv8 and DINOv2 crop training. 

**Output:**
- `data/tabular/annotations_split.json` - Annotation data with split labels

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

### 04_yolo_prep.py - YOLOv8 Dataset Preparation
**Purpose:** Prepares YOLOv8 dataset from arthropod annotation data for deep learning training.

**What it does:**
- Loads annotations following 02_process_crops.py conventions
- Creates class mapping for all arthropod classes
- Splits data into train/val/test sets (70/20/10 by images)
- Generates YOLOv8 format dataset with full-size images
- Creates symlinks to original images for memory efficiency
- Converts rotated bounding boxes to axis-aligned format

**Features:**
- Memory efficient: uses symlinks instead of copying large images
- Proper data splitting: ensures no image appears in multiple splits
- Class balance analysis: reports distribution across splits
- YOLO format: normalized center coordinates for training

**Output:**
- `data/yolo_dataset/` - YOLOv8 format dataset structure
- `data/yolo_dataset/data.yaml` - YOLOv8 data configuration
- `data/yolo_dataset/dataset_info.json` - Dataset metadata and statistics

### 05_yolo_train.py - YOLOv8 Training
**Purpose:** Trains YOLOv8 model using prepared dataset with on-the-fly cropping transforms.

**What it does:**
- Loads dataset configuration and metadata from prep step
- Train, val, test splits are defined in the data.yaml file, all images resized to 1280px. 
- Optimizes hyperparameters for small object detection
- Trains with validation monitoring and early stopping
- Saves best model and comprehensive training metrics

**Key Features:**
- On-the-fly cropping: random 1280px crops generated during training
- Scientific imagery optimized: disabled mosaic/mixup, conservative augmentation
- Memory efficient: works with large petri dish images without pre-cropping
- Comprehensive metrics: saves training plots, confusion matrix, validation examples

**Output:**
- `models/yolov8_arthropod_best.pt` - Best trained model
- `results/training_metrics.png` - Training loss/mAP curves
- `results/confusion_matrix.png` - Class confusion matrix
- `results/validation_examples.jpg` - Validation predictions visualization
- `runs/detect/arthropod_detection/` - Full training logs and checkpoints


## Usage

Run scripts in sequence:

```bash
# 1. Download data (requires GOOGLE_SERVICE_ACCOUNT_FILE env var)
python 00_prep_env.py

# 2. Visualize batch to verify data quality
python 01_plot_batch.py

# 3. Extract all crops for downstream analysis
python 02_process_crops.py

# 4. Prepare YOLOv8 dataset (creates symlinks and YOLO format labels)
python 03_yolo_prep.py

# 5. Train YOLOv8 model with on-the-fly cropping
python 04_yolo_train.py
```

## Dependencies

- GDAL/OGR (for efficient image reading)
- PIL/Pillow (for image processing and rotation)
- Google Drive API (for data download)
- NumPy, Matplotlib (for data processing and visualization)
- Multiprocessing (for parallel crop extraction)
- ultralytics (for YOLOv8 training and inference)
- PyYAML (for configuration files)
- scikit-learn (for data splitting)

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