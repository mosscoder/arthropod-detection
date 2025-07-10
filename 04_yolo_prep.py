#!/usr/bin/env python3
"""
04_yolo_prep.py - YOLOv8 Dataset Preparation for Arthropod Detection

Prepares YOLOv8 dataset from arthropod annotation data:
- Loads annotations following 02_process_crops.py conventions
- Creates single class mapping for arthropod detection (agnostic to taxonomic groups)
- Uses pre-established splits from 02_establish_data_splits.py
- Generates YOLO format labels and creates symlinks in standard structure
- Converts rotated bounding boxes to axis-aligned format
- All arthropod types are treated as a single 'arthropod' class

Key Features:
- Standard YOLO directory structure with proper images/labels hierarchy
- Symlinks to original images (no file duplication)
- Minimal footprint: only creates necessary label files and symlinks
- Compatible with YOLOv8 training expectations

Prerequisites:
- Run 00_prep_env.py first to download data
- Run 02_establish_data_splits.py to create train/val/test splits
- Requires data/tabular/annotations.json, data/tabular/split_map.json, and data/raster/petri_dish_src/

Output:
- data/yolo_dataset/train/images/ - Symlinks to training images
- data/yolo_dataset/train/labels/ - YOLO format training labels
- data/yolo_dataset/val/images/ - Symlinks to validation images
- data/yolo_dataset/val/labels/ - YOLO format validation labels
- data/yolo_dataset/test/images/ - Symlinks to test images
- data/yolo_dataset/test/labels/ - YOLO format test labels
- data/yolo_dataset/data.yaml - YOLO config with standard paths
- data/yolo_dataset/dataset_info.json - Dataset metadata with path mappings
"""

import os
import json
import yaml
import math
from pathlib import Path
from collections import defaultdict, Counter
from osgeo import gdal

# Configuration
ANNOTATIONS_PATH = "data/tabular/annotations.json"
SPLIT_MAP_PATH = "data/tabular/split_map.json"
IMAGES_DIR = "data/raster/petri_dish_src"
YOLO_DATASET_DIR = "data/yolo_dataset"

def setup_directories():
    """Create necessary directories for YOLOv8 dataset with proper structure."""
    dirs = [
        f"{YOLO_DATASET_DIR}/train/images",
        f"{YOLO_DATASET_DIR}/train/labels",
        f"{YOLO_DATASET_DIR}/val/images",
        f"{YOLO_DATASET_DIR}/val/labels",
        f"{YOLO_DATASET_DIR}/test/images",
        f"{YOLO_DATASET_DIR}/test/labels"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_annotations():
    """Load and parse annotations following 02_process_crops.py conventions."""
    print(f"📄 Loading annotations from: {ANNOTATIONS_PATH}")
    
    with open(ANNOTATIONS_PATH, 'r') as f:
        data = json.load(f)
    
    # Get all available image files for exact matching
    all_images = set()
    if os.path.exists(IMAGES_DIR):
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        for filename in os.listdir(IMAGES_DIR):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                all_images.add(filename)
    
    print(f"📁 Found {len(all_images)} total images in directory")
    
    # Find intersection of annotation filenames and actual image files
    annotation_filenames = set()
    for item in data:
        if 'data' in item and 'image' in item['data']:
            annotation_filenames.add(os.path.basename(item['data']['image']))
    
    matching_images = annotation_filenames.intersection(all_images)
    print(f"📄 Found {len(matching_images)} images with exact filename matches")
    
    # Process annotations to extract individual bbox annotations
    annotations = []
    processed_count = 0
    skipped_count = 0
    
    for item in data:
        if 'data' not in item or 'image' not in item['data']:
            skipped_count += 1
            continue
            
        if not item.get('annotations') or not item['annotations'][0].get('result'):
            skipped_count += 1
            continue
        
        # Get annotation filename and check for exact match
        annotation_filename = os.path.basename(item['data']['image'])
        
        if annotation_filename not in matching_images:
            skipped_count += 1
            continue
        
        # Get the actual image path
        image_path = os.path.join(IMAGES_DIR, annotation_filename)
        
        if not os.path.exists(image_path):
            skipped_count += 1
            continue
        
        first_result = item['annotations'][0]['result'][0]
        orig_w = first_result['original_width']
        orig_h = first_result['original_height']
        
        # Extract individual bounding boxes
        for ann in item['annotations']:
            if not ann.get('result'):
                continue
                
            for res in ann['result']:
                if res['type'] == 'rectanglelabels':
                    bbox = res['value']
                    label = res['value']['rectanglelabels'][0]
                    
                    # Convert percentage to absolute coordinates
                    abs_bbox = {
                        'x': bbox['x'] / 100 * orig_w,
                        'y': bbox['y'] / 100 * orig_h,
                        'width': bbox['width'] / 100 * orig_w,
                        'height': bbox['height'] / 100 * orig_h,
                        'rotation': bbox.get('rotation', 0.0)
                    }
                    
                    annotations.append({
                        'image_path': image_path,
                        'image_filename': annotation_filename,
                        'bbox': abs_bbox,
                        'label': label,
                        'image_width': orig_w,
                        'image_height': orig_h
                    })
                    processed_count += 1
    
    print(f"📊 Found {processed_count} individual annotations to process")
    print(f"   ⏭️  Skipped {skipped_count} items (no data, annotations, or file not found)")
    
    return annotations

def create_class_mapping(annotations):
    """Create single class mapping for arthropod detection (agnostic to taxonomic groups)."""
    # Get unique labels from annotations for counting purposes
    labels = set(ann['label'] for ann in annotations)
    
    # Remove invalid labels
    labels.discard('label')  # Header artifact
    
    # Create single class mapping - all arthropods map to class 0
    class_to_idx = {'arthropod': 0}
    idx_to_class = {0: 'arthropod'}
    
    # Count total annotations and show breakdown by original labels
    total_annotations = len(annotations)
    print(f"🏷️  Creating single 'arthropod' class from {len(labels)} original taxonomic groups:")
    for label in sorted(labels):
        count = sum(1 for ann in annotations if ann['label'] == label)
        print(f"   {label}: {count} samples → arthropod (class 0)")
    print(f"   Total: {total_annotations} samples → arthropod (class 0)")
    
    return class_to_idx, idx_to_class

def compute_rotated_corners(left, top, width, height, rotation_degrees):
    """Compute 4 corners of bounding box after rotation around top-left corner."""
    if abs(rotation_degrees) < 0.001:
        # No rotation - return original corners
        return [
            (left, top),
            (left + width, top),
            (left + width, top + height),
            (left, top + height)
        ]
    
    # Convert to radians
    rotation_rad = math.radians(rotation_degrees)
    cos_r = math.cos(rotation_rad)
    sin_r = math.sin(rotation_rad)
    
    # Original corners relative to top-left (rotation center)
    corners = [
        (0, 0),                    # top-left (rotation center)
        (width, 0),                # top-right
        (width, height),           # bottom-right
        (0, height)                # bottom-left
    ]
    
    # Apply rotation transformation around (0,0) then translate back
    rotated_corners = []
    for x, y in corners:
        # Rotate around origin
        new_x = x * cos_r - y * sin_r
        new_y = x * sin_r + y * cos_r
        # Translate back to absolute coordinates
        rotated_corners.append((left + new_x, top + new_y))
    
    return rotated_corners

def bbox_to_axis_aligned(bbox):
    """Convert rotated bounding box to axis-aligned bounding box."""
    left = bbox['x']
    top = bbox['y']
    width = bbox['width']
    height = bbox['height']
    rotation = bbox['rotation']
    
    if abs(rotation) < 0.001:
        # No rotation, return as-is
        return left, top, left + width, top + height
    
    # Get rotated corners
    corners = compute_rotated_corners(left, top, width, height, rotation)
    
    # Find axis-aligned bounding box
    x_coords = [corner[0] for corner in corners]
    y_coords = [corner[1] for corner in corners]
    
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

def group_annotations_by_image(annotations):
    """Group annotations by image for proper train/val/test splitting."""
    image_groups = defaultdict(list)
    
    for ann in annotations:
        image_groups[ann['image_filename']].append(ann)
    
    return dict(image_groups)

def load_split_map():
    """Load pre-established train/val/test splits from split_map.json."""
    print(f"📄 Loading split map from: {SPLIT_MAP_PATH}")
    
    with open(SPLIT_MAP_PATH, 'r') as f:
        split_map = json.load(f)
    
    # Organize by split
    train_images = [img for img, split in split_map.items() if split == 'train']
    val_images = [img for img, split in split_map.items() if split == 'val']
    test_images = [img for img, split in split_map.items() if split == 'test']
    
    print(f"📊 Loaded splits: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    return train_images, val_images, test_images

def create_dataset_files(image_groups, image_list, split_name, class_to_idx, dataset_dir):
    """Generate YOLO format labels and create symlinks to images in proper structure."""
    print(f"🔄 Creating files for {split_name} split...")
    
    # Create subdirectories for this split
    images_dir = os.path.join(dataset_dir, split_name, 'images')
    labels_dir = os.path.join(dataset_dir, split_name, 'labels')
    
    file_count = 0
    
    for image_filename in image_list:
        annotations = image_groups[image_filename]
        image_width = annotations[0]['image_width']
        image_height = annotations[0]['image_height']
        original_image_path = annotations[0]['image_path']
        
        # Create symlink to original image
        image_symlink_path = os.path.join(images_dir, image_filename)
        if os.path.exists(image_symlink_path):
            os.unlink(image_symlink_path)
        os.symlink(os.path.abspath(original_image_path), image_symlink_path)
        
        # Create label file
        label_filename = f"{Path(image_filename).stem}.txt"
        label_path = os.path.join(labels_dir, label_filename)
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                # Convert to normalized YOLO format for full image
                bbox = ann['bbox']
                
                # Convert rotated bbox to axis-aligned
                bbox_left, bbox_top, bbox_right, bbox_bottom = bbox_to_axis_aligned(bbox)
                
                # Normalize to image dimensions
                center_x = (bbox_left + bbox_right) / 2 / image_width
                center_y = (bbox_top + bbox_bottom) / 2 / image_height
                norm_width = (bbox_right - bbox_left) / image_width
                norm_height = (bbox_bottom - bbox_top) / image_height
                
                # Ensure coordinates are within bounds
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                norm_width = max(0, min(1, norm_width))
                norm_height = max(0, min(1, norm_height))
                
                # All arthropods get class index 0
                class_idx = 0
                f.write(f"{class_idx} {center_x:.6f} {center_y:.6f} "
                       f"{norm_width:.6f} {norm_height:.6f}\n")
        
        file_count += 1
    
    print(f"✅ Created {file_count} image symlinks and label files for {split_name} split")
    return file_count

def create_yolo_config(class_to_idx, dataset_dir):
    """Create YOLO configuration file (data.yaml) with standard directory structure."""
    
    # Create config with standard YOLO paths
    config = {
        'path': os.path.abspath(dataset_dir),
        'train': 'train/images',  # Standard YOLO path to training images
        'val': 'val/images',      # Standard YOLO path to validation images
        'test': 'test/images',    # Standard YOLO path to test images
        'nc': len(class_to_idx),
        'names': list(class_to_idx.keys())
    }
    
    config_path = os.path.join(dataset_dir, 'data.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"📄 Created YOLO config with standard structure: {config_path}")
    print(f"   📁 Train: {os.path.join(dataset_dir, 'train/images')}")
    print(f"   📁 Val: {os.path.join(dataset_dir, 'val/images')}")
    print(f"   📁 Test: {os.path.join(dataset_dir, 'test/images')}")
    return config_path

def save_dataset_info(annotations, class_to_idx, train_images, val_images, test_images, dataset_dir, image_groups):
    """Save dataset metadata for reference with original image paths."""
    
    # Create path mappings for each split
    train_paths = {img: image_groups[img][0]['image_path'] for img in train_images}
    val_paths = {img: image_groups[img][0]['image_path'] for img in val_images}
    test_paths = {img: image_groups[img][0]['image_path'] for img in test_images}
    
    dataset_info = {
        'total_annotations': len(annotations),
        'total_images': len(set(ann['image_filename'] for ann in annotations)),
        'classes': {
            'count': len(class_to_idx),
            'mapping': class_to_idx,
            'distribution': dict(Counter(ann['label'] for ann in annotations))
        },
        'splits': {
            'train': {
                'images': len(train_images), 
                'image_list': train_images,
                'image_paths': train_paths
            },
            'val': {
                'images': len(val_images), 
                'image_list': val_images,
                'image_paths': val_paths
            },
            'test': {
                'images': len(test_images), 
                'image_list': test_images,
                'image_paths': test_paths
            }
        },
        'source': {
            'split_method': 'Pre-established in 02_establish_data_splits.py',
            'split_map_file': SPLIT_MAP_PATH
        },
        'original_images_dir': os.path.abspath(IMAGES_DIR),
        'notes': 'Images are read directly from original locations - no symlinks or copies created'
    }
    
    info_path = os.path.join(dataset_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"📋 Saved dataset info: {info_path}")
    return info_path

def main():
    """Main function to prepare YOLOv8 dataset."""
    print("📋 Starting YOLOv8 dataset preparation for single-class arthropod detection...")
    print(f"📄 Annotations: {ANNOTATIONS_PATH}")
    print(f"📄 Split map: {SPLIT_MAP_PATH}")
    print(f"📁 Images: {IMAGES_DIR}")
    print(f"💾 Dataset output: {YOLO_DATASET_DIR}")
    print()
    
    # Check prerequisites
    if not os.path.exists(ANNOTATIONS_PATH):
        print(f"❌ Missing annotations file: {ANNOTATIONS_PATH}")
        print("Please run 00_prep_env.py first to download the data.")
        return 1
    
    if not os.path.exists(SPLIT_MAP_PATH):
        print(f"❌ Missing split map file: {SPLIT_MAP_PATH}")
        print("Please run 02_establish_data_splits.py first to create data splits.")
        return 1
    
    if not os.path.exists(IMAGES_DIR):
        print(f"❌ Missing images directory: {IMAGES_DIR}")
        print("Please run 00_prep_env.py first to download the data.")
        return 1
    
    # Setup directories
    setup_directories()
    
    # Load annotations
    try:
        annotations = load_annotations()
        
        if not annotations:
            print("❌ No valid annotations found to process")
            return 1
            
    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        return 1
    
    # Create class mapping
    class_to_idx, idx_to_class = create_class_mapping(annotations)
    
    # Group annotations by image
    image_groups = group_annotations_by_image(annotations)
    
    # Load pre-established splits
    train_images, val_images, test_images = load_split_map()
    
    print()
    print("🔄 Creating YOLOv8 dataset files...")
    
    # Create dataset files for each split (labels and image symlinks)
    train_count = create_dataset_files(
        image_groups, train_images, 'train', class_to_idx, YOLO_DATASET_DIR
    )
    
    val_count = create_dataset_files(
        image_groups, val_images, 'val', class_to_idx, YOLO_DATASET_DIR
    )
    
    test_count = create_dataset_files(
        image_groups, test_images, 'test', class_to_idx, YOLO_DATASET_DIR
    )
    
    # Create YOLO configuration with absolute paths
    config_path = create_yolo_config(class_to_idx, YOLO_DATASET_DIR)
    
    # Save dataset metadata
    info_path = save_dataset_info(annotations, class_to_idx, train_images, val_images, test_images, YOLO_DATASET_DIR, image_groups)
    
    print()
    print("🎉 YOLOv8 single-class arthropod dataset preparation complete!")
    print(f"   📊 Total annotations: {len(annotations)}")
    print(f"   🏷️  Classes: {len(class_to_idx)}")
    print(f"   📊 Using pre-established splits from: {SPLIT_MAP_PATH}")
    print(f"   🚂 Training images: {train_count}")
    print(f"   🔍 Validation images: {val_count}")
    print(f"   🧪 Test images: {test_count}")
    print(f"   📄 Data config: {config_path}")
    print(f"   📋 Dataset info: {info_path}")

    return 0

if __name__ == "__main__":
    exit(main())