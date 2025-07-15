#!/usr/bin/env python3
"""
04_yolo_prep.py - Standard Tiled YOLOv8 Dataset Preparation for Arthropod Detection

Prepares standard tiled YOLOv8 dataset from arthropod annotation data:
- Loads annotations following 02_process_crops.py conventions
- Creates single class mapping for arthropod detection (agnostic to taxonomic groups)
- Uses pre-established splits from 02_establish_data_splits.py
- Resizes images to 5120px (max dimension) while maintaining aspect ratio
- Training data: Tiles resized images into 1280x1280 squares with 50% overlap
- Validation data: Tiles resized images into 1280x1280 squares with 10% overlap
- Test data: Saves resized images (no tiling)
- Splits annotations appropriately across tiles based on bounding box intersections
- Converts rotated bounding boxes to axis-aligned format
- All arthropod types are treated as a single 'arthropod' class

Key Features:
- Resizes images to 5120px before tiling to ensure consistent scale
- Standard tiling approach for both training and validation
- Different overlap ratios for train (50%) vs val (10%)
- Direct tile saving without canvas complications
- Labels normalized to tile dimensions
- Handles annotations that span multiple tiles
- Filters out tiles with no annotations to reduce dataset size
- Preserves image quality through high-quality resizing
- Requires OpenCV (cv2) for fast image processing

Prerequisites:
- Run 00_prep_env.py first to download data
- Run 02_establish_data_splits.py to create train/val/test splits
- Requires data/tabular/annotations.json, data/tabular/split_map.json, and data/raster/petri_dish_src/

Output:
- data/yolo_dataset/train/images/ - Training tile images
- data/yolo_dataset/train/labels/ - YOLO format training labels
- data/yolo_dataset/val/images/ - Validation tile images
- data/yolo_dataset/val/labels/ - YOLO format validation labels
- data/yolo_dataset/test/images/ - Test resized images
- data/yolo_dataset/test/labels/ - YOLO format test labels
- data/yolo_dataset/data.yaml - YOLO config with standard paths
- data/yolo_dataset/dataset_info.json - Dataset metadata
- data/yolo_dataset/tile_mapping.json - Mapping of tiles to source images
"""

import os
import json
import yaml
import math
import multiprocessing
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import cv2

# Configuration
ANNOTATIONS_PATH = "data/tabular/annotations.json"
SPLIT_MAP_PATH = "data/tabular/split_map.json"
IMAGES_DIR = "data/raster/petri_dish_src"
YOLO_DATASET_DIR = "data/yolo_dataset"
RESIZE_SIZE = 5120  # Target size for resizing images before tiling
TILE_SIZE = 1280  # Size of each tile in pixels
TRAIN_OVERLAP_RATIO = 0.5  # 50% overlap for training
VAL_OVERLAP_RATIO = 0.1  # 10% overlap for validation
MIN_INTERSECTION_RATIO = 0.1  # Minimum 10% of bbox area must be in tile to include it

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

def calculate_tiles(image_width, image_height, tile_size, overlap_ratio):
    """Calculate tile positions for an image with overlap."""
    overlap = int(tile_size * overlap_ratio)
    stride = tile_size - overlap
    
    tiles = []
    tile_id = 0
    
    # Calculate number of tiles needed
    n_tiles_x = math.ceil((image_width - overlap) / stride)
    n_tiles_y = math.ceil((image_height - overlap) / stride)
    
    for row in range(n_tiles_y):
        for col in range(n_tiles_x):
            x = col * stride
            y = row * stride
            
            # Adjust last tiles to fit within image bounds
            if x + tile_size > image_width:
                x = image_width - tile_size
            if y + tile_size > image_height:
                y = image_height - tile_size
            
            # Ensure we don't go negative
            x = max(0, x)
            y = max(0, y)
            
            tiles.append({
                'id': tile_id,
                'x': x,
                'y': y,
                'width': min(tile_size, image_width - x),
                'height': min(tile_size, image_height - y),
                'row': row,
                'col': col
            })
            tile_id += 1
    
    return tiles

def get_bbox_tile_intersection(bbox_left, bbox_top, bbox_right, bbox_bottom, tile):
    """Calculate intersection between bbox and tile, return relative coordinates."""
    # Calculate intersection
    intersect_left = max(bbox_left, tile['x'])
    intersect_top = max(bbox_top, tile['y'])
    intersect_right = min(bbox_right, tile['x'] + tile['width'])
    intersect_bottom = min(bbox_bottom, tile['y'] + tile['height'])
    
    # Check if there's an intersection
    if intersect_left >= intersect_right or intersect_top >= intersect_bottom:
        return None
    
    # Calculate intersection area ratio
    bbox_area = (bbox_right - bbox_left) * (bbox_bottom - bbox_top)
    intersect_area = (intersect_right - intersect_left) * (intersect_bottom - intersect_top)
    intersection_ratio = intersect_area / bbox_area if bbox_area > 0 else 0
    
    # Convert to tile-relative coordinates
    rel_left = intersect_left - tile['x']
    rel_top = intersect_top - tile['y']
    rel_right = intersect_right - tile['x']
    rel_bottom = intersect_bottom - tile['y']
    
    return {
        'left': rel_left,
        'top': rel_top,
        'right': rel_right,
        'bottom': rel_bottom,
        'intersection_ratio': intersection_ratio
    }

def process_image_tiles(image_path, annotations, tile_size, overlap_ratio, min_intersection_ratio):
    """Process a single image into tiles with annotations."""
    # Open image to get dimensions using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not open image: {image_path}")
    original_height, original_width = img.shape[:2]
    
    # Calculate scale factor for resizing
    scale_factor = RESIZE_SIZE / max(original_width, original_height)
    
    # Calculate new dimensions maintaining aspect ratio
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # Use resized dimensions for tiling
    image_width = new_width
    image_height = new_height
    
    # Calculate tiles based on resized dimensions
    tiles = calculate_tiles(image_width, image_height, tile_size, overlap_ratio)
    
    # Process each tile
    tiles_with_annotations = []
    
    for tile in tiles:
        tile_annotations = []
        
        # Check each annotation for intersection with this tile
        for ann in annotations:
            bbox = ann['bbox']
            
            # Scale bbox coordinates to match resized image
            scaled_bbox = {
                'x': bbox['x'] * scale_factor,
                'y': bbox['y'] * scale_factor,
                'width': bbox['width'] * scale_factor,
                'height': bbox['height'] * scale_factor,
                'rotation': bbox['rotation']
            }
            
            # Convert to axis-aligned bbox
            bbox_left, bbox_top, bbox_right, bbox_bottom = bbox_to_axis_aligned(scaled_bbox)
            
            # Get intersection with tile
            intersection = get_bbox_tile_intersection(
                bbox_left, bbox_top, bbox_right, bbox_bottom, tile
            )
            
            if intersection and intersection['intersection_ratio'] >= min_intersection_ratio:
                tile_annotations.append({
                    'bbox': intersection,
                    'label': ann['label'],
                    'original_bbox': bbox,
                    'scaled_bbox': scaled_bbox
                })
        
        # Only keep tiles with annotations
        if tile_annotations:
            tiles_with_annotations.append({
                'tile': tile,
                'annotations': tile_annotations,
                'source_image': image_path,
                'scale_factor': scale_factor,
                'resized_dimensions': (image_width, image_height)
            })
    
    return tiles_with_annotations

def save_tile_image(source_image_path, tile, output_path, scale_factor):
    """Extract and save tile using OpenCV."""
    # Read image
    img = cv2.imread(source_image_path)
    if img is None:
        raise ValueError(f"Could not read image: {source_image_path}")
    
    # Calculate new dimensions
    original_height, original_width = img.shape[:2]
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # Resize image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Extract tile
    tile_img = resized_img[
        tile['y']:tile['y'] + tile['height'],
        tile['x']:tile['x'] + tile['width']
    ]
    
    # Save tile with high quality
    cv2.imwrite(output_path, tile_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

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

def process_single_image_worker(args):
    """Worker function to process a single image into tiles."""
    image_filename, image_groups, split_name, class_to_idx, dataset_dir, overlap_ratio = args
    
    if image_filename not in image_groups:
        return []
    
    annotations = image_groups[image_filename]
    image_path = annotations[0]['image_path']
    
    # Process image into tiles
    tiles_data = process_image_tiles(
        image_path, 
        annotations,
        TILE_SIZE,
        overlap_ratio,
        MIN_INTERSECTION_RATIO
    )
    
    if not tiles_data:
        return []
    
    results = []
    
    # Get scale factor from first tile (same for all tiles of an image)
    scale_factor = tiles_data[0]['scale_factor']
    
    # Read and resize image once with OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"   Failed to read image: {image_path}")
        return []
    
    original_height, original_width = img.shape[:2]
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Process all tiles from the resized image
    for tile_data in tiles_data:
        tile = tile_data['tile']
        tile_annotations = tile_data['annotations']
        
        # Create unique tile filename
        base_name = Path(image_filename).stem
        tile_filename = f"{base_name}_tile_{tile['row']}_{tile['col']}.jpg"
        
        # Prepare paths
        images_dir = os.path.join(dataset_dir, split_name, 'images')
        labels_dir = os.path.join(dataset_dir, split_name, 'labels')
        tile_image_path = os.path.join(images_dir, tile_filename)
        
        # Check if tile already exists
        if os.path.exists(tile_image_path):
            print(f"   Tile already exists, skipping: {tile_filename}")
        else:
            # Extract and save tile directly from resized image
            tile_img = resized_img[
                tile['y']:tile['y'] + tile['height'],
                tile['x']:tile['x'] + tile['width']
            ]
            cv2.imwrite(tile_image_path, tile_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Create label file
        label_filename = f"{base_name}_tile_{tile['row']}_{tile['col']}.txt"
        label_path = os.path.join(labels_dir, label_filename)
        
        # Check if label file already exists
        if os.path.exists(label_path):
            print(f"   Label already exists, skipping: {label_filename}")
        else:
            # Create new label file with coordinates normalized to tile size
            with open(label_path, 'w') as f:
                for ann in tile_annotations:
                    bbox = ann['bbox']
                    
                    # Convert to normalized YOLO format (relative to tile dimensions)
                    center_x = (bbox['left'] + bbox['right']) / 2 / tile['width']
                    center_y = (bbox['top'] + bbox['bottom']) / 2 / tile['height']
                    norm_width = (bbox['right'] - bbox['left']) / tile['width']
                    norm_height = (bbox['bottom'] - bbox['top']) / tile['height']
                    
                    # Ensure coordinates are within bounds
                    center_x = max(0, min(1, center_x))
                    center_y = max(0, min(1, center_y))
                    norm_width = max(0, min(1, norm_width))
                    norm_height = max(0, min(1, norm_height))
                    
                    # All arthropods get class index 0
                    class_idx = 0
                    
                    f.write(f"{class_idx} {center_x:.6f} {center_y:.6f} "
                           f"{norm_width:.6f} {norm_height:.6f}\n")
        
        # Return tile mapping info
        results.append({
            'tile_filename': tile_filename,
            'source_image': image_filename,
            'tile_info': tile,
            'split': split_name
        })
    
    return results

def create_dataset_files(image_groups, image_list, split_name, class_to_idx, dataset_dir, tile_mapping, use_tiling=True, overlap_ratio=0.1):
    """Generate tile images and YOLO format labels using multiprocessing, or resized images for test."""
    
    if use_tiling:
        print(f"🔄 Creating tiles for {split_name} split with {int(overlap_ratio*100)}% overlap...")
        
        # Create subdirectories for this split
        images_dir = os.path.join(dataset_dir, split_name, 'images')
        labels_dir = os.path.join(dataset_dir, split_name, 'labels')
        
        # Determine number of CPU cores to use
        n_cores = max(1, multiprocessing.cpu_count() - 1)
        print(f"   Using {n_cores} CPU cores for parallel processing")
        
        total_images = len(image_list)
        
        # Prepare arguments for worker function
        worker_args = [
            (image_filename, image_groups, split_name, class_to_idx, dataset_dir, overlap_ratio)
            for image_filename in image_list
        ]
        
        # Process images in parallel
        tile_results = []
        with multiprocessing.Pool(n_cores) as pool:
            # Use imap_unordered for better progress tracking
            for idx, results in enumerate(pool.imap_unordered(process_single_image_worker, worker_args)):
                tile_results.extend(results)
                # Show progress
                if (idx + 1) % 10 == 0 or (idx + 1) == total_images:
                    print(f"   Processed {idx + 1}/{total_images} images...")
        
        # Update tile mapping with results
        for result in tile_results:
            tile_mapping[result['tile_filename']] = {
                'source_image': result['source_image'],
                'tile_info': result['tile_info'],
                'split': result['split']
            }
        
        tile_count = len(tile_results)
        print(f"✅ Created {tile_count} tiles for {split_name} split")
        return tile_count
    else:
        print(f"🔄 Creating resized images for {split_name} split...")
        
        # Create subdirectories for this split
        images_dir = os.path.join(dataset_dir, split_name, 'images')
        labels_dir = os.path.join(dataset_dir, split_name, 'labels')
        
        # Create resized images and generate labels
        return create_resized_test_images(image_groups, image_list, split_name, class_to_idx, dataset_dir)

def create_resized_test_images(image_groups, image_list, split_name, class_to_idx, dataset_dir):
    """Create resized images and generate YOLO labels for test split."""
    images_dir = os.path.join(dataset_dir, split_name, 'images')
    labels_dir = os.path.join(dataset_dir, split_name, 'labels')
    
    total_images = len(image_list)
    
    for idx, image_filename in enumerate(image_list):
        # Create resized image
        source_image_path = os.path.join(IMAGES_DIR, image_filename)
        target_image_path = os.path.join(images_dir, image_filename)
        
        # Create resized image if source exists
        if os.path.exists(source_image_path):
            # Remove existing image if it exists
            if os.path.exists(target_image_path):
                os.unlink(target_image_path)
            
            # Read and resize image
            img = cv2.imread(source_image_path)
            if img is None:
                print(f"   Failed to read image: {source_image_path}")
                continue
            
            # Calculate resize dimensions
            original_height, original_width = img.shape[:2]
            scale_factor = RESIZE_SIZE / max(original_width, original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            # Resize and save image
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(target_image_path, resized_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Generate YOLO label file for this image
        if image_filename in image_groups:
            annotations = image_groups[image_filename]
            
            # Get original image dimensions for scale factor calculation
            original_height, original_width = img.shape[:2]
            scale_factor = RESIZE_SIZE / max(original_width, original_height)
            
            # Create YOLO label file
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_filename)
            
            with open(label_path, 'w') as f:
                for annotation in annotations:
                    # Convert rotated bbox to axis-aligned (returns pixel coordinates)
                    bbox = annotation['bbox']
                    # Scale bbox coordinates to match resized image
                    scaled_bbox = {
                        'x': bbox['x'] * scale_factor,
                        'y': bbox['y'] * scale_factor,
                        'width': bbox['width'] * scale_factor,
                        'height': bbox['height'] * scale_factor,
                        'rotation': bbox.get('rotation', 0.0)
                    }
                    x_min_px, y_min_px, x_max_px, y_max_px = bbox_to_axis_aligned(scaled_bbox)
                    
                    # Convert to YOLO format (center coordinates, normalized 0-1)
                    # Normalize based on resized dimensions
                    resized_width = int(original_width * scale_factor)
                    resized_height = int(original_height * scale_factor)
                    center_x = (x_min_px + x_max_px) / 2 / resized_width
                    center_y = (y_min_px + y_max_px) / 2 / resized_height
                    width = (x_max_px - x_min_px) / resized_width
                    height = (y_max_px - y_min_px) / resized_height
                    
                    # Ensure coordinates are within bounds [0, 1]
                    center_x = max(0, min(1, center_x))
                    center_y = max(0, min(1, center_y))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    # All arthropods are class 0 (single-class detection)
                    class_idx = 0
                    
                    # Write YOLO format line
                    f.write(f"{class_idx} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        # Show progress
        if (idx + 1) % 10 == 0 or (idx + 1) == total_images:
            print(f"   Processed {idx + 1}/{total_images} images...")
    
    print(f"✅ Created {total_images} resized images for {split_name} split")
    return total_images

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

def save_dataset_info(annotations, class_to_idx, train_count, val_count, test_count, dataset_dir, train_images, val_images, test_images):
    """Save dataset metadata for tiled dataset."""
    
    dataset_info = {
        'total_original_annotations': len(annotations),
        'total_original_images': len(set(ann['image_filename'] for ann in annotations)),
        'resize_size': RESIZE_SIZE,
        'tile_size': TILE_SIZE,
        'train_overlap_ratio': TRAIN_OVERLAP_RATIO,
        'val_overlap_ratio': VAL_OVERLAP_RATIO,
        'min_intersection_ratio': MIN_INTERSECTION_RATIO,
        'tiles': {
            'train': train_count,
            'val': val_count,
            'test': test_count,
            'total': train_count + val_count + test_count
        },
        'original_splits': {
            'train': len(train_images),
            'val': len(val_images),
            'test': len(test_images)
        },
        'classes': {
            'count': len(class_to_idx),
            'mapping': class_to_idx,
            'distribution': dict(Counter(ann['label'] for ann in annotations))
        },
        'source': {
            'split_method': 'Pre-established in 02_establish_data_splits.py',
            'split_map_file': SPLIT_MAP_PATH,
            'tiling_method': f'Resize to {RESIZE_SIZE}px then {TILE_SIZE}x{TILE_SIZE} tiles with {int(TRAIN_OVERLAP_RATIO*100)}% overlap (train) and {int(VAL_OVERLAP_RATIO*100)}% overlap (val)'
        },
        'original_images_dir': os.path.abspath(IMAGES_DIR),
        'notes': f'Images resized to {RESIZE_SIZE}px before tiling into {TILE_SIZE}x{TILE_SIZE} tiles'
    }
    
    info_path = os.path.join(dataset_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"📋 Saved dataset info: {info_path}")
    return info_path

def main():
    """Main function to prepare tiled YOLOv8 dataset."""
    print("🧩 Starting tiled YOLOv8 dataset preparation...")
    print(f"🔄 Resize images to: {RESIZE_SIZE}px (max dimension)")
    print(f"📐 Tile size: {TILE_SIZE}x{TILE_SIZE} pixels")
    print(f"🔄 Train overlap: {int(TRAIN_OVERLAP_RATIO * 100)}%")
    print(f"🔄 Val overlap: {int(VAL_OVERLAP_RATIO * 100)}%")
    print(f"📄 Annotations: {ANNOTATIONS_PATH}")
    print(f"📄 Split map: {SPLIT_MAP_PATH}")
    print(f"📁 Images: {IMAGES_DIR}")
    print(f"💾 Dataset output: {YOLO_DATASET_DIR}")
    
    print("🚀 Using OpenCV for fast image processing")
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
    print("🔄 Creating tiled YOLOv8 dataset...")
    
    # Create tile mapping to track tile origins
    tile_mapping = {}
    
    # Create dataset files for each split
    train_count = create_dataset_files(
        image_groups, train_images, 'train', class_to_idx, YOLO_DATASET_DIR, tile_mapping, 
        use_tiling=True, overlap_ratio=TRAIN_OVERLAP_RATIO
    )
    
    val_count = create_dataset_files(
        image_groups, val_images, 'val', class_to_idx, YOLO_DATASET_DIR, tile_mapping, 
        use_tiling=True, overlap_ratio=VAL_OVERLAP_RATIO
    )
    
    test_count = create_dataset_files(
        image_groups, test_images, 'test', class_to_idx, YOLO_DATASET_DIR, tile_mapping, 
        use_tiling=False
    )
    
    # Save tile mapping
    tile_mapping_path = os.path.join(YOLO_DATASET_DIR, 'tile_mapping.json')
    with open(tile_mapping_path, 'w') as f:
        json.dump(tile_mapping, f, indent=2)
    print(f"📋 Saved tile mapping: {tile_mapping_path}")
    
    # Create YOLO configuration with absolute paths
    config_path = create_yolo_config(class_to_idx, YOLO_DATASET_DIR)
    
    # Save dataset metadata
    info_path = save_dataset_info(annotations, class_to_idx, train_count, val_count, test_count, 
                                 YOLO_DATASET_DIR, train_images, val_images, test_images)
    
    print()
    print("🎉 YOLOv8 dataset preparation complete!")
    print(f"   📊 Original annotations: {len(annotations)}")
    print(f"   🚂 Training tiles ({int(TRAIN_OVERLAP_RATIO*100)}% overlap): {train_count}")
    print(f"   🔍 Validation tiles ({int(VAL_OVERLAP_RATIO*100)}% overlap): {val_count}")
    print(f"   🧪 Test images (resized): {test_count}")
    print(f"   📄 Data config: {config_path}")
    print(f"   📋 Dataset info: {info_path}")
    print(f"   🗺️  Tile mapping: {tile_mapping_path}")

    return 0

if __name__ == "__main__":
    exit(main())