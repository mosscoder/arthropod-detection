#!/usr/bin/env python3
"""
03_process_crops.py - Crop Extraction Pipeline

Extracts individual rectangular thumbnails for each annotation with proper rotation handling.
Handles rotation by calculating envelope sizes and applying proper affine transformations.
Produces rectangular crops (not square) that show annotations upright without black corners.
Metadata is managed separately in 02_establish_data_splits.py output files.

Prerequisites:
- Run 00_prep_env.py first to download data
- Run 02_establish_data_splits.py to create split metadata
- Requires data/tabular/annotations.json and data/raster/petri_dish_src/

Output:
- data/raster/crops/{id}.png - Individual crop images (rectangular, no rescaling)

Note: Metadata is available in data/tabular/annotations_split.json (created by 02_establish_data_splits.py)
"""

import os
import json
import uuid
import math
import numpy as np
from PIL import Image, ImageOps
from osgeo import gdal
from pathlib import Path
import multiprocessing as mp
from functools import partial


# Configuration
ANNOTATIONS_PATH = "data/tabular/annotations.json"
IMAGES_DIR = "data/raster/petri_dish_src"
CROPS_OUTPUT_DIR = "data/raster/crops"

def read_subregion(image_path, crop_box):
    """
    Read a subregion of an image using GDAL.
    
    Parameters:
        image_path (str): Path to the image file.
        crop_box (tuple): (left, top, right, bottom) defining the crop window.
    
    Returns:
        np.array: The cropped region as an array.
    """
    left, top, right, bottom = crop_box
    x_size = right - left
    y_size = bottom - top
    
    dataset = gdal.Open(image_path)
    if dataset is None:
        raise ValueError(f"Could not open image: {image_path}")
    
    # Read the subregion for all bands
    subregion = dataset.ReadAsArray(left, top, x_size, y_size)
    
    # If multi-band, transpose to (height, width, bands)
    if subregion.ndim == 3:
        subregion = np.transpose(subregion, (1, 2, 0))
    
    return subregion

def compute_bbox_absolute(bbox, orig_w, orig_h):
    """Convert percentage annotations to absolute pixel coordinates."""
    # x, y represent the top-left corner as a percentage
    left = (bbox['x'] / 100) * orig_w
    top = (bbox['y'] / 100) * orig_h
    width = (bbox['width'] / 100) * orig_w
    height = (bbox['height'] / 100) * orig_h
    
    # Check for rotation (optional field)
    rotation = bbox.get('rotation', 0.0)
    
    return {
        'left': left,
        'top': top,
        'width': width,
        'height': height,
        'rotation': rotation
    }

def compute_rotated_corners(left, top, width, height, rotation_degrees):
    """
    Compute the 4 corners of a bounding box after rotation around top-left corner.
    
    Args:
        left (float): Left coordinate of bounding box
        top (float): Top coordinate of bounding box  
        width (float): Width of bounding box
        height (float): Height of bounding box
        rotation_degrees (float): Rotation angle in degrees
        
    Returns:
        list: List of (x, y) tuples for the 4 corners
    """
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

def compute_exact_envelope(corners):
    """
    Compute the exact axis-aligned bounding box that contains all corners.
    
    Args:
        corners (list): List of (x, y) tuples
        
    Returns:
        tuple: (min_x, min_y, max_x, max_y)
    """
    x_coords = [corner[0] for corner in corners]
    y_coords = [corner[1] for corner in corners]
    
    return (
        min(x_coords),
        min(y_coords), 
        max(x_coords),
        max(y_coords)
    )

def extract_envelope_crop(image_path, bbox_abs):
    """
    Extract a crop using axis-aligned envelope that exactly contains the rotated bounding box.
    
    Args:
        image_path (str): Path to the source image
        bbox_abs (dict): Absolute bounding box coordinates with rotation
        
    Returns:
        tuple: (crop_image, envelope_coords) or (None, None) if extraction fails
    """
    left = bbox_abs['left']
    top = bbox_abs['top']
    width = bbox_abs['width']
    height = bbox_abs['height']
    rotation = bbox_abs['rotation']
    
    try:
        # Compute rotated corners
        corners = compute_rotated_corners(left, top, width, height, rotation)
        
        # Compute exact axis-aligned envelope
        min_x, min_y, max_x, max_y = compute_exact_envelope(corners)
        
        # Create crop box with integer coordinates
        crop_box = (
            int(max(0, min_x)),
            int(max(0, min_y)),
            int(max_x),
            int(max_y)
        )
        
        # Extract the crop
        crop_array = read_subregion(image_path, crop_box)
        if crop_array is None:
            return None, None
            
        crop_image = Image.fromarray(crop_array.astype(np.uint8))
        
        # Return the crop and envelope coordinates for metadata
        envelope_coords = {
            'envelope_left': min_x,
            'envelope_top': min_y,
            'envelope_width': max_x - min_x,
            'envelope_height': max_y - min_y
        }
        
        return crop_image, envelope_coords
        
    except Exception as e:
        print(f"Error extracting envelope crop: {e}")
        return None, None

def process_single_annotation(args):
    """Process a single annotation to extract crop."""
    annotation, image_path, orig_w, orig_h, label = args
    
    # Generate unique ID for this crop
    crop_id = str(uuid.uuid4())
    
    # Compute absolute bounding box
    bbox_abs = compute_bbox_absolute(annotation, orig_w, orig_h)
    
    # Extract crop using envelope method
    crop_image, envelope_coords = extract_envelope_crop(image_path, bbox_abs)
    
    if crop_image is None:
        return False
    
    # Save crop
    crop_filename = f"{crop_id}.png"
    crop_path = os.path.join(CROPS_OUTPUT_DIR, crop_filename)
    
    try:
        crop_image.save(crop_path, 'PNG')
        return True
        
    except Exception as e:
        print(f"Error saving crop {crop_id}: {e}")
        return False

def load_annotations():
    """Load and parse the annotations JSON file."""
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
    
    # Only use images that have exact filename matches
    matching_images = annotation_filenames.intersection(all_images)
    print(f"📄 Found {len(matching_images)} images with exact filename matches")
    
    # Process annotations to extract individual bbox tasks
    tasks = []
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
                    
                    tasks.append((bbox, image_path, orig_w, orig_h, label))
                    processed_count += 1
    
    print(f"📊 Found {processed_count} individual annotations to process")
    print(f"   ⏭️  Skipped {skipped_count} items (no data, annotations, or file not found)")
    
    return tasks

def main():
    """Main function to extract crops from all annotations."""
    print("🔪 Starting crop extraction pipeline...")
    print(f"📄 Annotations: {ANNOTATIONS_PATH}")
    print(f"📁 Images: {IMAGES_DIR}")
    print(f"💾 Crops output: {CROPS_OUTPUT_DIR}")
    print()
    
    # Check prerequisites
    if not os.path.exists(ANNOTATIONS_PATH):
        print(f"❌ Missing annotations file: {ANNOTATIONS_PATH}")
        print("Please run 00_prep_env.py first to download the data.")
        return 1
    
    if not os.path.exists(IMAGES_DIR):
        print(f"❌ Missing images directory: {IMAGES_DIR}")
        print("Please run 00_prep_env.py first to download the data.")
        return 1
    
    # Create output directories
    Path(CROPS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load annotations and create tasks
    try:
        tasks = load_annotations()
        
        if not tasks:
            print("❌ No valid annotations found to process")
            return 1
            
    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        return 1
    
    # Process crops using multiprocessing
    print(f"🔄 Processing {len(tasks)} crops using {mp.cpu_count()-1} cores...")
    
    try:
        with mp.Pool(processes=mp.cpu_count()-1) as pool:
            results = pool.map(process_single_annotation, tasks)
        
        # Count successful results
        successful_count = sum(1 for r in results if r is True)
        failed_count = len(results) - successful_count
        
        print(f"✅ Successfully processed {successful_count} crops")
        print(f"❌ Failed to process {failed_count} crops")
        
    except Exception as e:
        print(f"❌ Error during parallel processing: {e}")
        return 1
    
    print()
    print("🎉 Crop extraction complete!")
    print(f"   📁 {successful_count} crops saved to: {CROPS_OUTPUT_DIR}")
    print(f"   📊 Metadata available in: data/tabular/annotations_split.json")
    print(f"   🗂️  Split information in: data/tabular/split_map.json")
    
    return 0

if __name__ == "__main__":
    exit(main())