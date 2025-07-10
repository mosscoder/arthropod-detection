#!/usr/bin/env python3
"""
01_plot_batch.py - Batch Visualization Script

Loads locally prepared arthropod data and creates visualization batches.
Uses only images where annotation filenames exactly match actual image files.
Limited to first 12 matching images for faster processing.
Saves example batch plot to results/example_batch.png.

Prerequisites:
- Run 00_prep_env.py first to download data
- Requires data/tabular/annotations.json and data/raster/petri_dish_src/
"""

import os
import json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from osgeo import gdal
import math
from pathlib import Path

# Configuration
ANNOTATIONS_PATH = "data/tabular/annotations.json"
IMAGES_DIR = "data/raster/petri_dish_src"
OUTPUT_PATH = "results/example_batch.png"
BATCH_SIZE = 12
CROP_SIZE = int(1024 * 1.5)
NCOLS = 4  # Number of columns in the plot
MAX_IMAGES = 20  # Limit to first 20 images found (buffer for filtering)

def read_subregion(image_path, crop_box):
    """
    Read a subregion of an image using GDAL.

    Parameters:
        image_path (str): Path to the image file.
        crop_box (tuple): (left, top, right, bottom) defining the crop window.

    Returns:
        np.array: The cropped region as an array. For multi-band images,
                  the output shape is (height, width, bands).
    """
    left, top, right, bottom = crop_box
    x_size = right - left
    y_size = bottom - top

    dataset = gdal.Open(image_path)
    if dataset is None:
        raise ValueError(f"Could not open image: {image_path}")
    
    # Read the subregion for all bands. This returns an array of shape (bands, height, width)
    subregion = dataset.ReadAsArray(left, top, x_size, y_size)

    # If multi-band, transpose to (height, width, bands)
    if subregion.ndim == 3:
        subregion = np.transpose(subregion, (1, 2, 0))
    return subregion

def s3_to_local(s3_path, local_dir):
    """Helper to convert an S3 URL to a local path (assumes the basename is the same)"""
    basename = os.path.basename(s3_path)
    return os.path.join(local_dir, basename)

def compute_bbox_absolute(bbox, orig_w, orig_h):
    """Convert percentage annotations to absolute pixel coordinates."""
    # x, y represent the top-left corner as a percentage
    left = (bbox['x'] / 100) * orig_w
    top = (bbox['y'] / 100) * orig_h
    width = (bbox['width'] / 100) * orig_w
    height = (bbox['height'] / 100) * orig_h
    
    # Check for rotation (optional field)
    rotation = bbox.get('rotation', 0.0)  # Default to 0 degrees if not present
    
    # Return bbox with rotation: [x_min, y_min, x_max, y_max, rotation_degrees]
    return [left, top, left + width, top + height, rotation]

def get_crop_coordinates(centroid, crop_size, image_width, image_height):
    """Given a centroid and crop size, compute a crop box that is fully within the image."""
    cx, cy = centroid
    left = int(max(0, min(cx - crop_size // 2, image_width - crop_size)))
    top = int(max(0, min(cy - crop_size // 2, image_height - crop_size)))
    return left, top, left + crop_size, top + crop_size

def adjust_bbox_to_crop(bbox, crop_box):
    """Adjust a bounding box to the crop coordinates."""
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    
    # Handle both 4-element (old format) and 5-element (with rotation) bboxes
    if len(bbox) == 5:
        x_min, y_min, x_max, y_max, rotation = bbox
    else:
        x_min, y_min, x_max, y_max = bbox
        rotation = 0.0
    
    # Find intersection between bbox and crop window
    x_min = max(x_min, crop_left)
    y_min = max(y_min, crop_top)
    x_max = min(x_max, crop_right)
    y_max = min(y_max, crop_bottom)
    
    # If there is no intersection, return None
    if x_min >= x_max or y_min >= y_max:
        return None
    
    # Translate to crop coordinates and preserve rotation
    return [x_min - crop_left, y_min - crop_top, x_max - crop_left, y_max - crop_top, rotation]

def get_first_n_images(images_dir, n=12):
    """Get the first n image files from the directory."""
    if not os.path.exists(images_dir):
        return []
    
    # Get all image files (common extensions)
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    image_files = []
    
    for filename in sorted(os.listdir(images_dir)):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(filename)
            if len(image_files) >= n:
                break
    
    print(f"📷 Found {len(image_files)} images (limited to first {n})")
    return image_files

class PetriDishDataset(Dataset):
    """Dataset for loading petri dish images with arthropod annotations"""
    
    def __init__(self, json_path, local_dir, crop_size=1024, transform=None, max_images=None):
        """
        Initialize dataset with local paths
        
        Args:
            json_path: Path to JSON annotation file
            local_dir: Local directory containing images
            crop_size: Size of crops to extract
            transform: Optional transform to apply
            max_images: Maximum number of images to process (None for all)
        """
        self.local_dir = local_dir
        self.crop_size = crop_size
        self.transform = transform
        self.records = []
        
        # Load JSON data
        print(f"📄 Loading annotations from: {json_path}")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Get all available image files for exact matching
        all_images = set()
        if os.path.exists(local_dir):
            image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
            for filename in os.listdir(local_dir):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    all_images.add(filename)
        
        print(f"📁 Found {len(all_images)} total images in directory")
        
        # Find intersection of annotation filenames and actual image files
        annotation_filenames = set()
        for item in self.data:
            if 'data' in item and 'image' in item['data']:
                annotation_filenames.add(os.path.basename(item['data']['image']))
        
        # Only use images that have exact filename matches
        matching_images = annotation_filenames.intersection(all_images)
        print(f"📄 Found {len(matching_images)} images with exact filename matches")
        
        # Debug: Show some examples of each type
        print(f"   📄 First 5 annotation filenames: {sorted(list(annotation_filenames))[:5]}")
        print(f"   📁 First 5 actual image filenames: {sorted(list(all_images))[:5]}")
        
        if matching_images:
            print(f"   ✅ Exact matches found: {sorted(matching_images)}")
        else:
            print(f"   ❌ No exact filename matches found")
            print(f"   💡 Consider running with updated image filenames or annotation file")
        
        # Limit to first N matching images (sorted for consistency)
        if max_images and len(matching_images) > max_images:
            matching_images = set(sorted(matching_images)[:max_images])
            print(f"📷 Limited to first {max_images} matching images")
        
        allowed_images = matching_images
        
        # Process annotations
        processed_count = 0
        skipped_no_data = 0
        skipped_no_annotations = 0
        skipped_no_file = 0
        skipped_not_allowed = 0
        skipped_no_bboxes = 0
        
        # Debug: Show first few annotation file references
        annotation_files = []
        for i, item in enumerate(self.data[:5]):  # First 5 items
            if 'data' in item and 'image' in item['data']:
                image_ref = item['data']['image']
                annotation_files.append(os.path.basename(image_ref))
        print(f"   📄 First few annotation file references: {annotation_files}")
        
        for item in self.data:
            if 'data' not in item or 'image' not in item['data']:
                skipped_no_data += 1
                continue
            if not item.get('annotations') or not item['annotations'][0].get('result'):
                skipped_no_annotations += 1
                continue
            
            # Get annotation filename and check for exact match
            annotation_filename = os.path.basename(item['data']['image'])
            
            # Skip if not in allowed images list (exact matches only)
            if allowed_images and annotation_filename not in allowed_images:
                skipped_not_allowed += 1
                continue
            
            # Get the actual image path
            image_path = os.path.join(local_dir, annotation_filename)
            
            # Skip if image doesn't exist locally (double-check)
            if not os.path.exists(image_path):
                skipped_no_file += 1
                continue
            
            first_result = item['annotations'][0]['result'][0]
            orig_w = first_result['original_width']
            orig_h = first_result['original_height']

            bboxes = []
            labels = []
            for ann in item['annotations']:
                if not ann.get('result'):
                    continue
                for res in ann['result']:
                    if res['type'] == 'rectanglelabels':
                        bbox = compute_bbox_absolute(res['value'], orig_w, orig_h)
                        bboxes.append(bbox)
                        labels.append(res['value']['rectanglelabels'][0])
            
            if bboxes:
                self.records.append({
                    'image_path': image_path,
                    'orig_w': orig_w,
                    'orig_h': orig_h,
                    'bboxes': bboxes,
                    'labels': labels
                })
                processed_count += 1
            else:
                skipped_no_bboxes += 1
        
        print(f"📊 Processed {processed_count} records with annotations")
        print(f"   ⏭️  Skipped {skipped_no_data} items (no data)")
        print(f"   ⏭️  Skipped {skipped_no_annotations} items (no annotations)")
        print(f"   ⏭️  Skipped {skipped_no_file} items (file not found)")
        print(f"   ⏭️  Skipped {skipped_not_allowed} items (not in exact filename matches)")
        print(f"   ⏭️  Skipped {skipped_no_bboxes} items (no bounding boxes)")
        
        if allowed_images:
            print(f"   📷 Using exact matches: {sorted(allowed_images)}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        filename = record['image_path']

        # Choose a bounding box (e.g., at random) and compute its centroid
        chosen_bbox = random.choice(record['bboxes'])
        # Handle both 4-element and 5-element bboxes
        cx = (chosen_bbox[0] + chosen_bbox[2]) / 2
        cy = (chosen_bbox[1] + chosen_bbox[3]) / 2

        # Determine the crop window centered around the chosen centroid
        crop_box = get_crop_coordinates((cx, cy), self.crop_size, record['orig_w'], record['orig_h'])

        # Use GDAL to load just the subregion (crop)
        crop = read_subregion(record['image_path'], crop_box)

        # Adjust bounding boxes to the crop coordinate space
        crop_bboxes = []
        crop_labels = []
        for bbox, label in zip(record['bboxes'], record['labels']):
            adj = adjust_bbox_to_crop(bbox, crop_box)
            if adj is not None:
                crop_bboxes.append(adj)
                crop_labels.append(label)

        if self.transform:
            crop = self.transform(crop)

        sample = {
            'image': crop,
            'bboxes': crop_bboxes,
            'labels': crop_labels,
            'filename': os.path.basename(filename)
        }
        return sample

def draw_rotated_rectangle(ax, bbox, color, label):
    """Draw a rotated rectangle on the given axis."""
    # Handle both 4-element and 5-element bboxes
    if len(bbox) == 5:
        x_min, y_min, x_max, y_max, rotation = bbox
    else:
        x_min, y_min, x_max, y_max = bbox
        rotation = 0.0
    
    # Calculate center and dimensions
    width = x_max - x_min
    height = y_max - y_min
    center_x = x_min + width / 2
    center_y = y_min + height / 2
    
    if rotation == 0.0:
        # Use simple rectangle for non-rotated boxes
        rect = patches.Rectangle((x_min, y_min), width, height,
                               linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        label_x, label_y = center_x, y_min - 5
    else:
        # Create rotated rectangle using matplotlib transforms
        from matplotlib.transforms import Affine2D
        
        # Option 1: Rotate around center (current implementation)
        # rect = patches.Rectangle((-width/2, -height/2), width, height,
        #                        linewidth=2, edgecolor=color, facecolor='none')
        # transform = Affine2D().rotate_deg(rotation).translate(center_x, center_y) + ax.transData
        
        # Option 2: Rotate around top-left corner (annotation format might expect this)
        rect = patches.Rectangle((x_min, y_min), width, height,
                               linewidth=2, edgecolor=color, facecolor='none')
        transform = Affine2D().rotate_deg_around(x_min, y_min, rotation) + ax.transData
        
        rect.set_transform(transform)
        ax.add_patch(rect)
        
        # For rotated boxes, place label at original center position
        label_x, label_y = center_x, center_y - height/2 - 5
    
    # Add label
    ax.text(label_x, label_y, label, fontsize=8, color=color,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, pad=2),
            ha='center')

def custom_collate_fn(batch):
    """Custom collate function for batching samples"""
    images = []
    bboxes = []
    labels = []
    filenames = []
    
    for sample in batch:
        # Convert the numpy array image to a torch tensor and permute to (C, H, W)
        images.append(torch.tensor(sample['image']).permute(2, 0, 1))
        bboxes.append(torch.tensor(sample['bboxes'], dtype=torch.float32))
        labels.append(sample['labels'])
        filenames.append(sample['filename'])
    
    images = torch.stack(images, dim=0)
    return {'image': images, 'bboxes': bboxes, 'labels': labels, 'filenames': filenames}

def plot_batch(batch, ncols=3, save_path=None):
    """
    Plots a batch of images with bounding boxes and labels, displaying filenames above each image.
    
    Args:
        batch (dict): A dictionary containing:
                      'image': list of tensors of shape (C, H, W),
                      'bboxes': list of bounding box coordinates,
                      'labels': list of class labels,
                      'filenames': list of filenames.
        ncols (int): Number of columns in the plot.
        save_path (str): Path to save the plot. If None, displays the plot.
    """
    # Create a color mapping for each unique class label in the batch
    all_labels = [label for labels in batch['labels'] for label in labels]
    unique_classes = list(set(all_labels))

    # Use a predefined set of colors from TABLEAU_COLORS
    colors = list(mcolors.TABLEAU_COLORS.values())
    class_to_color = {cls: colors[i % len(colors)] for i, cls in enumerate(unique_classes)}

    bs = len(batch['image'])
    nrows = math.ceil(bs / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i in range(nrows * ncols):
        if i < bs:
            ax = axes[i]
            img = batch['image'][i]
            bboxes = batch['bboxes'][i]
            labels = batch['labels'][i]
            filename = batch['filenames'][i]

            # Convert image tensor from (C, H, W) to (H, W, C)
            img_np = img.permute(1, 2, 0).numpy()
            
            # Ensure image values are in valid range
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            ax.imshow(img_np)

            # Display the filename at the top of the image
            ax.set_title(filename, fontsize=8, pad=10)

            # Draw bounding boxes and labels (with rotation support)
            for bbox, label in zip(bboxes, labels):
                color = class_to_color[label]
                draw_rotated_rectangle(ax, bbox, color, label)
            ax.axis('off')
        else:
            axes[i].axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved batch plot to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def check_prerequisites():
    """Check if required data files exist"""
    missing_files = []
    
    if not os.path.exists(ANNOTATIONS_PATH):
        missing_files.append(ANNOTATIONS_PATH)
    
    if not os.path.exists(IMAGES_DIR):
        missing_files.append(IMAGES_DIR)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print()
        print("Please run 00_prep_env.py first to download the data.")
        return False
    
    return True

def main():
    """Main function to create and save batch visualization"""
    print("🎨 Starting batch visualization...")
    print(f"📄 Annotations: {ANNOTATIONS_PATH}")
    print(f"📁 Images: {IMAGES_DIR}")
    print(f"💾 Output: {OUTPUT_PATH}")
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        return 1
    
    # Create output directory
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    try:
        dataset = PetriDishDataset(
            json_path=ANNOTATIONS_PATH,
            local_dir=IMAGES_DIR,
            crop_size=CROP_SIZE,
            max_images=MAX_IMAGES
        )
        
        if len(dataset) == 0:
            print("❌ No valid samples found in dataset")
            return 1
        
        print(f"📊 Dataset loaded with {len(dataset)} samples")
        
        # Check if we have enough samples for the requested batch size
        if len(dataset) < BATCH_SIZE:
            print(f"⚠️  Warning: Dataset has only {len(dataset)} samples, but batch size is {BATCH_SIZE}")
            print(f"   💡 Consider increasing MAX_IMAGES buffer or reducing BATCH_SIZE")
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return 1
    
    # Create dataloader
    try:
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=8, 
            collate_fn=custom_collate_fn
        )
        
        print(f"🔄 Created dataloader with batch size {BATCH_SIZE}")
        
    except Exception as e:
        print(f"❌ Error creating dataloader: {e}")
        return 1
    
    # Generate and save batch
    try:
        print("🎯 Generating batch...")
        batch = next(iter(dataloader))
        
        print(f"📊 Batch contains {len(batch['image'])} samples")
        
        # Count labels in batch
        label_counts = {}
        for labels in batch['labels']:
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"🏷️  Labels in batch: {dict(label_counts)}")
        
        # Create and save plot
        plot_batch(batch, ncols=NCOLS, save_path=OUTPUT_PATH)
        
    except Exception as e:
        print(f"❌ Error generating batch: {e}")
        return 1
    
    print()
    print("🎉 Batch visualization complete!")
    print(f"   💾 Saved to: {OUTPUT_PATH}")
    
    return 0

if __name__ == "__main__":
    exit(main())