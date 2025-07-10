#!/usr/bin/env python3
"""
02_establish_data_splits.py - Data Splits

Creates train/val/test splits of the dataset for arthropod detection.

What it does:
- Loads annotations and creates train/val/test splits
- For all images with labels, splits data into train/val/test sets (70/20/10 by images)
- Stratify by label, if a class has fewer than three images, assign to the training set
- If has exactly three, assign one to each split
- Saves a data/tabular/split_map.json file that maps each image to its split
- This is used for YOLOv8 and DINOv2 crop training

Prerequisites:
- Run 00_prep_env.py first to download data
- Requires data/tabular/annotations.json

Output:
- data/tabular/annotations_split.json - Annotation data with split labels
- data/tabular/split_map.json - Maps each image to its split assignment
"""

import os
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

# Configuration
ANNOTATIONS_PATH = "data/tabular/annotations.json"
OUTPUT_ANNOTATIONS_PATH = "data/tabular/annotations_split.json"
OUTPUT_SPLIT_MAP_PATH = "data/tabular/split_map.json"
IMAGES_DIR = "data/raster/petri_dish_src"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

def load_annotations():
    """Load and parse annotations."""
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
    
    # Filter annotations to only include images that exist
    filtered_annotations = []
    for item in data:
        if 'data' in item and 'image' in item['data']:
            filename = os.path.basename(item['data']['image'])
            if filename in matching_images:
                filtered_annotations.append(item)
    
    print(f"✅ Filtered to {len(filtered_annotations)} annotations with matching images")
    return filtered_annotations

def extract_image_labels(annotations):
    """Extract labels for each image."""
    image_labels = defaultdict(set)
    
    for item in annotations:
        if not item.get('annotations') or not item['annotations'][0].get('result'):
            continue
            
        filename = os.path.basename(item['data']['image'])
        
        # Extract labels from all annotations for this image
        for ann in item['annotations']:
            if not ann.get('result'):
                continue
                
            for res in ann['result']:
                if res['type'] == 'rectanglelabels':
                    label = res['value']['rectanglelabels'][0]
                    if label != 'label':  # Skip header artifacts
                        image_labels[filename].add(label)
    
    # Convert sets to lists for JSON serialization
    image_labels = {img: list(labels) for img, labels in image_labels.items()}
    
    print(f"📊 Found labels for {len(image_labels)} images")
    return image_labels

def analyze_class_distribution(image_labels):
    """Analyze class distribution across images."""
    class_image_counts = defaultdict(int)
    
    for filename, labels in image_labels.items():
        for label in labels:
            class_image_counts[label] += 1
    
    print(f"\n📊 Class distribution across images:")
    for label, count in sorted(class_image_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {label}: {count} images")
    
    return class_image_counts

def stratified_split(image_labels, class_image_counts):
    """Create stratified splits by label with special handling for rare classes."""
    
    train_images = set()
    val_images = set()
    test_images = set()
    
    print(f"\n🎯 Applying stratified splitting strategy:")
    
    # Step 1: Handle rare classes first (< 3 images) - all go to train
    rare_classes = {label for label, count in class_image_counts.items() if count < 3}
    if rare_classes:
        print(f"   📌 Rare classes (< 3 images): {rare_classes}")
        for filename, labels in image_labels.items():
            if any(label in rare_classes for label in labels):
                train_images.add(filename)
                print(f"   📌 Assigned {filename} to train (contains rare class)")
    
    # Step 2: Handle classes with exactly 3 images - one to each split
    three_image_classes = {label for label, count in class_image_counts.items() if count == 3}
    if three_image_classes:
        print(f"   ⚖️  Classes with exactly 3 images: {three_image_classes}")
        
        for label in three_image_classes:
            # Find images with this label that haven't been assigned yet
            candidate_images = []
            for filename, labels in image_labels.items():
                if label in labels and filename not in train_images:
                    candidate_images.append(filename)
            
            if len(candidate_images) >= 3:
                # Shuffle for random assignment
                random.shuffle(candidate_images)
                train_images.add(candidate_images[0])
                val_images.add(candidate_images[1])
                test_images.add(candidate_images[2])
                print(f"   ⚖️  {label}: distributed 1 to each split")
    
    # Step 3: For remaining classes (>3 images), ensure at least one in each split
    # then do stratified sampling
    remaining_classes = {label for label, count in class_image_counts.items() 
                        if count > 3 and label not in rare_classes and label not in three_image_classes}
    
    # Track which classes need representation in each split
    class_split_needs = {label: {'train': True, 'val': True, 'test': True} for label in remaining_classes}
    
    print(f"   🎲 Stratifying {len(remaining_classes)} classes with >3 images")
    
    # Get remaining unassigned images
    remaining_images = []
    for filename, labels in image_labels.items():
        if filename not in train_images and filename not in val_images and filename not in test_images:
            remaining_images.append((filename, labels))
    
    # Step 3a: Ensure each class gets at least one image in each split
    for label in remaining_classes:
        print(f"   📍 Ensuring {label} appears in all splits...")
        
        # Find images with this label that are still unassigned
        candidate_images = [(filename, labels) for filename, labels in remaining_images 
                           if label in labels]
        
        if len(candidate_images) >= 3:
            random.shuffle(candidate_images)
            
            # Assign one to each split if not already covered
            assignments = []
            for split_name in ['train', 'val', 'test']:
                if class_split_needs[label][split_name]:
                    for filename, labels in candidate_images:
                        if filename not in train_images and filename not in val_images and filename not in test_images:
                            if split_name == 'train':
                                train_images.add(filename)
                            elif split_name == 'val':
                                val_images.add(filename)
                            else:  # test
                                test_images.add(filename)
                            
                            assignments.append((filename, split_name))
                            class_split_needs[label][split_name] = False
                            
                            # Remove from remaining images
                            remaining_images = [(f, l) for f, l in remaining_images if f != filename]
                            break
            
            print(f"   📍 {label}: {len(assignments)} guaranteed assignments")
    
    # Step 3b: Stratified sampling for remaining images
    print(f"   📊 Stratified sampling for {len(remaining_images)} remaining images")
    
    # Calculate class frequencies in remaining images
    class_frequencies = defaultdict(int)
    for filename, labels in remaining_images:
        for label in labels:
            class_frequencies[label] += 1
    
    # Sort images by rarity of their classes (rarest first) for better stratification
    def image_rarity_score(filename_labels):
        filename, labels = filename_labels
        if not labels:
            return float('inf')
        return min(class_frequencies[label] for label in labels)
    
    remaining_images.sort(key=image_rarity_score)
    
    # Calculate target split sizes for remaining images
    n_remaining = len(remaining_images)
    target_train = int(n_remaining * TRAIN_RATIO)
    target_val = int(n_remaining * VAL_RATIO)
    target_test = n_remaining - target_train - target_val
    
    current_train = len([img for img in remaining_images[:target_train]])
    current_val = len([img for img in remaining_images[target_train:target_train + target_val]])
    current_test = len([img for img in remaining_images[target_train + target_val:]])
    
    # Assign remaining images to maintain ratios
    for i, (filename, labels) in enumerate(remaining_images):
        if i < target_train:
            train_images.add(filename)
        elif i < target_train + target_val:
            val_images.add(filename)
        else:
            test_images.add(filename)
    
    return list(train_images), list(val_images), list(test_images)

def analyze_split_distribution(train_images, val_images, test_images, image_labels):
    """Analyze class distribution across splits."""
    
    def count_class_distribution(image_list, split_name):
        class_counts = defaultdict(int)
        for filename in image_list:
            if filename in image_labels:
                for label in image_labels[filename]:
                    class_counts[label] += 1
        return class_counts
    
    train_dist = count_class_distribution(train_images, "train")
    val_dist = count_class_distribution(val_images, "val")
    test_dist = count_class_distribution(test_images, "test")
    
    all_classes = set(train_dist.keys()) | set(val_dist.keys()) | set(test_dist.keys())
    
    print(f"\n📊 Class distribution across splits:")
    print(f"{'Class':<20} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("-" * 60)
    
    for label in sorted(all_classes):
        train_count = train_dist.get(label, 0)
        val_count = val_dist.get(label, 0)
        test_count = test_dist.get(label, 0)
        total = train_count + val_count + test_count
        
        print(f"{label:<20} {train_count:<8} {val_count:<8} {test_count:<8} {total:<8}")
        
        # Check if class appears in all splits (for classes with >3 images)
        if total > 3:
            missing_splits = []
            if train_count == 0:
                missing_splits.append("train")
            if val_count == 0:
                missing_splits.append("val")
            if test_count == 0:
                missing_splits.append("test")
            
            if missing_splits:
                print(f"   ⚠️  {label} missing from: {', '.join(missing_splits)}")

def create_split_annotations(annotations, split_map):
    """Add split information to annotations."""
    split_annotations = []
    
    for item in annotations:
        item_copy = item.copy()
        filename = os.path.basename(item['data']['image'])
        
        if filename in split_map:
            item_copy['split'] = split_map[filename]
        else:
            item_copy['split'] = 'unknown'
            
        split_annotations.append(item_copy)
    
    return split_annotations

def save_outputs(split_annotations, split_map, train_images, val_images, test_images):
    """Save split annotations and split map."""
    
    # Ensure output directory exists
    Path(OUTPUT_ANNOTATIONS_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # Save annotations with split information
    with open(OUTPUT_ANNOTATIONS_PATH, 'w') as f:
        json.dump(split_annotations, f, indent=2)
    
    print(f"💾 Saved annotations with splits to: {OUTPUT_ANNOTATIONS_PATH}")
    
    # Save split map
    with open(OUTPUT_SPLIT_MAP_PATH, 'w') as f:
        json.dump(split_map, f, indent=2)
    
    print(f"💾 Saved split map to: {OUTPUT_SPLIT_MAP_PATH}")
    
    # Print summary statistics
    print(f"\n📊 Split Summary:")
    print(f"   🚂 Train: {len(train_images)} images ({len(train_images)/len(split_map)*100:.1f}%)")
    print(f"   🔍 Val: {len(val_images)} images ({len(val_images)/len(split_map)*100:.1f}%)")
    print(f"   🧪 Test: {len(test_images)} images ({len(test_images)/len(split_map)*100:.1f}%)")
    print(f"   📊 Total: {len(split_map)} images")

def main():
    """Main function to create data splits."""
    print("📋 Starting data split creation for arthropod detection...")
    print(f"📄 Input: {ANNOTATIONS_PATH}")
    print(f"💾 Output annotations: {OUTPUT_ANNOTATIONS_PATH}")
    print(f"💾 Output split map: {OUTPUT_SPLIT_MAP_PATH}")
    print(f"📊 Split ratios: {TRAIN_RATIO:.0%} train, {VAL_RATIO:.0%} val, {TEST_RATIO:.0%} test")
    print()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Check prerequisites
    if not os.path.exists(ANNOTATIONS_PATH):
        print(f"❌ Missing annotations file: {ANNOTATIONS_PATH}")
        print("Please run 00_prep_env.py first to download the data.")
        return 1
    
    if not os.path.exists(IMAGES_DIR):
        print(f"❌ Missing images directory: {IMAGES_DIR}")
        print("Please run 00_prep_env.py first to download the data.")
        return 1
    
    try:
        # Load annotations
        annotations = load_annotations()
        
        if not annotations:
            print("❌ No valid annotations found to process")
            return 1
        
        # Extract image labels
        image_labels = extract_image_labels(annotations)
        
        if not image_labels:
            print("❌ No images with labels found")
            return 1
        
        # Analyze class distribution
        class_image_counts = analyze_class_distribution(image_labels)
        
        # Create stratified splits
        train_images, val_images, test_images = stratified_split(image_labels, class_image_counts)
        
        # Analyze final split distribution
        analyze_split_distribution(train_images, val_images, test_images, image_labels)
        
        # Create split map
        split_map = {}
        for img in train_images:
            split_map[img] = 'train'
        for img in val_images:
            split_map[img] = 'val'
        for img in test_images:
            split_map[img] = 'test'
        
        # Create annotations with split information
        split_annotations = create_split_annotations(annotations, split_map)
        
        # Save outputs
        save_outputs(split_annotations, split_map, train_images, val_images, test_images)
        
        print("\n🎉 Data split creation complete!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during data split creation: {e}")
        return 1

if __name__ == "__main__":
    exit(main())