#!/usr/bin/env python3
"""
06_inference.py - SAHI Inference for Arthropod Detection

Performs SAHI (Slicing Aided Hyper Inference) on test images with comprehensive evaluation:
- Loads trained YOLOv8 model and processes test images with sliced inference
- Generates side-by-side comparisons: ground truth vs predictions
- Calculates detailed metrics: precision, recall, F1-score per image and overall
- Saves results in organized output structure with JSON exports

Key Features:
- SAHI sliced inference optimized for large petri dish images
- IoU-based matching between ground truth and predictions
- Comprehensive metrics calculation and visualization
- Side-by-side comparison images for visual evaluation
- JSON export of all detection results and metrics

Prerequisites:
- Run 04_yolo_prep.py to prepare test images (resized, not tiled)
- Run 05_yolo_train.py to train the model
- Requires models/yolo_arthropod_best.pt and data/yolo_dataset/test/

Output:
- results/sahi_inference/comparisons/ - Side-by-side truth vs predictions
- results/sahi_inference/predictions/ - JSON detection results per image
- results/sahi_inference/metrics/ - Performance metrics (image-level and overall)
- results/sahi_inference/summary_report.html - Human-readable summary
"""

import os
import json
import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

# SAHI and YOLO imports
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
from ultralytics import YOLO
import pandas as pd

# Configuration
MODEL_PATH = "models/yolo_arthropod_best.pt"
TEST_IMAGES_DIR = "data/yolo_dataset/test/images"
TEST_LABELS_DIR = "data/yolo_dataset/test/labels"
OUTPUT_DIR = "results/sahi_inference"
DATASET_INFO_PATH = "data/yolo_dataset/dataset_info.json"

# SAHI Configuration (Optimized for Recall)
SAHI_CONFIG = {
    'slice_height': 1280,
    'slice_width': 1280,
    'overlap_height_ratio': 0.3,  # Increased from 0.2 for better boundary coverage
    'overlap_width_ratio': 0.3,   # Increased from 0.2 for better boundary coverage
    'postprocess_type': 'NMS',
    'postprocess_match_metric': 'IOU',
    'postprocess_match_threshold': 0.5,
    'postprocess_class_agnostic': True
}

# Detection thresholds (Optimized for Recall)
CONF_THRESHOLD = 0.15  # Lowered from 0.25 to capture more potential detections
IOU_THRESHOLD = 0.35   # Lowered from 0.45 for more lenient NMS
IOU_MATCH_THRESHOLD = 0.3  # Lowered from 0.5 for more lenient evaluation matching

def setup_output_directories():
    """Create output directories for inference results."""
    dirs = [
        f"{OUTPUT_DIR}/comparisons",
        f"{OUTPUT_DIR}/predictions", 
        f"{OUTPUT_DIR}/metrics"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_dataset_info():
    """Load dataset information from preparation step."""
    if os.path.exists(DATASET_INFO_PATH):
        with open(DATASET_INFO_PATH, 'r') as f:
            return json.load(f)
    return {}

def setup_sahi_model():
    """Initialize SAHI detection model."""
    print("🤖 Setting up SAHI model...")
    
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov11',
        model_path=MODEL_PATH,
        confidence_threshold=CONF_THRESHOLD,
        device='cuda'  # Use GPU if available
    )
    
    print(f"✅ SAHI model loaded: {MODEL_PATH}")
    return detection_model

def parse_yolo_label(label_path, image_width, image_height):
    """Parse YOLO format label file to extract ground truth bounding boxes."""
    ground_truth = []
    
    if not os.path.exists(label_path):
        return ground_truth
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                continue
            
            class_id, center_x, center_y, width, height = map(float, parts)
            
            # Convert normalized coordinates to pixel coordinates
            center_x_px = center_x * image_width
            center_y_px = center_y * image_height
            width_px = width * image_width
            height_px = height * image_height
            
            # Convert to x_min, y_min, x_max, y_max
            x_min = center_x_px - width_px / 2
            y_min = center_y_px - height_px / 2
            x_max = center_x_px + width_px / 2
            y_max = center_y_px + height_px / 2
            
            ground_truth.append({
                'class_id': int(class_id),
                'bbox': [x_min, y_min, x_max, y_max],
                'confidence': 1.0  # Ground truth has confidence 1.0
            })
    
    return ground_truth

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    # box format: [x_min, y_min, x_max, y_max]
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    intersect_x_min = max(x1_min, x2_min)
    intersect_y_min = max(y1_min, y2_min)
    intersect_x_max = min(x1_max, x2_max)
    intersect_y_max = min(y1_max, y2_max)
    
    if intersect_x_min >= intersect_x_max or intersect_y_min >= intersect_y_max:
        return 0.0
    
    intersect_area = (intersect_x_max - intersect_x_min) * (intersect_y_max - intersect_y_min)
    
    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - intersect_area
    
    return intersect_area / union_area if union_area > 0 else 0.0

def match_predictions_to_ground_truth(predictions, ground_truth, iou_threshold=0.5):
    """Match predictions to ground truth using IoU threshold."""
    matches = []
    used_gt = set()
    used_pred = set()
    
    # Sort predictions by confidence (highest first)
    sorted_predictions = sorted(enumerate(predictions), key=lambda x: x[1]['confidence'], reverse=True)
    
    for pred_idx, prediction in sorted_predictions:
        if pred_idx in used_pred:
            continue
        
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in used_gt:
                continue
            
            iou = calculate_iou(prediction['bbox'], gt['bbox'])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_gt_idx != -1:
            matches.append({
                'pred_idx': pred_idx,
                'gt_idx': best_gt_idx,
                'iou': best_iou,
                'confidence': prediction['confidence']
            })
            used_pred.add(pred_idx)
            used_gt.add(best_gt_idx)
    
    return matches, used_pred, used_gt

def calculate_image_metrics(predictions, ground_truth):
    """Calculate precision, recall, and F1-score for a single image."""
    if len(ground_truth) == 0 and len(predictions) == 0:
        return {
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
    
    if len(ground_truth) == 0:
        return {
            'precision': 0.0,
            'recall': 1.0,  # No ground truth to miss
            'f1_score': 0.0,
            'true_positives': 0,
            'false_positives': len(predictions),
            'false_negatives': 0
        }
    
    if len(predictions) == 0:
        return {
            'precision': 1.0,  # No false positives
            'recall': 0.0,
            'f1_score': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': len(ground_truth)
        }
    
    matches, used_pred, used_gt = match_predictions_to_ground_truth(
        predictions, ground_truth, IOU_MATCH_THRESHOLD
    )
    
    true_positives = len(matches)
    false_positives = len(predictions) - len(used_pred)
    false_negatives = len(ground_truth) - len(used_gt)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def run_sahi_inference(image_path, detection_model):
    """Run SAHI inference on a single image."""
    print(f"   🔍 Running SAHI inference on: {os.path.basename(image_path)}")
    
    # Run sliced prediction
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=SAHI_CONFIG['slice_height'],
        slice_width=SAHI_CONFIG['slice_width'],
        overlap_height_ratio=SAHI_CONFIG['overlap_height_ratio'],
        overlap_width_ratio=SAHI_CONFIG['overlap_width_ratio'],
        postprocess_type=SAHI_CONFIG['postprocess_type'],
        postprocess_match_metric=SAHI_CONFIG['postprocess_match_metric'],
        postprocess_match_threshold=SAHI_CONFIG['postprocess_match_threshold'],
        postprocess_class_agnostic=SAHI_CONFIG['postprocess_class_agnostic']
    )
    
    # Extract predictions
    predictions = []
    for detection in result.object_prediction_list:
        bbox = detection.bbox
        predictions.append({
            'class_id': detection.category.id,
            'class_name': detection.category.name,
            'confidence': detection.score.value,
            'bbox': [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy]
        })
    
    return predictions

def create_comparison_image(image_path, ground_truth, predictions, output_path):
    """Create side-by-side comparison image: ground truth vs predictions."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"   ⚠️  Failed to load image: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    
    # Create figure with side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Ground truth visualization (left)
    ax1.imshow(image_rgb)
    ax1.set_title(f'Ground Truth ({len(ground_truth)} annotations)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    for gt in ground_truth:
        x_min, y_min, x_max, y_max = gt['bbox']
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                        linewidth=2, edgecolor='green', facecolor='none')
        ax1.add_patch(rect)
        ax1.text(x_min, y_min - 5, 'GT', color='green', fontsize=12, fontweight='bold')
    
    # Predictions visualization (right)
    ax2.imshow(image_rgb)
    ax2.set_title(f'Predictions ({len(predictions)} detections)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    for pred in predictions:
        x_min, y_min, x_max, y_max = pred['bbox']
        conf = pred['confidence']
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                        linewidth=2, edgecolor='red', facecolor='none')
        ax2.add_patch(rect)
        ax2.text(x_min, y_min - 5, f'{conf:.2f}', color='red', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Comparison saved: {os.path.basename(output_path)}")

def process_all_test_images(detection_model):
    """Process all test images with SAHI inference."""
    print("🔄 Processing all test images...")
    
    # Get list of test images
    test_images = []
    if os.path.exists(TEST_IMAGES_DIR):
        for filename in os.listdir(TEST_IMAGES_DIR):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(filename)
    
    if not test_images:
        print("❌ No test images found")
        return {}, {}
    
    print(f"📊 Found {len(test_images)} test images")
    
    all_predictions = {}
    all_metrics = {}
    
    for idx, image_filename in enumerate(test_images):
        print(f"📷 Processing image {idx + 1}/{len(test_images)}: {image_filename}")
        
        # Paths
        image_path = os.path.join(TEST_IMAGES_DIR, image_filename)
        label_path = os.path.join(TEST_LABELS_DIR, 
                                 os.path.splitext(image_filename)[0] + '.txt')
        
        # Load image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            print(f"   ⚠️  Failed to load image: {image_path}")
            continue
        
        height, width = image.shape[:2]
        
        # Parse ground truth
        ground_truth = parse_yolo_label(label_path, width, height)
        
        # Run SAHI inference
        predictions = run_sahi_inference(image_path, detection_model)
        
        # Calculate metrics
        metrics = calculate_image_metrics(predictions, ground_truth)
        metrics['ground_truth_count'] = len(ground_truth)
        metrics['prediction_count'] = len(predictions)
        if predictions:
            metrics['average_confidence'] = np.mean([p['confidence'] for p in predictions])
        else:
            metrics['average_confidence'] = 0.0
        
        # Store results
        all_predictions[image_filename] = {
            'ground_truth': ground_truth,
            'predictions': predictions,
            'image_dimensions': [width, height]
        }
        all_metrics[image_filename] = metrics
        
        # Create comparison image
        base_name = os.path.splitext(image_filename)[0]
        comparison_path = os.path.join(OUTPUT_DIR, 'comparisons', f"{base_name}_comparison.jpg")
        create_comparison_image(image_path, ground_truth, predictions, comparison_path)
        
        # Save individual predictions
        pred_path = os.path.join(OUTPUT_DIR, 'predictions', f"{base_name}_predictions.json")
        with open(pred_path, 'w') as f:
            json.dump(all_predictions[image_filename], f, indent=2)
        
        print(f"   📊 Metrics - P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}")
    
    return all_predictions, all_metrics

def calculate_overall_metrics(all_metrics):
    """Calculate overall dataset metrics."""
    print("📈 Calculating overall metrics...")
    
    total_tp = sum(m['true_positives'] for m in all_metrics.values())
    total_fp = sum(m['false_positives'] for m in all_metrics.values())
    total_fn = sum(m['false_negatives'] for m in all_metrics.values())
    total_gt = sum(m['ground_truth_count'] for m in all_metrics.values())
    total_pred = sum(m['prediction_count'] for m in all_metrics.values())
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    # Calculate mean metrics across images
    mean_precision = np.mean([m['precision'] for m in all_metrics.values()])
    mean_recall = np.mean([m['recall'] for m in all_metrics.values()])
    mean_f1 = np.mean([m['f1_score'] for m in all_metrics.values()])
    
    # Confidence distribution
    all_confidences = []
    for filename, metrics in all_metrics.items():
        if metrics['prediction_count'] > 0:
            all_confidences.append(metrics['average_confidence'])
    
    confidence_stats = {
        'mean': np.mean(all_confidences) if all_confidences else 0.0,
        'std': np.std(all_confidences) if all_confidences else 0.0,
        'min': np.min(all_confidences) if all_confidences else 0.0,
        'max': np.max(all_confidences) if all_confidences else 0.0
    }
    
    overall_metrics = {
        'dataset_summary': {
            'total_images': len(all_metrics),
            'total_ground_truth': total_gt,
            'total_predictions': total_pred,
            'total_true_positives': total_tp,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1_score': overall_f1,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_f1_score': mean_f1,
            'confidence_distribution': confidence_stats
        },
        'sahi_config': SAHI_CONFIG,
        'detection_thresholds': {
            'confidence_threshold': CONF_THRESHOLD,
            'iou_threshold': IOU_THRESHOLD,
            'iou_match_threshold': IOU_MATCH_THRESHOLD
        }
    }
    
    return overall_metrics

def save_metrics(all_metrics, overall_metrics):
    """Save metrics to JSON files."""
    print("💾 Saving metrics...")
    
    # Save image-level metrics
    image_metrics_path = os.path.join(OUTPUT_DIR, 'metrics', 'image_metrics.json')
    with open(image_metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Save overall metrics
    overall_metrics_path = os.path.join(OUTPUT_DIR, 'metrics', 'overall_metrics.json')
    with open(overall_metrics_path, 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    
    print(f"📊 Image metrics saved: {image_metrics_path}")
    print(f"📊 Overall metrics saved: {overall_metrics_path}")

def generate_summary_report(all_metrics, overall_metrics):
    """Generate HTML summary report."""
    print("📄 Generating summary report...")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SAHI Inference Summary Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .metrics {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .config {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .good {{ color: #008000; }}
            .medium {{ color: #ff8000; }}
            .poor {{ color: #ff0000; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 SAHI Inference Summary Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="metrics">
            <h2>📊 Overall Performance</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Overall Precision</td><td>{overall_metrics['dataset_summary']['overall_precision']:.3f}</td></tr>
                <tr><td>Overall Recall</td><td>{overall_metrics['dataset_summary']['overall_recall']:.3f}</td></tr>
                <tr><td>Overall F1-Score</td><td>{overall_metrics['dataset_summary']['overall_f1_score']:.3f}</td></tr>
                <tr><td>Mean Precision</td><td>{overall_metrics['dataset_summary']['mean_precision']:.3f}</td></tr>
                <tr><td>Mean Recall</td><td>{overall_metrics['dataset_summary']['mean_recall']:.3f}</td></tr>
                <tr><td>Mean F1-Score</td><td>{overall_metrics['dataset_summary']['mean_f1_score']:.3f}</td></tr>
            </table>
        </div>
        
        <div class="metrics">
            <h2>🔢 Detection Statistics</h2>
            <table>
                <tr><th>Statistic</th><th>Value</th></tr>
                <tr><td>Total Images</td><td>{overall_metrics['dataset_summary']['total_images']}</td></tr>
                <tr><td>Total Ground Truth</td><td>{overall_metrics['dataset_summary']['total_ground_truth']}</td></tr>
                <tr><td>Total Predictions</td><td>{overall_metrics['dataset_summary']['total_predictions']}</td></tr>
                <tr><td>True Positives</td><td>{overall_metrics['dataset_summary']['total_true_positives']}</td></tr>
                <tr><td>False Positives</td><td>{overall_metrics['dataset_summary']['total_false_positives']}</td></tr>
                <tr><td>False Negatives</td><td>{overall_metrics['dataset_summary']['total_false_negatives']}</td></tr>
            </table>
        </div>
        
        <div class="config">
            <h2>⚙️ Configuration</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Slice Size</td><td>{overall_metrics['sahi_config']['slice_height']}x{overall_metrics['sahi_config']['slice_width']}</td></tr>
                <tr><td>Overlap Ratio</td><td>{overall_metrics['sahi_config']['overlap_height_ratio']:.1%}</td></tr>
                <tr><td>Confidence Threshold</td><td>{overall_metrics['detection_thresholds']['confidence_threshold']}</td></tr>
                <tr><td>IoU Threshold</td><td>{overall_metrics['detection_thresholds']['iou_threshold']}</td></tr>
                <tr><td>IoU Match Threshold</td><td>{overall_metrics['detection_thresholds']['iou_match_threshold']}</td></tr>
            </table>
        </div>
        
        <div class="metrics">
            <h2>📈 Per-Image Results</h2>
            <table>
                <tr><th>Image</th><th>GT</th><th>Pred</th><th>Precision</th><th>Recall</th><th>F1</th><th>Conf</th></tr>
    """
    
    for image_name, metrics in all_metrics.items():
        f1_class = 'good' if metrics['f1_score'] > 0.7 else 'medium' if metrics['f1_score'] > 0.4 else 'poor'
        html_content += f"""
                <tr>
                    <td>{image_name}</td>
                    <td>{metrics['ground_truth_count']}</td>
                    <td>{metrics['prediction_count']}</td>
                    <td>{metrics['precision']:.3f}</td>
                    <td>{metrics['recall']:.3f}</td>
                    <td class="{f1_class}">{metrics['f1_score']:.3f}</td>
                    <td>{metrics['average_confidence']:.3f}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    report_path = os.path.join(OUTPUT_DIR, 'summary_report.html')
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"📄 Summary report saved: {report_path}")

def main():
    """Main function to run SAHI inference on test images."""
    print("🚀 Starting SAHI inference for arthropod detection...")
    print(f"🤖 Model: {MODEL_PATH}")
    print(f"📁 Test images: {TEST_IMAGES_DIR}")
    print(f"💾 Output: {OUTPUT_DIR}")
    print()
    
    # Check prerequisites
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("Please run 05_yolo_train.py first to train the model.")
        return 1
    
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"❌ Test images directory not found: {TEST_IMAGES_DIR}")
        print("Please run 04_yolo_prep.py first to prepare test images.")
        return 1
    
    if not os.path.exists(TEST_LABELS_DIR):
        print(f"❌ Test labels directory not found: {TEST_LABELS_DIR}")
        print("Please run 04_yolo_prep.py first to prepare test labels.")
        return 1
    
    # Setup output directories
    setup_output_directories()
    
    # Load dataset info
    dataset_info = load_dataset_info()
    
    # Setup SAHI model
    try:
        detection_model = setup_sahi_model()
    except Exception as e:
        print(f"❌ Error setting up SAHI model: {e}")
        return 1
    
    print()
    print("🔧 SAHI Configuration (Optimized for Recall):")
    print(f"   📐 Slice size: {SAHI_CONFIG['slice_height']}x{SAHI_CONFIG['slice_width']}")
    print(f"   🔄 Overlap ratio: {SAHI_CONFIG['overlap_height_ratio']:.1%} (increased for better boundary coverage)")
    print(f"   🎯 Confidence threshold: {CONF_THRESHOLD} (lowered to capture more detections)")
    print(f"   🎯 IoU threshold: {IOU_THRESHOLD} (lowered for more lenient NMS)")
    print(f"   🎯 IoU match threshold: {IOU_MATCH_THRESHOLD} (lowered for more lenient evaluation)")
    print()
    
    # Process all test images
    try:
        all_predictions, all_metrics = process_all_test_images(detection_model)
        
        if not all_metrics:
            print("❌ No test images were processed successfully")
            return 1
        
        # Calculate overall metrics
        overall_metrics = calculate_overall_metrics(all_metrics)
        
        # Save metrics
        save_metrics(all_metrics, overall_metrics)
        
        # Generate summary report
        import pandas as pd  # For timestamp in HTML report
        generate_summary_report(all_metrics, overall_metrics)
        
        print()
        print("🎉 SAHI inference complete!")
        print(f"   📊 Processed {len(all_metrics)} images")
        print(f"   📈 Overall Precision: {overall_metrics['dataset_summary']['overall_precision']:.3f}")
        print(f"   📈 Overall Recall: {overall_metrics['dataset_summary']['overall_recall']:.3f}")
        print(f"   📈 Overall F1-Score: {overall_metrics['dataset_summary']['overall_f1_score']:.3f}")
        print(f"   📁 Results saved in: {OUTPUT_DIR}")
        print()
        print("📋 Output files:")
        print(f"   🖼️  Side-by-side comparisons: {OUTPUT_DIR}/comparisons/")
        print(f"   🔍 Detection results: {OUTPUT_DIR}/predictions/")
        print(f"   📊 Metrics: {OUTPUT_DIR}/metrics/")
        print(f"   📄 Summary report: {OUTPUT_DIR}/summary_report.html")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())