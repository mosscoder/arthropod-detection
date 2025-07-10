#!/usr/bin/env python3
"""
00_prep_env.py - Data Preparation Script

Downloads arthropod annotation data and images from Google Drive to local directories.
This script sets up the local environment for arthropod detection experiments.

Directory Structure Created:
- data/tabular/annotations.json
- data/raster/petri_dish_src/[images]
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import io
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from googleapiclient.http import MediaIoBaseDownload
import time

# Configuration
ANNOTATION_FILE_ID = "157d7IfklIyNipZ2Q5bz2ZOAOwEx6LLMZ"  # JSON annotation file
IMAGE_FOLDER_ID = "12PXW15hlPx_FBGXy5imzcwGrwc9m3S_R"     # Image folder
ANNOTATIONS_PATH = "data/tabular/annotations.json"
IMAGES_DIR = "data/raster/petri_dish_src"
RESULTS_DIR = "results"

def setup_drive_service():
    """Set up Google Drive service with service account credentials"""
    load_dotenv()
    service_account_file = os.getenv('GOOGLE_SERVICE_ACCOUNT_FILE')
    if not service_account_file:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_FILE environment variable not set")
    
    scopes = ['https://www.googleapis.com/auth/drive']
    credentials = Credentials.from_service_account_file(service_account_file, scopes=scopes)
    return build('drive', 'v3', credentials=credentials)

def create_directories():
    """Create necessary directory structure"""
    dirs_to_create = [
        Path(ANNOTATIONS_PATH).parent,
        Path(IMAGES_DIR),
        Path(RESULTS_DIR)
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

def download_annotation_file(service, file_id: str, local_path: str) -> bool:
    """Download annotation JSON file from Google Drive"""
    try:
        print(f"📄 Downloading annotation file {file_id}...")
        
        # Get file metadata
        file_metadata = service.files().get(fileId=file_id).execute()
        filename = file_metadata['name']
        print(f"   📄 File: {filename}")
        
        # Download file
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"   📥 Download progress: {int(status.progress() * 100)}%")
        
        # Save to local file
        with open(local_path, 'wb') as f:
            f.write(fh.getvalue())
        
        print(f"✅ Successfully downloaded annotation file to: {local_path}")
        
        # Validate JSON
        with open(local_path, 'r') as f:
            data = json.load(f)
            print(f"   📊 Loaded {len(data)} annotation records")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading annotation file: {e}")
        return False

def get_folder_files(service, folder_id: str) -> List[Dict]:
    """Get list of files in a Google Drive folder"""
    try:
        print(f"📁 Scanning folder {folder_id}...")
        
        results = service.files().list(
            q=f"'{folder_id}' in parents",
            fields="files(id, name, mimeType, size)",
            pageSize=1000
        ).execute()
        
        files = results.get('files', [])
        image_files = [f for f in files if f['mimeType'].startswith('image/')]
        
        print(f"   📄 Found {len(image_files)} image files")
        return image_files
        
    except Exception as e:
        print(f"❌ Error scanning folder: {e}")
        return []

def download_file(service, file_id: str, filename: str, local_path: str) -> bool:
    """Download a single file from Google Drive with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Skip if file already exists
            if os.path.exists(local_path):
                print(f"   ⏭️  Skipping {filename} (already exists)")
                return True
            
            # Download file
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            # Save to local file
            with open(local_path, 'wb') as f:
                f.write(fh.getvalue())
            
            return True
            
        except Exception as e:
            print(f"   ⚠️  Attempt {attempt + 1} failed for {filename}: {e}")
            if attempt < max_retries - 1:
                print(f"   🔄 Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"   ❌ Failed to download {filename} after {max_retries} attempts")
                return False

def download_images(service, folder_id: str, local_dir: str) -> int:
    """Download all images from Google Drive folder"""
    files = get_folder_files(service, folder_id)
    if not files:
        print("❌ No files found in folder")
        return 0
    
    print(f"📥 Starting download of {len(files)} images...")
    downloaded_count = 0
    
    for i, file_info in enumerate(files, 1):
        file_id = file_info['id']
        filename = file_info['name']
        size = file_info.get('size', 'Unknown')
        
        if size != 'Unknown':
            size_mb = f"{int(size) / (1024*1024):.1f}MB"
        else:
            size_mb = "Unknown size"
        
        print(f"   📄 [{i}/{len(files)}] {filename} ({size_mb})")
        
        local_path = os.path.join(local_dir, filename)
        if download_file(service, file_id, filename, local_path):
            downloaded_count += 1
        
        # Small delay to be respectful to the API
        time.sleep(0.1)
    
    print(f"✅ Downloaded {downloaded_count}/{len(files)} images")
    return downloaded_count

def main():
    """Main function to prepare the environment"""
    print("🔧 Starting data preparation...")
    print(f"📄 Annotation file ID: {ANNOTATION_FILE_ID}")
    print(f"📁 Image folder ID: {IMAGE_FOLDER_ID}")
    print()
    
    # Create directories
    create_directories()
    print()
    
    # Setup Google Drive service
    try:
        service = setup_drive_service()
        print("✅ Google Drive service initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Google Drive service: {e}")
        return 1
    
    print()
    
    # Download annotation file
    annotation_success = download_annotation_file(service, ANNOTATION_FILE_ID, ANNOTATIONS_PATH)
    if not annotation_success:
        print("❌ Failed to download annotation file")
        return 1
    
    print()
    
    # Download images
    image_count = download_images(service, IMAGE_FOLDER_ID, IMAGES_DIR)
    if image_count == 0:
        print("❌ No images downloaded")
        return 1
    
    print()
    print("🎉 Data preparation complete!")
    print(f"   📄 Annotations: {ANNOTATIONS_PATH}")
    print(f"   📁 Images: {IMAGES_DIR} ({image_count} files)")
    print(f"   📊 Results directory: {RESULTS_DIR}")
    print()
    print("Ready to run 01_plot_batch.py!")
    
    return 0

if __name__ == "__main__":
    exit(main())