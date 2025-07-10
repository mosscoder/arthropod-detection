# Arthropod Detection Pipeline - Environment Setup

## Environment Creation with Mamba/Conda

This project provides a complete environment specification for reproducing the arthropod detection pipeline on any system.

### Quick Setup

1. **Create the environment from the YAML file:**
   ```bash
   mamba env create -f environment.yml
   ```
   
   Or if using conda:
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**
   ```bash
   mamba activate arthropod_detector
   # or
   conda activate arthropod_detector
   ```

3. **Verify installation:**
   ```bash
   python -c "import ultralytics, numpy, PIL, matplotlib, yaml; print('All dependencies loaded successfully!')"
   ```

### Alternative Manual Installation

If you prefer to install packages manually:

```bash
# Create new environment
mamba create -n arthropod_detector python=3.10

# Activate environment
mamba activate arthropod_detector

# Install core dependencies via mamba/conda
mamba install -c conda-forge numpy pillow matplotlib gdal pytorch pyyaml python-dotenv scikit-learn jupyter

# Install pip-only dependencies
pip install ultralytics google-api-python-client google-auth google-auth-oauthlib opencv-python
```

### Key Dependencies Explained

- **numpy, matplotlib, pillow**: Core scientific computing and image processing
- **gdal**: Efficient reading of large geospatial images (crucial for petri dish images)
- **pytorch**: Data loading utilities and tensor operations
- **ultralytics**: YOLOv8 implementation for object detection
- **pyyaml**: YAML configuration file handling
- **google-api-python-client**: Google Drive API access for data download
- **python-dotenv**: Environment variable management
- **scikit-learn**: Data splitting and scientific utilities
- **opencv-python**: Additional image processing capabilities

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: 8GB+ RAM recommended for processing large images
- **Storage**: 10GB+ free space for datasets and models
- **GPU**: Optional but recommended for faster training (CUDA-compatible)

### Google Drive API Setup (Optional)

If you plan to use the data download script (`00_prep_env.py`):

1. Create a Google Cloud project and enable Drive API
2. Create a service account and download the JSON credentials
3. Set environment variable: `GOOGLE_SERVICE_ACCOUNT_FILE=/path/to/credentials.json`

### Troubleshooting

**GDAL Installation Issues:**
```bash
# If GDAL fails to install via conda-forge
mamba install -c conda-forge gdal=3.6.*
```

**PyTorch GPU Support:**
```bash
# For CUDA support (check PyTorch website for latest commands)
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Ultralytics Installation Issues:**
```bash
# If ultralytics has dependency conflicts
pip install --upgrade ultralytics
```

### Environment Verification

Run this script to verify all components are working:

```python
#!/usr/bin/env python3
import sys
import importlib

dependencies = [
    'numpy', 'PIL', 'matplotlib', 'yaml', 'torch', 
    'ultralytics', 'osgeo.gdal', 'google.auth',
    'sklearn', 'cv2'
]

print("Checking dependencies...")
for dep in dependencies:
    try:
        importlib.import_module(dep)
        print(f"✅ {dep}")
    except ImportError:
        print(f"❌ {dep}")

print("\\nEnvironment check complete!")
```

### Usage

Once the environment is set up, you can run the complete pipeline:

```bash
# Download data (requires Google credentials)
python 00_prep_env.py

# Create data splits
python 02_establish_data_splits.py

# Extract crops (optional)
python 03_process_crops.py

# Prepare YOLO dataset
python 04_yolo_prep.py

# Train YOLOv8 model
python 05_yolo_train.py
```