# PANDA Challenge: Prostate Cancer Grade Assessment
**Deep Learning Final Project - Group 12 [Asahi]**

An ensemble deep learning approach for automated prostate cancer grading achieving top-3 performance in the PANDA Challenge with QWK score of 0.94012.

## Project Overview

This project addresses the critical challenge of automated prostate cancer grading using deep learning on whole-slide histopathological images. Our solution combines multiple EfficientNet architectures in an ensemble approach to predict ISUP grades (0-5) from gigapixel medical images.

## Architecture

### Ensemble Composition
Our winning solution combines three complementary architectures:

1. **EfficientNet-B1** (multiple folds: 1, 4, 5)
2. **EfficientNetV2-S** (fold 2 + top 3 folds)  
3. **EfficientNetV2-M** (top 3 folds)

### Key Technical Innovations
- **Tile-based Processing**: 6×6 grid (36 tiles) of 256×256 pixels each
- **Strategic Layer Freezing**: First 3 blocks frozen for optimal transfer learning
- **Ordinal Regression**: Custom classifier design respecting ISUP grade hierarchy
- **Mixed Precision Training**: ~1.5x speedup with maintained accuracy

## Dataset

The PANDA dataset contains **10,616 whole-slide images** from two medical centers:
- **Radboud University Medical Center** (Netherlands)
- **Karolinska Institute** (Sweden)

**Grade Distribution:**
- Grade 0: 27.24% (benign)
- Grade 1: 25.11% 
- Grade 2: 12.65%
- Grade 3: 11.70%
- Grade 4: 11.77%
- Grade 5: 11.53% (most aggressive)
Courtesy of [rohitsingh9990's PANDA EDA](https://www.kaggle.com/code/rohitsingh9990/panda-eda-better-visualization-simple-baseline)

## Quick Start

### Prerequisites

```bash
# Create conda environment
conda env create -f environment.yml
conda activate PANDA
```

### Installation

```bash
git clone https://github.com/yourusername/PANDA-Challenge.git
cd PANDA-Challenge
pip install -r requirements.txt
```

### Training

```bash
# Single model training
python Train.py

# Configure training parameters in Train.py:
# - FOLD: Cross-validation fold (1-5)
# - modelname: 'efficientnet-b1', 'efficientnet-m', 'efficientnet-s'
# - n_epochs: Training epochs (default: 20)
# - batch_size: Batch size (default: 2)
```

### Inference

```bash
# Run inference with ensemble
jupyter notebook panda-effnet-inference.ipynb
```

## Project Structure

```
PANDA-Challenge/
├── environment.yml              # Conda environment configuration
├── Train.py                    # Main training script
├── panda-effnet-inference.ipynb # Inference notebook with ensemble
├── README.md                   # This file
├── models/                     # Saved model checkpoints
├── log/                       # Training logs and metrics
├── data_csv/                  # Dataset CSV files
├── train_256_36/             # Preprocessed training tiles
└── val/                      # Validation data
```

## Configuration

### Key Training Parameters

```python
# Model Configuration
tile_size = 256           # Individual tile size
image_size = 256         # Processing resolution  
n_tiles = 36            # Total tiles per image (6x6 grid)
batch_size = 2          # Batch size for training
n_epochs = 20           # Maximum training epochs
init_lr = 1e-4          # Initial learning rate

# Architecture Selection
modelname = 'efficientnet-m'  # Options: 'efficientnet-b1', 'efficientnet-m', 'efficientnet-s'
freeze_blocks = 3            # Number of blocks to freeze for transfer learning
```

### Data Augmentation

```python
transforms_train = A.Compose([
    A.ShiftScaleRotate(scale_limit=0.25, rotate_limit=180, p=0.5),
    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.CoarseDropout(num_holes_range=(1, 10), hole_height_range=(10, 128), hole_width_range=(10, 128), p=0.5),
    A.Transpose(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5)
])
```

## Results

### Performance Comparison

| Rank | Team/Method | Architecture | QWK Score |
|------|-------------|--------------|-----------|
| 1 | Team PND | EfficientNet-B0 + B1 Ensemble | 0.94085 |
| **2** | **Our Approach** | **EfficientNet-B1 + EfficientNetV2-S + EfficientNetV2-M** | **0.94012** |
| 3 | Team Save The Prostate | ResNet34 + SE block | 0.93768 |

### Ablation Study Results

| Configuration | Architecture Details | QWK Score |
|---------------|---------------------|-----------|
| Single Model | EfficientNet-B1 (fold 1) | 0.93361 |
| Multi-fold | EfficientNet-B1 (folds 1,4,5) | 0.93438 |
| EfficientNetV2 | EfficientNetV2-M (top 3 folds) | 0.93262 |
| **Full Ensemble** | **All architectures combined** | **0.94012** |

## Technical Details

### Tile-based Processing Pipeline

1. **Image Preprocessing**: Gigapixel WSIs → 6×6 tile grid
2. **Information Ranking**: Select 36 most informative tiles
3. **Tile Arrangement**: Reconstruct as 1536×1536 composite image
4. **Feature Extraction**: Process through EfficientNet architectures
5. **Ensemble Prediction**: Combine multiple model outputs

### Loss Function & Evaluation

```python
# Ordinal regression with cumulative encoding
criterion = nn.BCEWithLogitsLoss()

# Evaluation metric
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')
```

## Requirements

### Core Dependencies
- Python 3.10+
- PyTorch 2.7.0
- torchvision 0.22.0
- efficientnet-pytorch 0.7.1
- albumentations 2.0.6
- scikit-learn 1.6.1
- pandas 2.2.3
- numpy 2.2.5
- opencv-python 4.11.0.86

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070/4060 Ti or better)
- **RAM**: 16GB+ system memory

## Usage Examples

### Training a Single Model

```python
# Configure in Train.py
FOLD = 1
modelname = 'efficientnet-m'
n_epochs = 20
batch_size = 2

# Run training
python Train.py
```

### Loading Pretrained Models

```python
import torch
from Train import enetv2

# Load trained model
model = enetv2(out_dim=5, types='m')
checkpoint = torch.load('models/efficientnet-m-1/efficientnet-m-1_best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Ensemble Inference

```python
# See panda-effnet-inference.ipynb for complete ensemble pipeline
# Key steps:
# 1. Load multiple trained models
# 2. Apply test-time augmentation
# 3. Combine predictions with weighted averaging
# 4. Apply optimized thresholding for final grades
```

## Acknowledgments

- **PANDA Challenge Organizers** for providing the comprehensive dataset
- **Radboud University Medical Center** and **Karolinska Institute** for data contribution
- **EfficientNet authors** for the foundational architecture
- **Team Members**: 
  - 褚敏匡 (313553023)
  - 林家輝 (313561001) 
  - 梁巍濤 (313561002)
