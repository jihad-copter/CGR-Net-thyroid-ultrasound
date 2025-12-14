# CGR-Net: CGAN-Enhanced Deep Learning for Thyroid Classification

A comprehensive framework combining Conditional Generative Adversarial Networks (CGAN) for data augmentation with state-of-the-art deep learning architectures for medical image classification.

## 🔬 What is CGR-Net?

**CGR-Net** (Conditional GAN-Residual Network) is a two-stage deep learning pipeline designed for thyroid ultrasound classification:

1. **Stage 1 (CGAN):** Generates high-quality synthetic medical images to augment limited training data
2. **Stage 2 (Classification):** Trains multiple architectures (ResNet50, Swin Transformer, EfficientNet, ViT) on the augmented dataset

The framework systematically evaluates the impact of synthetic data augmentation on model performance across multiple architectures and preprocessing strategies.

## 📁 Dataset Structure

Organize your data in the following structure:

```
project_root/
│
├── Thyroid Data/                           # Original dataset
│   ├── benign/                             # Benign cases
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   └── malignant/                          # Malignant cases
│       ├── image001.jpg
│       ├── image002.jpg
│       └── ...
│
├── image_conditional_cgan_results/         # CGAN outputs (auto-generated)
│   └── generated_variations/
│       ├── benign/                         # Synthetic benign images
│       │   ├── generated_001.png
│       │   └── ...
│       └── malignant/                      # Synthetic malignant images
│           ├── generated_001.png
│           └── ...
│
├── cgan_training.py                        # Stage 1: CGAN training script
├── classification_comparison.py            # Stage 2: Classification script
└── requirements.txt
```

**Important Notes:**
- Class folder names must match between original and generated datasets
- Supported image formats: `.jpg`, `.jpeg`, `.png`
- Do not upload actual medical data to public repositories
- The `image_conditional_cgan_results/` folder will be created automatically during CGAN training

## 🚀 Usage

### Installation

```bash
pip install torch torchvision scikit-learn xgboost pandas matplotlib seaborn numpy Pillow joblib scipy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Step 1: Train CGAN (Data Augmentation)

```bash
python cgan_training.py
```

**What happens:**
- Loads original thyroid images from `Thyroid Data/`
- Trains conditional GAN for each class
- Generates synthetic images with controlled variations
- Saves outputs to `image_conditional_cgan_results/generated_variations/`
- **Seeds are set** for reproducibility (manual_seed, cudnn settings)

**Key Parameters** (configurable in script):
- `num_epochs`: 500 (default)
- `batch_size`: 16
- `latent_dim`: 100
- `num_variations`: Number of synthetic images per class

### Step 2: Train Classification Models

```bash
python classification_comparison.py
```

**What happens:**
- Loads both original and CGAN-generated data
- Trains 14 model variants:
  - 4 architectures × 3 configurations (with CGAN, without CGAN, with CGAN+Gaussian)
  - 2 feature-based ML models (XGBoost)
- Evaluates each model with comprehensive metrics
- Generates visualizations and reports
- **Seeds are set** for reproducible train/val/test splits

**Outputs:**
- `comprehensive_model_comparison_results.csv` - Complete results table
- `confusion_matrix_{model}.png` - Per-model confusion matrices
- `roc_curves_per_class_{model}.png` - ROC curves
- `pr_curves_per_class_{model}.png` - Precision-Recall curves
- `comprehensive_model_comparison.png` - Overall comparison
- `detailed_model_comparison.png` - Architecture analysis heatmaps
- `model_{model_name}.pth` - Trained model weights

## 📊 Evaluation Metrics

The framework provides comprehensive evaluation:

- **Classification:** Accuracy, Precision, Recall, F1-Score
- **Clinical:** Sensitivity, Specificity
- **Threshold-Independent:** AUC-ROC (macro/weighted/per-class), Average Precision
- **Visualizations:** Confusion matrices, ROC curves, PR curves, comparative heatmaps

## 🎯 Key Features

- ✅ **Reproducible Results:** Fixed random seeds throughout pipeline
- ✅ **Class Imbalance Handling:** Automatic class weight computation
- ✅ **Multiple Architectures:** ResNet50, Swin Transformer, EfficientNet, ViT
- ✅ **Preprocessing Strategies:** Standard, Gaussian noise injection
- ✅ **Feature-Based ML:** ResNet50 features + XGBoost
- ✅ **Comprehensive Metrics:** 10+ evaluation metrics per model
- ✅ **Automatic Visualization:** Publication-ready plots and heatmaps

## 🔧 Configuration

### CGAN Training Parameters

Edit `cgan_training.py`:

```python
num_epochs = 500          # Training epochs
batch_size = 16           # Batch size
latent_dim = 100          # Noise vector dimension
lr = 0.0002              # Learning rate
num_variations = 1000     # Synthetic images per class
```

### Classification Parameters

Edit `classification_comparison.py`:

```python
img_size = 224           # Input image size
batch_size = 32          # Batch size
epochs = 100             # Training epochs
lr = 1e-4               # Learning rate
```

---

**Status:** ✨ Research Ready | 🔬 Reproducible | 🏥 Medical AI Application
