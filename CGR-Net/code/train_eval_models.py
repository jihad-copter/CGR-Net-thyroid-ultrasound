import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, ConcatDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay, 
                           accuracy_score, roc_auc_score, roc_curve, precision_recall_curve,
                           f1_score, precision_score, recall_score, average_precision_score)
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import joblib
from scipy import interp
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Paths
original_path = r"Thyroid Data"
augmented_path = r"\image_conditional_cgan_results\generated_variations"

# Parameters
img_size = 224
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define different preprocessing transforms
class GaussianNoise(torch.nn.Module):
    """Add Gaussian noise to images"""
    def __init__(self, mean=0., std=0.1):
        super().__init__()
        self.std = std
        self.mean = mean
        
    def forward(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# Basic preprocessing (current "transformer" preprocessing)
transform_basic = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Gaussian preprocessing
transform_gaussian = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    GaussianNoise(mean=0.0, std=0.05)  # Add Gaussian noise
])

# Load datasets with different preprocessing
original_dataset_basic = ImageFolder(original_path, transform=transform_basic)
augmented_dataset_basic = ImageFolder(augmented_path, transform=transform_basic)

original_dataset_gaussian = ImageFolder(original_path, transform=transform_gaussian)
augmented_dataset_gaussian = ImageFolder(augmented_path, transform=transform_gaussian)

class_names = original_dataset_basic.classes
print(f"Classes detected: {class_names}")

# Compute class weights from original dataset
targets = []
for _, label in original_dataset_basic:
    targets.append(label)
targets = np.array(targets).flatten()

print(f"Targets shape: {targets.shape}")
print(f"Unique targets: {np.unique(targets)}")

class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Split original dataset into train, val, test
train_size = int(0.7 * len(original_dataset_basic))
val_size = int(0.15 * len(original_dataset_basic))
test_size = len(original_dataset_basic) - train_size - val_size
print(f"Dataset split - Train: {train_size}, Val: {val_size}, Test: {test_size}")

train_original_basic, val_data_basic, test_data_basic = random_split(
    original_dataset_basic, [train_size, val_size, test_size])

train_original_gaussian, val_data_gaussian, test_data_gaussian = random_split(
    original_dataset_gaussian, [train_size, val_size, test_size])

# Create combined datasets for different preprocessing approaches
train_dataset_basic = ConcatDataset([train_original_basic, augmented_dataset_basic])
train_dataset_gaussian = ConcatDataset([train_original_gaussian, augmented_dataset_gaussian])

# Results summary storage
results_summary = []
roc_data = {}

def get_predictions_and_probabilities(model, test_loader, device):
    """Get predictions and probabilities for evaluation"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    all_labels = np.array(all_labels).flatten()
    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs)
    
    return all_labels, all_preds, all_probs

def calculate_comprehensive_metrics(y_true, y_pred, y_proba, class_names):
    """Calculate comprehensive metrics including AUC-ROC for both classes"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if y_proba.ndim == 1:
        y_proba = np.column_stack([1 - y_proba, y_proba])
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # AUC-ROC metrics
    try:
        auc_macro = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        auc_weighted = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
    except ValueError as e:
        print(f"Warning: Could not calculate AUC-ROC: {e}")
        auc_macro = 0.0
        auc_weighted = 0.0
    
    # Per-class AUC-ROC
    auc_per_class = []
    for i in range(len(class_names)):
        try:
            y_true_binary = (y_true == i).astype(int)
            if len(np.unique(y_true_binary)) > 1:
                auc_class = roc_auc_score(y_true_binary, y_proba[:, i])
            else:
                auc_class = 0.0
            auc_per_class.append(auc_class)
        except Exception as e:
            print(f"Warning: Could not calculate AUC for class {i}: {e}")
            auc_per_class.append(0.0)
    
    # Average Precision (AP) for each class
    ap_per_class = []
    for i in range(len(class_names)):
        try:
            y_true_binary = (y_true == i).astype(int)
            if len(np.unique(y_true_binary)) > 1:
                ap_class = average_precision_score(y_true_binary, y_proba[:, i])
            else:
                ap_class = 0.0
            ap_per_class.append(ap_class)
        except Exception as e:
            print(f"Warning: Could not calculate AP for class {i}: {e}")
            ap_per_class.append(0.0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_macro': auc_macro,
        'auc_weighted': auc_weighted,
        'auc_per_class': auc_per_class,
        'ap_per_class': ap_per_class,
        'sensitivity': recall[1] if len(recall) > 1 else recall[0],
        'specificity': recall[0] if len(recall) > 1 else recall[0]
    }

def plot_roc_curves_per_class(y_true, y_proba, class_names, model_name):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(12, 5))
    
    for i, class_name in enumerate(class_names):
        plt.subplot(1, 2, i+1)
        y_true_binary = (y_true == i).astype(int)
        
        if len(np.unique(y_true_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
            auc_score = roc_auc_score(y_true_binary, y_proba[:, i])
            
            plt.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {class_name} Class\n{model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, f'Only one class present\nfor {class_name}', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'ROC Curve - {class_name} Class\n{model_name}')
    
    plt.tight_layout()
    plt.savefig(f'roc_curves_per_class_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curves(y_true, y_proba, class_names, model_name):
    """Plot Precision-Recall curves for each class"""
    plt.figure(figsize=(12, 5))
    
    for i, class_name in enumerate(class_names):
        plt.subplot(1, 2, i+1)
        y_true_binary = (y_true == i).astype(int)
        
        if len(np.unique(y_true_binary)) > 1:
            precision, recall, _ = precision_recall_curve(y_true_binary, y_proba[:, i])
            ap_score = average_precision_score(y_true_binary, y_proba[:, i])
            
            plt.plot(recall, precision, linewidth=2, label=f'{class_name} (AP = {ap_score:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {class_name} Class\n{model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, f'Only one class present\nfor {class_name}', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'Precision-Recall Curve - {class_name} Class\n{model_name}')
    
    plt.tight_layout()
    plt.savefig(f'pr_curves_per_class_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model_comprehensive(model, name, test_loader, device, class_names):
    """Comprehensive model evaluation with all metrics and visualizations"""
    y_true, y_pred, y_proba = get_predictions_and_probabilities(model, test_loader, device)
    
    # Calculate all metrics
    metrics = calculate_comprehensive_metrics(y_true, y_pred, y_proba, class_names)
    
    # Create detailed classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curves for each class
    plot_roc_curves_per_class(y_true, y_proba, class_names, name)
    
    # Precision-Recall Curves
    plot_precision_recall_curves(y_true, y_proba, class_names, name)
    
    # Store ROC data for comparison plot
    roc_data[name] = {'y_true': y_true, 'y_proba': y_proba, 'auc_macro': metrics['auc_macro']}
    
    # Save detailed report
    pd.DataFrame(report).transpose().to_csv(f"detailed_report_{name}.csv")
    
    # Add to results summary
    results_summary.append({
        "Model": name,
        "Accuracy": metrics['accuracy'] * 100,
        "Sensitivity": metrics['sensitivity'],
        "Specificity": metrics['specificity'],
        "F1_Score": np.mean(metrics['f1']),
        "AUC_Macro": metrics['auc_macro'],
        "AUC_Weighted": metrics['auc_weighted'],
        f"AUC_{class_names[0]}": metrics['auc_per_class'][0],
        f"AUC_{class_names[1]}": metrics['auc_per_class'][1] if len(metrics['auc_per_class']) > 1 else 0,
        f"AP_{class_names[0]}": metrics['ap_per_class'][0],
        f"AP_{class_names[1]}": metrics['ap_per_class'][1] if len(metrics['ap_per_class']) > 1 else 0
    })
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Sensitivity: {metrics['sensitivity']:.3f}")
    print(f"Specificity: {metrics['specificity']:.3f}")
    print(f"F1-Score: {np.mean(metrics['f1']):.3f}")
    print(f"AUC (Macro): {metrics['auc_macro']:.3f}")
    print(f"AUC (Weighted): {metrics['auc_weighted']:.3f}")
    for i, class_name in enumerate(class_names):
        if i < len(metrics['auc_per_class']):
            print(f"AUC {class_name}: {metrics['auc_per_class'][i]:.3f}")
            print(f"AP {class_name}: {metrics['ap_per_class'][i]:.3f}")

def train_model(model, train_loader, device, class_weights, epochs=100, lr=1e-4):
    """Generic training function for any model"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
    
    return model

def create_model(model_type, num_classes=2):
    """Create model based on type"""
    if model_type == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "SwinTransformer":
        from torchvision.models.swin_transformer import swin_t
        model = swin_t(weights="DEFAULT")
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif model_type == "EfficientNet":
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == "ViT":
        from torchvision.models import vit_b_16
        model = vit_b_16(weights="DEFAULT")
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

# =============================================================================
# SYSTEMATIC MODEL COMPARISON
# =============================================================================

# Define models to test
model_types = ["ResNet50", "SwinTransformer", "EfficientNet", "ViT"]

# Create test loaders
test_loader = DataLoader(test_data_basic, batch_size=batch_size)

print("="*80)
print("SYSTEMATIC MODEL COMPARISON")
print("="*80)

for model_type in model_types:
    print(f"\n{'='*60}")
    print(f"TESTING {model_type.upper()} ARCHITECTURE")
    print(f"{'='*60}")
    
    # 1. Model with CGAN (basic preprocessing)
    print(f"\n1. Training {model_type} with CGAN")
    model = create_model(model_type)
    train_loader = DataLoader(train_dataset_basic, batch_size=batch_size, shuffle=True)
    model = train_model(model, train_loader, device, class_weights)
    
    model_name = f"{model_type}_with_CGAN"
    torch.save(model.state_dict(), f"model_{model_name}.pth")
    evaluate_model_comprehensive(model, model_name, test_loader, device, class_names)
    
    # 2. Model without CGAN (only original data)
    print(f"\n2. Training {model_type} without CGAN")
    model = create_model(model_type)
    train_loader = DataLoader(train_original_basic, batch_size=batch_size, shuffle=True)
    model = train_model(model, train_loader, device, class_weights)
    
    model_name = f"{model_type}_without_CGAN"
    torch.save(model.state_dict(), f"model_{model_name}.pth")
    evaluate_model_comprehensive(model, model_name, test_loader, device, class_names)
    
    # 3. Model with CGAN + Gaussian preprocessing
    print(f"\n3. Training {model_type} with CGAN + Gaussian Preprocessing")
    model = create_model(model_type)
    train_loader = DataLoader(train_dataset_gaussian, batch_size=batch_size, shuffle=True)
    model = train_model(model, train_loader, device, class_weights)
    
    model_name = f"{model_type}_with_CGAN_Gaussian"
    torch.save(model.state_dict(), f"model_{model_name}.pth")
    evaluate_model_comprehensive(model, model_name, test_loader, device, class_names)

# =============================================================================
# FEATURE EXTRACTION + MACHINE LEARNING COMPARISON
# =============================================================================

print(f"\n{'='*60}")
print("FEATURE EXTRACTION + MACHINE LEARNING")
print(f"{'='*60}")

def extract_features(model, dataset):
    """Extract features using pre-trained model"""
    model.fc = nn.Identity()  # Remove final classifier for ResNet50
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features, labels = [], []
    
    with torch.no_grad():
        for images, labs in loader:
            images = images.to(device)
            feats = model(images).cpu().numpy()
            features.extend(feats)
            labels.extend(labs.numpy())
    
    features = np.array(features)
    labels = np.array(labels).flatten()
    
    return features, labels

def evaluate_ml_model(y_true, y_pred, y_proba, model_name, class_names):
    """Evaluate ML model and create visualizations"""
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    
    # Calculate metrics
    metrics = calculate_comprehensive_metrics(y_true, y_pred, y_proba, class_names)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC and PR curves
    plot_roc_curves_per_class(y_true, y_proba, class_names, model_name)
    plot_precision_recall_curves(y_true, y_proba, class_names, model_name)
    
    # Store data and results
    roc_data[model_name] = {'y_true': y_true, 'y_proba': y_proba, 'auc_macro': metrics['auc_macro']}
    
    results_summary.append({
        "Model": model_name,
        "Accuracy": metrics['accuracy'] * 100,
        "Sensitivity": metrics['sensitivity'],
        "Specificity": metrics['specificity'],
        "F1_Score": np.mean(metrics['f1']),
        "AUC_Macro": metrics['auc_macro'],
        "AUC_Weighted": metrics['auc_weighted'],
        f"AUC_{class_names[0]}": metrics['auc_per_class'][0],
        f"AUC_{class_names[1]}": metrics['auc_per_class'][1] if len(metrics['auc_per_class']) > 1 else 0,
        f"AP_{class_names[0]}": metrics['ap_per_class'][0],
        f"AP_{class_names[1]}": metrics['ap_per_class'][1] if len(metrics['ap_per_class']) > 1 else 0
    })

# Test XGBoost on ResNet50 features
print("\n1. XGBoost on ResNet50 Features (Original Data)")
model_feat = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
X_train, y_train = extract_features(model_feat, train_original_basic)
X_test, y_test = extract_features(model_feat, test_data_basic)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf_xgb = XGBClassifier(eval_metric='logloss', random_state=42)
clf_xgb.fit(X_train, y_train)
y_pred = clf_xgb.predict(X_test)
y_proba = clf_xgb.predict_proba(X_test)

evaluate_ml_model(y_test, y_pred, y_proba, "XGBoost_ResNet50_Features", class_names)
joblib.dump(clf_xgb, 'model_XGBoost_ResNet50_Features.pkl')

# Test XGBoost on ResNet50 features with CGAN data
print("\n2. XGBoost on ResNet50 Features (CGAN Data)")
X_train_cgan, y_train_cgan = extract_features(model_feat, train_dataset_basic)
X_train_cgan = scaler.fit_transform(X_train_cgan)
X_test = scaler.transform(X_test)

clf_xgb_cgan = XGBClassifier(eval_metric='logloss', random_state=42)
clf_xgb_cgan.fit(X_train_cgan, y_train_cgan)
y_pred_cgan = clf_xgb_cgan.predict(X_test)
y_proba_cgan = clf_xgb_cgan.predict_proba(X_test)

evaluate_ml_model(y_test, y_pred_cgan, y_proba_cgan, "XGBoost_ResNet50_Features_CGAN", class_names)
joblib.dump(clf_xgb_cgan, 'model_XGBoost_ResNet50_Features_CGAN.pkl')

# =============================================================================
# COMPREHENSIVE RESULTS ANALYSIS AND VISUALIZATION
# =============================================================================

print(f"\n{'='*80}")
print("COMPREHENSIVE RESULTS ANALYSIS")
print(f"{'='*80}")

# Save Final Results Summary CSV
df_results = pd.DataFrame(results_summary)
df_results.to_csv("comprehensive_model_comparison_results.csv", index=False)

# Display top performing models
print("\nTop 5 Models by Accuracy:")
top_models = df_results.nlargest(5, 'Accuracy')
print(top_models[['Model', 'Accuracy', 'AUC_Macro', 'F1_Score']])

# 1. Overall Comparison Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Accuracy comparison
axes[0, 0].bar(range(len(df_results)), df_results['Accuracy'], color='skyblue')
axes[0, 0].set_xticks(range(len(df_results)))
axes[0, 0].set_xticklabels(df_results['Model'], rotation=45, ha='right')
axes[0, 0].set_ylabel('Accuracy (%)')
axes[0, 0].set_title('Model Accuracy Comparison')
axes[0, 0].grid(True, alpha=0.3)

# AUC Macro comparison
axes[0, 1].bar(range(len(df_results)), df_results['AUC_Macro'], color='lightcoral')
axes[0, 1].set_xticks(range(len(df_results)))
axes[0, 1].set_xticklabels(df_results['Model'], rotation=45, ha='right')
axes[0, 1].set_ylabel('AUC (Macro)')
axes[0, 1].set_title('Model AUC Macro Comparison')
axes[0, 1].grid(True, alpha=0.3)

# F1-Score comparison
axes[1, 0].bar(range(len(df_results)), df_results['F1_Score'], color='lightgreen')
axes[1, 0].set_xticks(range(len(df_results)))
axes[1, 0].set_xticklabels(df_results['Model'], rotation=45, ha='right')
axes[1, 0].set_ylabel('F1-Score')
axes[1, 0].set_title('Model F1-Score Comparison')
axes[1, 0].grid(True, alpha=0.3)

# Sensitivity vs Specificity
axes[1, 1].scatter(df_results['Sensitivity'], df_results['Specificity'], 
                   s=100, alpha=0.7, c=range(len(df_results)), cmap='viridis')
for i, model in enumerate(df_results['Model']):
    axes[1, 1].annotate(model, (df_results['Sensitivity'][i], df_results['Specificity'][i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
axes[1, 1].set_xlabel('Sensitivity')
axes[1, 1].set_ylabel('Specificity')
axes[1, 1].set_title('Sensitivity vs Specificity')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("comprehensive_model_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Architecture-wise comparison
architectures = ['ResNet50', 'SwinTransformer', 'EfficientNet', 'ViT']
preprocessing_types = ['with_CGAN', 'without_CGAN', 'with_CGAN_Gaussian']

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Accuracy comparison by architecture and preprocessing
accuracy_matrix = np.zeros((len(architectures), len(preprocessing_types)))
for i, arch in enumerate(architectures):
    for j, prep in enumerate(preprocessing_types):
        model_name = f"{arch}_{prep}"
        result = df_results[df_results['Model'] == model_name]
        if not result.empty:
            accuracy_matrix[i, j] = result['Accuracy'].values[0]

im1 = axes[0, 0].imshow(accuracy_matrix, cmap='Blues', aspect='auto')
axes[0, 0].set_xticks(range(len(preprocessing_types)))
axes[0, 0].set_xticklabels(preprocessing_types, rotation=45, ha='right')
axes[0, 0].set_yticks(range(len(architectures)))
axes[0, 0].set_yticklabels(architectures)
axes[0, 0].set_title('Accuracy Heatmap by Architecture and Preprocessing')
for i in range(len(architectures)):
    for j in range(len(preprocessing_types)):
        axes[0, 0].text(j, i, f'{accuracy_matrix[i, j]:.1f}%', 
                       ha='center', va='center', color='red')

# AUC comparison by architecture and preprocessing
auc_matrix = np.zeros((len(architectures), len(preprocessing_types)))
for i, arch in enumerate(architectures):
    for j, prep in enumerate(preprocessing_types):
        model_name = f"{arch}_{prep}"
        result = df_results[df_results['Model'] == model_name]
        if not result.empty:
            auc_matrix[i, j] = result['AUC_Macro'].values[0]

im2 = axes[0, 1].imshow(auc_matrix, cmap='Reds', aspect='auto')
axes[0, 1].set_xticks(range(len(preprocessing_types)))
axes[0, 1].set_xticklabels(preprocessing_types, rotation=45, ha='right')
axes[0, 1].set_yticks(range(len(architectures)))
axes[0, 1].set_yticklabels(architectures)
axes[0, 1].set_title('AUC Macro Heatmap by Architecture and Preprocessing')
for i in range(len(architectures)):
    for j in range(len(preprocessing_types)):
        axes[0, 1].text(j, i, f'{auc_matrix[i, j]:.3f}', 
                       ha='center', va='center', color='blue')

# Effect of CGAN and Gaussian preprocessing
effect_data = []
for arch in architectures:
    base_model = df_results[df_results['Model'] == f"{arch}_without_CGAN"]
    cgan_model = df_results[df_results['Model'] == f"{arch}_with_CGAN"]
    gaussian_model = df_results[df_results['Model'] == f"{arch}_with_CGAN_Gaussian"]
    
    if not base_model.empty and not cgan_model.empty:
        cgan_effect = cgan_model['Accuracy'].values[0] - base_model['Accuracy'].values[0]
        effect_data.append({'Architecture': arch, 'Effect': cgan_effect, 'Type': 'CGAN'})
    
    if not cgan_model.empty and not gaussian_model.empty:
        gaussian_effect = gaussian_model['Accuracy'].values[0] - cgan_model['Accuracy'].values[0]
        effect_data.append({'Architecture': arch, 'Effect': gaussian_effect, 'Type': 'Gaussian'})

effect_df = pd.DataFrame(effect_data)
if not effect_df.empty:
    pivot_effect = effect_df.pivot(index='Architecture', columns='Type', values='Effect')
    pivot_effect.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Effect of CGAN and Gaussian Preprocessing on Accuracy')
    axes[1, 0].set_ylabel('Accuracy Improvement (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)

# Best model summary
best_model = df_results.loc[df_results['Accuracy'].idxmax()]
axes[1, 1].text(0.1, 0.9, f"Best Model: {best_model['Model']}", fontsize=14, weight='bold')
axes[1, 1].text(0.1, 0.8, f"Accuracy: {best_model['Accuracy']:.2f}%", fontsize=12)
axes[1, 1].text(0.1, 0.7, f"AUC Macro: {best_model['AUC_Macro']:.3f}", fontsize=12)
axes[1, 1].text(0.1, 0.6, f"F1-Score: {best_model['F1_Score']:.3f}", fontsize=12)
axes[1, 1].text(0.1, 0.5, f"Sensitivity: {best_model['Sensitivity']:.3f}", fontsize=12)
axes[1, 1].text(0.1, 0.4, f"Specificity: {best_model['Specificity']:.3f}", fontsize=12)
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_ylim(0, 1)
axes[1, 1].set_title('Best Model Summary')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig("detailed_model_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"Total models evaluated: {len(df_results)}")
print(f"Best performing model: {best_model['Model']} with {best_model['Accuracy']:.2f}% accuracy")
print("All results saved to 'comprehensive_model_comparison_results.csv'")
print("="*80)
