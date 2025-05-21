#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "Guillermo Torres, Debora Gil and Pau Cano (adapted)"
__license__ = "GPLv3"
__email__ = "gtorres,debora,pau@cvc.uab.cat"
__year__ = "2023"
"""

import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from pathlib import Path

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ CONFIG ============
ROOT_FOLDER_IMAGE = Path(__file__).parent.parent
VOI_IMAGE_FOLDER = ROOT_FOLDER_IMAGE / 'VOIs' / 'VOIs' / 'image'
ROOT_FOLDER_GLCM = Path(__file__).parent
GLCM_FEATURES_FILE = ROOT_FOLDER_GLCM / 'slice_glcm1d.npz'
TOP_K_GLCM_FEATURES = 8  # Number of GLCM features to keep

# Load GLCM data
glcm_data = np.load(GLCM_FEATURES_FILE, allow_pickle=True)
glcm_features = glcm_data["slice_features"]
slice_meta = glcm_data["slice_meta"]
ranking_idx = glcm_data["features_rankin_idx"]

# Use only top N GLCM features (based on statistical significance)
top_n_glcm_features = 10
glcm_features = glcm_features[:, ranking_idx[:top_n_glcm_features]]

# Load VGG model (pretrained) and truncate at first FC ReLU
model = models.vgg16(pretrained=True).to(device)
vgg_features = model.features
vgg_avgpool = model.avgpool
vgg_classifier = nn.Sequential(*list(model.classifier.children())[:2])  # Up to first ReLU
model.eval()

# Image transform for VGG
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_vgg_feature(image_slice):
    X = np.stack([image_slice]*3, axis=2)
    X = X.transpose((2, 0, 1))
    tensor = torch.from_numpy(X).float()
    tensor = transform(tensor)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        out = vgg_features(tensor)
        out = vgg_avgpool(out)
        out = out.view(1, -1)
        out = vgg_classifier(out)
    return out.cpu().numpy().flatten()

# Main feature extraction loop
all_features = []
all_labels = []


# Feature extraction loop
for i, meta in enumerate(slice_meta):
    filename, _, _, diagnosis = meta
    if diagnosis not in ("Benign", "Malignant"):
        continue

    # Rename to R_ format
    voiname = meta[0]  
    parts = voiname.split('_') 
    filename = f"{parts[0]}_R_{parts[2]}.nii.gz"
    slice_index = int(parts[2])

    image_path = os.path.join(VOI_IMAGE_FOLDER, filename)


    if not os.path.exists(image_path):
        print('This path does not exists',image_path)
        continue

    # Load and check slice shape
    nii = nib.load(image_path)
    img_data = nii.get_fdata()

        # Handle 3D image: extract center slice if needed
    if img_data.ndim == 3:
        center_slice_idx = img_data.shape[2] // 2
        img_data = img_data[:, :, center_slice_idx]

    # Still check shape
    if img_data.ndim != 2:
        print(f"Warning: skipping {image_path} with shape {img_data.shape}")
        continue

    # Prepare for VGG
    img = np.stack([img_data] * 3, axis=2).astype(np.uint8)
    img = img.transpose((2, 0, 1))
    tensor = transform(torch.from_numpy(img)).unsqueeze(0)

    with torch.no_grad():
        out = vgg_features(tensor)
        out = vgg_avgpool(out).view(1, -1)
        out = vgg_classifier(out)
        feature_vector = out.numpy().squeeze()

    
    glcm_vector = glcm_features[i]  # Already filtered using ranking_idx
    combined_vector = np.concatenate([ feature_vector, glcm_vector])
    all_features.append(combined_vector)
    all_labels.append(0 if diagnosis == "Benign" else 1)
    print(f"[{i}] Processed {filename} | Label: {diagnosis} ({all_labels[-1]}) | VGG shape: { feature_vector.shape} | GLCM shape: {glcm_vector.shape} | Combined: {combined_vector.shape}")


print("Total slices processed:", len(all_features))
print("Feature matrix shape:", np.array(all_features).shape)
print("Labels shape:", np.array(all_labels).shape)
unique, counts = np.unique(all_labels, return_counts=True)
print("Label distribution:", dict(zip(unique, counts)))

# Convert and reduce features
all_features = np.array(all_features)
all_labels = np.array(all_labels)




pca = PCA(n_components=50)
features_pca = pca.fit_transform(all_features)
print("PCA output shape:", features_pca.shape)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features_pca, all_labels, test_size=0.2, random_state=42)

# Train SVM + calibration
clf = SVC(probability=True, class_weight='balanced')
calibrated_clf = CalibratedClassifierCV(clf, n_jobs=-1)
calibrated_clf.fit(X_train, y_train)

# Evaluate
y_pred = calibrated_clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))
