# -*- coding: utf-8 -*-
"""
Created on May 18, 2025
Feature Extraction using VGG16 + PCA
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Step 1: Load VGG16 + FC1 Layer
# ---------------------------
vgg16 = models.vgg16(pretrained=True)
vgg16.eval()

# Keep features up to first FC layer (FC1 outputs 4096 features)
model_features = nn.Sequential(*list(vgg16.classifier.children())[:-2])

# ---------------------------
# Step 2: Transformations for VGG input
# ---------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------
# Step 3: Load slices and labels from Excel
# ---------------------------
def load_slices_from_excel(features, label_col='diagnosis'):
    df = pd.read_excel(features)
    slices = []
    labels = []

    for _, row in df.iterrows():
        patient_id = row['patient_id']
        nodule_id = row['nodule_id']
        slice_number = row['slice_number']
        label = row[label_col]

        slice_path = f"slices/{patient_id}_R_{nodule_id}_{slice_number}.npy"
        if os.path.exists(slice_path):
            slice_array = np.load(slice_path)

            # Convert grayscale to RGB
            if slice_array.ndim == 2:
                slice_array = np.stack([slice_array] * 3, axis=-1)

            slices.append(slice_array)
            labels.append(label)

    return slices, labels

# ---------------------------
# Step 4: Extract VGG16 FC1 features
# ---------------------------
def extract_vgg_features(slices):
    features = []
    with torch.no_grad():
        for img in tqdm(slices, desc="Extracting VGG features"):
            img_t = transform(img).unsqueeze(0)  # [1, 3, 224, 224]
            x = vgg16.features(img_t)
            x = torch.flatten(x, 1)  # [1, 512*7*7]
            fc1_feats = model_features(x)  # [1, 4096]
            features.append(fc1_feats.squeeze().numpy())
    return np.array(features)

# ---------------------------
# Step 5: Standardize + Reduce with PCA
# ---------------------------
def reduce_features_with_pca(features, variance_retained=0.95):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=variance_retained)
    features_pca = pca.fit_transform(features_scaled)

    print(f"Original shape: {features.shape}")
    print(f"Reduced shape: {features_pca.shape}")
    return features_pca, scaler, pca

# ---------------------------
# Step 6: Main Pipeline
# ---------------------------
def main():
    features_xlsx = "features.xlsx"  
    output_xlsx= "vgg_pca_features.xlsx"

    print("Loading slice data...")
    slices, labels = load_slices_from_excel(features_xlsx )

    print("Extracting VGG FC1 features...")
    vgg_feats = extract_vgg_features(slices)

    print("Reducing features with PCA...")
    vgg_pca, _, _ = reduce_features_with_pca(vgg_feats)

    # Combine with labels and save
    df_out = pd.DataFrame(vgg_pca)
    df_out['label'] = labels
    df_out.to_excel(output_xlsx, index=False)
    print(f"Saved reduced features to {output_xlsx}")

if __name__ == "__main__":
    main()
