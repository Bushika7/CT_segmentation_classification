import os
import re
import numpy as np
import nibabel as nib
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold


Features = []  # Features
Labels = []  # Labels
Nodule_id = [] #Nodule ID


root_folder = Path(__file__).parent.parent
image_folder = root_folder / 'VOIs' /'VOIs' / 'image'
mask_folder = root_folder / 'VOIs'  /'VOIs' / 'nodule_mask'

for file in os.listdir(image_folder):
    if file.endswith('.nii.gz'):
        image = nib.load(os.path.join(image_folder, file)).get_fdata()
        match = re.search(r'_R_(\d+)\.nii\.gz$', file)
        if match:
            nodule_id = int(match.group(1))
            print(f"File: {file} → Nodule ID: {nodule_id}")
            #Nodule_id.append(nodule_id)
        mask_file = file.replace('image', 'mask')
        mask = nib.load(os.path.join(mask_folder, mask_file)).get_fdata()

        for i in range(image.shape[2]):  # iterate over slices
            img_slice = image[:, :, i]
            mask_slice = mask[:, :, i]
            
            if np.count_nonzero(img_slice) == 0:
                continue  # Skip empty slices

            label = int(np.any(mask_slice > 0))  # 1 if any nodule pixel, else 0
            features = extract_glcm_features(img_slice)
            
            Features.append(features)
            Labels.append(label)
            Nodule_id.append(nodule_id)

Features = np.array(Features)
Labels = np.array(Labels)
Nodule_id = np.array(Nodule_id)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = SVC()  # default parameters

for fold, (train_idx, test_idx) in enumerate(skf.split(Features, Labels)):
    print(f"\n--- Fold {fold + 1} ---")
    X_train, X_test = Features[train_idx], Features[test_idx]
    y_train, y_test = Labels[train_idx], Labels[test_idx]

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, digits=4))

Features = np.array(Features.tolist())  # Convert to feature matrix
groups = np.array(Nodule_id)
print("Are all the data the same lenght?" ,len(Features) == len(Labels) == len(groups))


cv = StratifiedGroupKFold(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(cv.split(Features, Labels, groups)):
    print(f"Fold {fold + 1}")

    X_train, X_test = Features[train_idx], Features[test_idx]
    y_train, y_test = Labels[train_idx], Labels[test_idx]

    clf = SVC()
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"Accuracy: {acc:.3f}")
