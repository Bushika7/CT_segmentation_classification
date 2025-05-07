# -*- coding: utf-8 -*-
"""
Milestone 1: Segmentation - Generation of an Annotated Dataset
@author: sansg
"""

### === PYTHON LIBRARIES ===
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import os
from skimage.filters import threshold_otsu
from skimage import morphology as Morpho
from scipy.ndimage import filters as filt
from NiftyIO import readNifty
from scipy.ndimage import distance_transform_edt as bwdist
from skimage.measure import find_contours as contour
from pathlib import Path

print("All packages installed and working!")

### === PATH SETUP ===
root_folder = Path(__file__).parent.parent
voi_path = Path(root_folder, r'CT')
output_file = Path(root_folder, "Final_Metadata_MultiNodules.xlsx")
voi_output_folder = Path(root_folder, 'Exported_VOIs')
voi_output_folder.mkdir(parents=True, exist_ok=True)

image_folder = Path(voi_path, 'image')
nodule_mask_folder = Path(voi_path, 'nodule_mask')

image_files = [f for f in os.listdir(image_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]
nodule_mask_files = [f for f in os.listdir(nodule_mask_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]

### === FUNCTION TO SAVE VOI ===
def save_voi(image, mask, patient_id, nodule_id):
    coords = np.array(np.where(mask > 0))
    if coords.size == 0:
        return  # Skip empty masks

    minz, miny, minx = coords.min(axis=1)
    maxz, maxy, maxx = coords.max(axis=1)

    margin = 5
    minz = max(minz - margin, 0)
    miny = max(miny - margin, 0)
    minx = max(minx - margin, 0)
    maxz = min(maxz + margin, mask.shape[0])
    maxy = min(maxy + margin, mask.shape[1])
    maxx = min(maxx + margin, mask.shape[2])

    voi_image = image[minz:maxz, miny:maxy, minx:maxx]
    voi_mask = mask[minz:maxz, miny:maxy, minx:maxx]

    affine = np.eye(4)  # Placeholder affine
    image_nii = nib.Nifti1Image(voi_image, affine)
    mask_nii = nib.Nifti1Image(voi_mask, affine)

    nib.save(image_nii, voi_output_folder / f"{patient_id}_R_{nodule_id}_voi_image.nii.gz")
    nib.save(mask_nii, voi_output_folder / f"{patient_id}_R_{nodule_id}_voi_mask.nii.gz")

### === LOAD CT IMAGES METADATA ===
data_image = []
for file in image_files:
    file_path = Path(image_folder, file)
    img, _ = readNifty(file_path)
    patient_id = file.replace('.nii.gz', '').replace('.nii', '')
    
    data_image.append({
        'filename': file,
        'patient_id': patient_id,
        'shape': img.shape,
        'mean_intensity': np.mean(img),
        'max_intensity': np.max(img),
        'min_intensity': np.min(img)
    })

### === LOAD NODULE MASKS METADATA ===
data_nodule_mask = []
for file in nodule_mask_files:
    file_path = Path(nodule_mask_folder, file)
    nodule_mask, _ = readNifty(file_path)

    base = file.replace('.nii.gz', '').replace('.nii', '')
    parts = base.split('_R_')
    patient_id = parts[0]
    nodule_id = parts[1] if len(parts) > 1 else '0'

    data_nodule_mask.append({
        'filename': file,
        'patient_id': patient_id,
        'nodule_id': nodule_id,
        'shape': nodule_mask.shape,
        'mean_intensity': np.mean(nodule_mask),
        'max_intensity': np.max(nodule_mask),
        'min_intensity': np.min(nodule_mask)
    })

print("All data imported!")
print("Length of data_mask: ", len(data_nodule_mask))
print("Length of data_image: ", len(data_image))

### === CREATE DATAFRAMES ===
ct_image = pd.DataFrame(data_image)
ct_nodule_mask = pd.DataFrame(data_nodule_mask)

ct_image = ct_image.rename(columns={
    'shape': 'Image shape',
    'mean_intensity': 'Image mean intensity',
    'max_intensity': 'Image max intensity',
    'min_intensity': 'Image min intensity'
})

ct_nodule_mask = ct_nodule_mask.rename(columns={
    'shape': 'Nodule shape',
    'mean_intensity': 'Nodule mean intensity',
    'max_intensity': 'Nodule max intensity',
    'min_intensity': 'Nodule min intensity'
})

### === MERGE CT IMAGE & MASK METADATA ===
vois = pd.merge(ct_image, ct_nodule_mask, on='patient_id', how='left')
print("DataFrame created! Printing the first 5 elements")
print(vois.head(5))

### === IMPORT AND PROCESS METADATA FILE ===
metadata_file = Path(root_folder, "MetadatabyNoduleMaxVoting.xlsx")
df = pd.read_excel(metadata_file)

available_patients = [file.replace('.nii.gz', '').replace('.nii', '') for file in image_files]
df_filtered = df[df['patient_id'].isin(available_patients)]

max_voting = df_filtered.groupby(['patient_id', 'nodule_id']).agg(lambda x: x.mode()[0]).reset_index()
print("Grouped metadata by patient and nodule:")
print(max_voting.head(5))

print("# Unique patients in original metadata:", len(df['patient_id'].unique()))
print("# Unique patients after grouping:", len(max_voting['patient_id'].unique()))

### === MAX-VOTING DIAGNOSIS ===
malign = np.where(max_voting['Malignancy_value'] > 3)
benign = np.where(max_voting['Malignancy_value'] <= 3)
max_voting.loc[malign[0], 'Diagnosis_value'] = 1
max_voting.loc[malign[0], 'Diagnosis'] = 'Malign'
max_voting.loc[benign[0], 'Diagnosis_value'] = 0
max_voting.loc[benign[0], 'Diagnosis'] = 'Benign'

print("Diagnosis assigned using max-voting:")
print(max_voting[['patient_id', 'nodule_id', 'Diagnosis']].head())

### === EXPORT VOIs BASED ON METADATA ===
print("Saving VOIs to disk...")

for idx, row in max_voting.iterrows():
    patient_id = row['patient_id']
    nodule_id = str(row['nodule_id'])

    possible_mask_names = [
        f"{patient_id}_R_{nodule_id}.nii.gz",
        f"{patient_id}_R_{nodule_id}.nii"
    ]
    
    mask_file = next((f for f in nodule_mask_files if f in possible_mask_names), None)
    if mask_file is None:
        print(f"❌ Mask not found for {patient_id}, Nodule {nodule_id}")
        continue
    
    mask_path = Path(nodule_mask_folder, mask_file)
    mask, _ = readNifty(mask_path)

    image_file = next((f for f in image_files if f.startswith(patient_id)), None)
    if image_file is None:
        print(f"❌ CT image not found for {patient_id}")
        continue

    image_path = Path(image_folder, image_file)
    image, _ = readNifty(image_path)

    save_voi(image, mask, patient_id, nodule_id)

print("✅ All VOIs exported!")

### === MERGE FINAL DATAFRAME ===
# Ensure same type for merging
ct_nodule_mask['nodule_id'] = ct_nodule_mask['nodule_id'].astype(str)
max_voting['nodule_id'] = max_voting['nodule_id'].astype(str)
df_final = pd.merge(ct_nodule_mask, max_voting, on=['patient_id', 'nodule_id'], how='inner')
df_final.to_excel(output_file, index=False)
print(f"✅ Final metadata saved to: {output_file}")
