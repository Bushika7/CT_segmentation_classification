# -*- coding: utf-8 -*-
"""
Created on Sun May  4 16:56:45 2025

@author: sansg
"""
### IMPORT PY LIBRARIES
# Python Library 2 manage volumetric data
import numpy as np
import nibabel as nib
import pandas as pd

# Pyhton standard Visualization Library
import matplotlib.pyplot as plt
# Pyhton standard IOs Library
import os
# Basic Processing
from skimage.filters import threshold_otsu
from skimage import morphology as Morpho
from scipy.ndimage import filters as filt
from NiftyIO import readNifty




# Validation 
from scipy.ndimage import distance_transform_edt as bwdist
from skimage.measure import find_contours as contour


from pathlib import Path

print("All packages installed and working!")

######## 1. LOAD DATA

#### Data Folders

root_folder = Path(__file__).parent.parent #  automatically finds the folder containing this, Milestone1.py
voi_path = Path(root_folder, r'CT') # using relative paths means it's easier to work with the project on multiple computers
output_file = Path(root_folder, "Final_Metadata_MultiNodules.xlsx")



#### Load Images
image_folder = Path(voi_path,'image')
image_files = [f for f in os.listdir(image_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]


# Load Nodules Masks
nodule_mask_folder = Path(voi_path,'nodule_mask')
nodule_mask_files = [f for f in os.listdir(nodule_mask_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]


# List to hold information
data_image = []
for file in image_files:
    file_path = Path(image_folder, file)  
    # Load the .nii.gz file
    img ,_=readNifty(file_path)
    patient_id = file.replace('.nii.gz', '').replace('.nii', '')
    
    # Extract some information
    file_info = {
        'filename': file,
        'patient_id': patient_id,
        'shape': img.shape,
        'mean_intensity': np.mean(img),
        'max_intensity': np.max(img),
        'min_intensity': np.min(img)
    }
    data_image.append(file_info)
    
data_nodule_mask = []
for file in nodule_mask_files:
    file_path = Path(nodule_mask_folder, file)
    # Load the .nii.gz file
    nodule_mask ,_=readNifty(file_path)
     # Extract patient_id and nodule_id
    base = file.replace('.nii.gz', '').replace('.nii', '')
    parts = base.split('_R_')
    patient_id = parts[0]
    nodule_id = parts[1] if len(parts) > 1 else '0'  # fallback in case missing


    # Extract some information
    file_info = {
        'filename': file,
        'patient_id': patient_id,
        'nodule_id': nodule_id,
        'shape': nodule_mask.shape,
        'mean_intensity': np.mean(nodule_mask),
        'max_intensity': np.max(nodule_mask),
        'min_intensity': np.min(nodule_mask)
    }
    
    data_nodule_mask.append(file_info)
    
print("All data imported!")

print("Length of data_mask: ", len(data_nodule_mask))
print("Length of data_image: ", len(data_image))


# Create a DataFrame
ct_image = pd.DataFrame(data_image)
ct_nodule_mask = pd.DataFrame(data_nodule_mask)

# Extract the VOI: Only where mask > 0
ct_image = ct_image.rename(columns= {'shape' : 'Image shape', 'mean_intensity' : 'Image mean intensity', 'max_intensity' : 'Image max intensity', 'min_intensity' : 'Image min intensity' })
ct_image.head(5)

# Extract the VOI: Only where mask > 0
ct_nodule_mask = ct_nodule_mask.rename(columns= {'shape' : 'Nodule shape', 'mean_intensity' : 'Nodule mean intensity', 'max_intensity' : 'Nodule max intensity', 'min_intensity' : 'Nodule min intensity' })
ct_nodule_mask.head(5)

vois = pd.merge(ct_image, ct_nodule_mask, on ='patient_id',how='left')
print("DataFrame created! Printing the first 5 elements")
vois.head(5)

#Procedure to Export the VOIs


# Where to save VOIs
voi_output_folder = Path(root_folder, 'Exported_VOIs')
voi_output_folder.mkdir(parents=True, exist_ok=True)

def save_voi(image, mask, patient_id, nodule_id):
    # Find bounding box of the nodule
    coords = np.array(np.where(mask > 0))
    if coords.size == 0:
        return  # skip empty masks
    minz, miny, minx = coords.min(axis=1)
    maxz, maxy, maxx = coords.max(axis=1)

    # Add small margin
    margin = 5
    minz = max(minz - margin, 0)
    miny = max(miny - margin, 0)
    minx = max(minx - margin, 0)
    maxz = min(maxz + margin, mask.shape[0])
    maxy = min(maxy + margin, mask.shape[1])
    maxx = min(maxx + margin, mask.shape[2])

    # Crop
    voi_image = image[minz:maxz, miny:maxy, minx:maxx]
    voi_mask = mask[minz:maxz, miny:maxy, minx:maxx]

    # Save as NIfTI
    affine = np.eye(4)  # Use identity if you don’t have real affine
    image_nii = nib.Nifti1Image(voi_image, affine)
    mask_nii = nib.Nifti1Image(voi_mask, affine)

    nib.save(image_nii, voi_output_folder / f"{patient_id}_R_{nodule_id}_voi_image.nii.gz")
    nib.save(mask_nii, voi_output_folder / f"{patient_id}_R_{nodule_id}_voi_mask.nii.gz")



# We import the meta data file
dirname=Path(root_folder,"MetadatabyNoduleMaxVoting.xlsx")
df = pd.read_excel(dirname)
df.head()

max_voting = df.groupby(['patient_id']).agg(lambda x: x.mode()[0]).reset_index()
print("Data imported and grouped by patient! Printing the first 5 elements")
max_voting.head(5)

#We check the number of different patients in df and we compare it to the number of different patients in max_voting
print("# Different patients in the original dataset: ", len(df['patient_id'].unique()))
print("# Different patients in the dataset grouped by patients: ", len(max_voting['patient_id'].unique()))

#Make Max-Voting to obtain the “Diagnosis”:
#    1. if two or more radiologists have characterized the nodule with a Malignancy score > 3, then Diagnosis=1 (malignant),
#    2. otherwise Diagnosis=0 (benign).
malign = np.where(max_voting['Malignancy_value'] > 3)
benign = np.where(max_voting['Malignancy_value'] <= 3)
max_voting.loc[malign[0], 'Diagnosis_value'] = 1
max_voting.loc[malign[0], 'Diagnosis'] = 'Malign'
max_voting.loc[benign[0], 'Diagnosis_value'] = 0
max_voting.loc[benign[0], 'Diagnosis'] = 'Benign'
print("Max-Voting created! Printing the first 5 elements")
print(max_voting.head(5))

# 10. MERGE EVERYTHING
df_final = pd.merge(vois, max_voting, on='patient_id', how='left')

# 11. EXPORT FINAL METADATA
df_final.to_excel(output_file, index=False)
print(f"\n✅ Final metadata saved to: {output_file}")