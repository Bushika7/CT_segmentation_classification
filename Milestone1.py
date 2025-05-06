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
# .nii Read Data
# from PyCode.Code import VolumeCutBrowser
from source.NiftyIO import readNifty
# from NiftyIO import readNifty
from source.BasicSegmentation import BasicSegmentation

# Validation 
from scipy.ndimage.morphology import distance_transform_edt as bwdist
from scipy.ndimage import distance_transform_edt as bwdist
from skimage.measure import find_contours as contour
from source.BasicSegmentation import BasicSegmentation

# Segmentation Quality Scores
from source.SegmentationQualityScores import RelVolDiff, VOE, DICE, DistScores

from pathlib import Path

print("All packages installed and working!")

######## 1. LOAD DATA

#### Data Folders

root_folder = Path(__file__).parent #  automatically finds the folder containing this, Milestone1.py
voi_path = Path(root_folder, r'VOIs\VOIs') # using relative paths means it's easier to work with the project on multiple computers

NiiFile='LIDC-IDRI-0001_R_1.nii.gz'


#### Load Intensity Volume
# NiiFile=Path(voi_path,'image', NiiFile)
NiiFile=Path(voi_path, 'image', NiiFile)
print("PATH OF THE FILE ABOVE", NiiFile)
niiROI,_=readNifty(NiiFile)
image_folder = Path(voi_path,'image')
image_files = [f for f in os.listdir(image_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]


# Ground truth  mask of the lesion (niiROIGT)
NiiFile=Path(voi_path,'nodule_mask',NiiFile)
niiROIGT,_=readNifty(NiiFile)
nodule_mask_folder = Path(voi_path,'nodule_mask')
nodule_mask_files = [f for f in os.listdir(nodule_mask_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]


# List to hold information
data_image = []
for file in image_files:
    file_path = Path(image_folder, file)  
    # Load the .nii.gz file
    img ,_=readNifty(file_path)
    
    # Extract some information
    file_info = {
        'filename': file,
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

    # Extract some information
    file_info = {
        'filename': file,
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

# Force them to be numpy arrays
#ct_data = np.array(ct_data)
#data_mask = np.array(data_nodule_mask)

# Extract the VOI: Only where mask > 0
ct_image = ct_image.rename(columns= {'shape' : 'Image shape', 'mean_intensity' : 'Image mean intensity', 'max_intensity' : 'Image max intensity', 'min_intensity' : 'Image min intensity' })
ct_image.head(5)

# Extract the VOI: Only where mask > 0
ct_nodule_mask = ct_nodule_mask.rename(columns= {'shape' : 'Nodule shape', 'mean_intensity' : 'Nodule mean intensity', 'max_intensity' : 'Nodule max intensity', 'min_intensity' : 'Nodule min intensity' })
ct_nodule_mask.head(5)

vois = pd.merge(ct_image, ct_nodule_mask, on ='filename')
print("DataFrame created! Printing the first 5 elements")
vois.head(5)

# We import the data from the directory where it is saved in our computer
# dirname = "C:\\Users\\sansg\\Downloads\\MetadatabyNoduleMaxVoting.xlsx"
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

# 1. Use a classic standard pipeline over intensity volumes.
# 2. Use Otsu threholding and different morphological operations.
row = vois.iloc[0]
print(row)

# data_folder = r'C:\Users\sansg\OneDrive\Escriptori\Assignment 2'
data_folder = voi_path
segmenter = BasicSegmentation(row, data_folder=data_folder)

segmenter.visualize_original()
segmenter.preprocess()
segmenter.binarize()
segmenter.plot_histogram()
segmenter.visualize_segmentation()
segmenter.postprocess()

# 3. Quantify the performance using fair segmentation metrics.
pred_mask = segmenter.postprocess()
gt_mask = segmenter.gt_mask.astype(bool)
print(gt_mask)
print(pred_mask)
print("pred_mask shape:", np.asarray(pred_mask).shape)
print("gt_mask shape:", np.asarray(gt_mask).shape)
print("DICE: ", DICE(pred_mask, gt_mask))
print("VOE: ", VOE(pred_mask, gt_mask))
print("RelVolDiff: ", RelVolDiff(pred_mask, gt_mask))
print("DistScores: ", DistScores(pred_mask, gt_mask))

# 4. Use kmeans over classic filter banks
segmenter.segment_with_kmeans(n_clusters=2)

