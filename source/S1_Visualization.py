"""
This is the source code for volume visualization

Machine Learning for Precision Medicine
Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,pcano@cvc.uab.es"
__year__ = "2023"
"""

### IMPORT PY LIBRARIES
# Python Library 2 manage volumetric data
import numpy as np

# Pyhton standard Visualization Library
import matplotlib.pyplot as plt

# Pyhton standard IOs Library
import os

### IMPORT SESSION FUNCTIONS
#### Session Code Folder (change to your path)
SessionPyFolder = "/Users/pau/Downloads/OneDrive_1_12-4-2023"  # r'D:\Teaching\Master\DataSci4Health\2023_ML4PM\Week 08 - Introduction\PyCode_Session8'
os.chdir(SessionPyFolder)  # Change Dir 2 load session functions
# .nii Read Data
from NiftyIO import readNifty

# Volume Visualization
from VolumeCutBrowser import VolumeCutBrowser


######## LOAD DATA

#### Data Folders (change to your path)
SessionDataFolder = "/Users/pau/Downloads/OneDrive_1_12-4-2023"  # r'D:\Teaching\Master\DataSci4Health\2023_ML4PM\Week 08 - Introduction\Dataset'
os.chdir(SessionDataFolder)


CaseFolder = "CT"
NiiFile = "LIDC-IDRI-0001.nii.gz"


#### Load Intensity Volume
NiiFile = os.path.join(SessionDataFolder, CaseFolder, "image", NiiFile)
niivol, niimetada = readNifty(NiiFile)
#### Load Nodule Mask
NiiFile = os.path.join(SessionDataFolder, CaseFolder, "nodule_mask", NiiFile)
niimask, niimetada = readNifty(NiiFile)

######## VOLUME METADATA
print("Voxel Resolution (mm): ", niimetada.spacing)
print("Volume origin (mm): ", niimetada.origen)
print("Axes direction: ", niimetada.direction)
######## VISUALIZE VOLUMES

### Interactive Volume Visualization
# Short Axis View
VolumeCutBrowser(niivol)
VolumeCutBrowser(niivol, IMSSeg=niimask)
# Coronal View
VolumeCutBrowser(niivol, Cut="Cor")
# Sagital View
VolumeCutBrowser(niivol, Cut="Sag")


### Short Axis (SA) Image
# Define SA cut
k = int(niivol.shape[2] / 2)  # Cut at the middle of the volume
SA = niivol[:, :, k]
# Image
fig1 = plt.figure()
plt.imshow(SA, cmap="gray")
plt.close(fig1)  # close figure fig1

# Cut Level Sets
levels = [400]
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect="equal")
ax1.imshow(SA, cmap="gray")
plt.contour(SA, levels, colors="r", linewidths=2)
plt.close("all")  # close all plt figures
