#!/usr/bin/env python

"""
This code had been adjusted to be aligned with the challenge 
"""

import os
import numpy as np
import pandas as pd
from collections import OrderedDict
import SimpleITK as sitk
from radiomics import featureextractor
from NiftyIO import readNifty
from pathlib import Path

from radiomics import setVerbosity
setVerbosity(60)



def ShiftValues(image, value):
    image = image + value
    print("Range after Shift: {:.2f} - {:.2f}".format(image.min(), image.max()))
    return image


def SetRange(image, in_min, in_max):            
    image = (image - image.min()) / (image.max() - image.min())
    image = image * (in_max - in_min) + in_min   
    
    image[image<0] = 0
    image[image>image.max()] = image.max()
    print("Range after SetRange: {:.2f} - {:.2f}".format(image.min(), image.max()))
    return image
    

def SetGrayLevel(image, levels):
    # array's values between 0 & 1
    image = image * levels 
    image = image.astype(np.uint8) # get into integer values
    print("Range after SetGrayLevel: {:.2f} - {:.2f}".format(image.min(), image.max()))
    return image


def saveXLSX(filename, df):
    # write to a .xlsx file.

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    # Close the Pandas Excel writer and output the Excel file.
    writer._save()
    

def GetFeatures(featureVector, i, patient_id, nodule_id, diagnosis):
    new_row = {}
    # Showing the features and its calculated values
    for featureName in featureVector.keys():
        #print("Computed {}: {}".format(featureName, featureVector[featureName]))
        if ('firstorder' in featureName) or ('glszm' in featureName) or \
            ('glcm' in featureName) or ('glrlm' in featureName) or \
            ('gldm' in featureName) or ('shape' in featureName):
                new_row.update({featureName: featureVector[featureName]})
    lst = sorted(new_row.items())  # Ordering the new_row dictionary
    # Adding some columns  
    lst.insert(0, ('diagnosis', diagnosis))
    lst.insert(0, ('slice_number', i))
    lst.insert(0, ('nodule_id', nodule_id))
    lst.insert(0, ('patient_id', patient_id))
    od = OrderedDict(lst)
    return od



def SliceMode(patient_id, nodule_id, diagnosis, image, mask, meta1, meta2, extractor, maskMinPixels=200):

    myList = []
    i = 0

    while i < image.shape[2]:   # X, Y, Z
        # Get the axial cut
        img_slice = image[:,:,i]
        mask_slice = mask[:,:,i]
        try:
            if maskMinPixels < mask_slice.sum():
                # Get back to the format sitk
                img_slice_sitk = sitk.GetImageFromArray(img_slice)
                mask_slice_sitk = sitk.GetImageFromArray(mask_slice)
                    
                # Recover the pixel dimension in X and Y
                (x1, y1, z1) = meta1.spacing
                (x2, y2, z2) = meta2.spacing
                img_slice_sitk.SetSpacing((float(x1), float(y1)))
                mask_slice_sitk.SetSpacing((float(x2), float(y2)))
   
                # Extract features
                featureVector = extractor.execute(img_slice_sitk,
                                                  mask_slice_sitk,
                                                  voxelBased=False)
                od = GetFeatures(featureVector, i, patient_id, nodule_id, diagnosis)
                myList.append(od)
            # else:
            #     print("features extraction skipped in slice-i: {}".format(i))
        except:
            print("Exception: skipped in slice-i: {}".format(i))
        i = i+1
            
    df = pd.DataFrame.from_dict(myList)
    return df


   
#### Paths
root_folder = Path(__file__).parent.parent
imageDirectory = root_folder / 'CT' / 'image'
maskDirectory = root_folder / 'CT'  / 'nodule_mask'
imageName = os.path.join(imageDirectory, 'LIDC-IDRI-0003.nii.gz')
maskName  = os.path.join(maskDirectory, 'LIDC-IDRI-0003_R_2.nii.gz')
####
    

# Use a parameter file, this customizes the extraction settings and
# also specifies the input image types to use and
# which features should be extracted.
params = 'Params.yaml'

# Initializing the feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor(params)


# Reading image and mask
image, meta1 = readNifty(imageName, CoordinateOrder='xyz')
mask, meta2 = readNifty(maskName, CoordinateOrder='xyz')

patient_id = 'LIDC-IDRI-0003'
nodule_id = 2

df_mv = pd.read_excel('MetadatabyNoduleMaxVoting.xlsx', 
                      sheet_name='ML4PM_MetadatabyNoduleMaxVoting', 
                      engine='openpyxl'
        )

diagnosis = df_mv[(df_mv.patient_id==patient_id) & (df_mv.nodule_id==nodule_id)].Diagnosis_value.values[0]

### PREPROCESSING
image = ShiftValues(image, value=1024)
image = SetRange(image, in_min=0, in_max=4000)
image = SetGrayLevel(image, levels=24)



# Extract features slice by slice.
df = SliceMode(patient_id, nodule_id, diagnosis, image, mask, meta1, meta2, extractor, maskMinPixels=200)

saveXLSX('features.xlsx', df)