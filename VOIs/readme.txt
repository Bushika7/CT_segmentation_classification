Annotated DataSet (VOIs.zip, MetadatabyNoduleMaxVoting.xlsx). 

The file VOIs.zip contains two folders:

image. Contains, for each nodule, .nii.gz files of the VOIs of the CT scan named with the ID of the patient and the ID of the nodule (LIDC-IDRI-0003_R_2.nii.gz, for nodule 2 of patient LIDC-IDRI-0003).
nodule_mask. Contains .nii.gz files of a binary mask of each nodule VOI with the same name that identifies the nodule in the image folder.
The excel file MetadatabyNoduleMaxVoting.xlsx has annotations for each lesion, including the diagnosis (Diagnosis='Malign','Benign'; Diagnosis_value=1,0).