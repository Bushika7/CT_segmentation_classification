import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Paths to your data
root_folder = Path(__file__).parent.parent
ct_dir = root_folder / 'CT' / 'image'
mask_dir = root_folder / 'CT'  / 'nodule_mask'
out_dir = "slices"         # Output folder

os.makedirs(out_dir, exist_ok=True)

# Get list of patients
patients = [f for f in os.listdir(ct_dir) if f.endswith(".nii.gz")]

for ct_file in tqdm(patients, desc="Processing patients"):
    patient_id = ct_file.replace("_ct.nii.gz", "")
    
    ct_path = os.path.join(ct_dir, f"{patient_id}_ct.nii.gz")
    mask_path = os.path.join(mask_dir, f"{patient_id}_mask.nii.gz")
    
    # Load CT and mask
    ct_nii = nib.load(ct_path)
    mask_nii = nib.load(mask_path)
    ct_img = ct_nii.get_fdata()
    mask_img = mask_nii.get_fdata()

    # Check shape match
    if ct_img.shape != mask_img.shape:
        print(f"Shape mismatch for {patient_id}")
        continue

    # Loop over slices with non-zero mask
    for i in range(ct_img.shape[2]):  # axial slices
        if np.any(mask_img[:, :, i]):
            slice_array = ct_img[:, :, i]
            
            # Optionally normalize to [0, 255] and convert to uint8
            slice_array = (slice_array - np.min(slice_array)) / (np.max(slice_array) - np.min(slice_array) + 1e-5)
            slice_array = (slice_array * 255).astype(np.uint8)

            # Save as .npy
            out_name = f"{patient_id}_R_nodule_{i}.npy"
            np.save(os.path.join(out_dir, out_name), slice_array)
