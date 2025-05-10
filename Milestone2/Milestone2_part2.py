from radiomics import featureextractor
import SimpleITK as sitk
import os
from pathlib import Path
import pandas as pd

# Define paths
root_folder = Path(__file__).parent.parent
voi_image_dir = root_folder / 'VOIs' / 'VOIs' / 'image'
voi_mask_dir = root_folder / 'VOIs' / 'VOIs' / 'nodule_mask'
output_csv = "radiomic_features.csv"

# Initialize the extractor
extractor = featureextractor.RadiomicsFeatureExtractor()

# Checking how many mask files we have (since for an image we can have more than one mask)
mask_files = sorted([f for f in os.listdir(voi_mask_dir) if f.endswith(".nii.gz")])
print(f"Found {len(mask_files)} mask files")

# Limit how many files to process for testing purposes
max_files = 10

# Collect features
all_features = []

for i, mask_file in enumerate(mask_files):
    if i >= max_files:
        print("Reached test file limit.")
        break

    base_id = mask_file.replace(".nii.gz", "")
    image_path = voi_image_dir / (base_id + ".nii.gz")
    mask_path = voi_mask_dir / mask_file

    print(f"Processing: {base_id}")

    if not image_path.exists():
        print(f"Skipped {base_id}: image not found.")
        continue

    try:

        image = sitk.ReadImage(str(image_path))
        mask = sitk.ReadImage(str(mask_path))
        result = extractor.execute(image, mask)
        result["Nodule_ID"] = base_id
        result["Image_ID"] = base_id
        all_features.append(result)
        print(f"Done: {base_id}")
    except Exception as e:
        print(f"Error with {base_id}: {e}")

# Save to CSV for later use 
if all_features:
    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)
    print(f"Features saved into csv file")
else:
    print("No features extracted.")
