# Combined Code for Lung Nodule Analysis Project

# === COMMON PYTHON LIBRARIES ===
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import re

# For Milestone 1: Segmentation and VOI Generation
from skimage.filters import threshold_otsu
from skimage import morphology as Morpho
from scipy.ndimage import filters as filt
from scipy.ndimage import distance_transform_edt as bwdist
from skimage.measure import find_contours as contour
# For Milestone 2: Data Exploration and Clustering
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.manifold import TSNE, trustworthiness # For t-SNE and trustworthiness
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.metrics import silhouette_score

# For Milestone 3: Classifier Training on GLCM features
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GridSearchCV, RandomizedSearchCV, cross_validate, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, make_scorer
from scipy.stats import ttest_ind, loguniform
from sklearn.pipeline import Pipeline
import scipy.stats as st
# from radiomics import featureextractor # Uncomment if you fix the feature extraction in Milestone2_part2
import SimpleITK as sitk # Required if using PyRadiomics, included based on Milestone2_part2.py snippet

print("All necessary packages are imported!")

# === GLOBAL PANDAS SETTINGS ===
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# ==============================================================================
# --- MILESTONE 1: SEGMENTATION AND VOI GENERATION ---
# ==============================================================================

# === PATH SETUP ===
# User-specified root_folder. Ensure this path points to 'C:\Users\sansg\Downloads'
root_folder = Path("C:\\Users\\sansg\\Downloads")

# Paths structured relative to the user's root_folder
voi_base_dir = root_folder / 'VOIs' / 'VOIs'
image_folder = voi_base_dir / 'image'
nodule_mask_folder = voi_base_dir / 'nodule_mask'

output_file = root_folder / "Final_Metadata_MultiNodules.xlsx"
voi_output_folder = root_folder / 'Exported_VOIs'

# Ensure all necessary directories exist
voi_base_dir.mkdir(parents=True, exist_ok=True)
image_folder.mkdir(parents=True, exist_ok=True)
nodule_mask_folder.mkdir(parents=True, exist_ok=True)
voi_output_folder.mkdir(parents=True, exist_ok=True)

print("Path setup complete.")
print(f"Root folder: {root_folder}")
print(f"Image folder: {image_folder}")
print(f"Nodule mask folder: {nodule_mask_folder}")


# === NIFTYIO Functions (from NiftyIO.py - assumed external or defined here) ===
def readNifty(path):
    """Reads a NIfTI file and returns its data and affine matrix."""
    img = nib.load(path)
    data = img.get_fdata()
    return data, img.affine

def save_voi(image_data, mask_data, patient_id, nodule_id, output_folder, affine):
    """MOCK: Saves a VOI (image and mask) as NIfTI files."""
    print(f"MOCK: Saving VOI for patient {patient_id}, nodule {nodule_id}...")
    # Simulate saving files
    image_output_path = output_folder / 'image' / f"{patient_id}_R_{nodule_id}.nii.gz"
    mask_output_path = output_folder / 'nodule_mask' / f"{patient_id}_R_{nodule_id}.nii.gz"
    print(f"MOCK: Image saved to {image_output_path}")
    print(f"MOCK: Mask saved to {mask_output_path}")
    # In a real scenario, you'd use nibabel.Nifti1Image and nibabel.save
    # import nibabel as nib
    # nib.save(nib.Nifti1Image(image_data, affine), image_output_path)
    # nib.save(nib.Nifti1Image(mask_data, affine), mask_output_path)


# === INITIAL NIFTI FILE SCAN ===
# Explicitly ensure image_folder and nodule_mask_folder are Path objects
# This guards against potential issues where they might be treated as strings
image_folder = Path(image_folder)
nodule_mask_folder = Path(nodule_mask_folder)

# List all .nii.gz files in the specified folders
image_files_list = [f for f in os.listdir(image_folder) if f.endswith('.nii.gz')]
nodule_mask_files_list = [f for f in os.listdir(nodule_mask_folder) if f.endswith('.nii.gz')]

# List to hold information
data_image = []
data_mask = []

print("\nScanning image files...")
# Loop through each image file
for file_name in image_files_list:
    file_path = image_folder / file_name # Use pathlib for path joining
    
    # Load the .nii.gz file using the defined readNifty function
    try:
        img_data, _ = readNifty(file_path)
        
        # Extract some information
        file_info = {
            'filename': file_name,
            'shape': img_data.shape,
            'mean_intensity': np.mean(img_data),
            'max_intensity': np.max(img_data),
            'min_intensity': np.min(img_data)
        }
        data_image.append(file_info)
    except Exception as e:
        print(f"Error loading image file {file_name}: {e}")

print("Scanning mask files...")
# Loop through each nodule mask file
for file_name in nodule_mask_files_list:
    file_path = nodule_mask_folder / file_name # Use pathlib for path joining
    
    # Load the .nii.gz file using the defined readNifty function
    try:
        img_data, _ = readNifty(file_path)
        
        # Extract some information
        file_info = {
            'filename': file_name,
            'shape': img_data.shape,
            'mean_intensity': np.mean(img_data),
            'max_intensity': np.max(img_data),
            'min_intensity': np.min(img_data)
        }
        data_mask.append(file_info)
    except Exception as e:
        print(f"Error loading mask file {file_name}: {e}")

print(f"Length of data_mask: {len(data_mask)}")
print(f"Length of data_image: {len(data_image)}")

# Create DataFrames
ct_data = pd.DataFrame(data_image) # This will store image file info
ct_image = pd.DataFrame(data_mask) # This will store mask file info (name 'ct_image' for mask data is a bit confusing, but keeping user's naming)

print("Initial NIfTI file scan complete. DataFrames created.")


# === SEGMENTATION FUNCTION ===
def lung_segmentation(img):
    # Scale image values to 0-255 range and convert to uint8
    min_val = np.min(img)
    max_val = np.max(img)
    scaled_img = (255 * ((img - min_val) / (max_val - min_val))).astype(np.uint8)

    # Apply Otsu's thresholding
    thresh = threshold_otsu(scaled_img)
    binary = scaled_img > thresh

    # Perform morphological operations
    filled_lung = Morpho.remove_small_holes(binary, area_threshold=2000000000)
    eroded_lung = Morpho.binary_erosion(filled_lung, Morpho.disk(30))
    opened_lung = Morpho.binary_opening(eroded_lung, Morpho.disk(5))
    closed_lung = Morpho.binary_closing(opened_lung, Morpho.disk(10))
    final_lung = Morpho.remove_small_objects(closed_lung, min_size=20000000)

    # Invert to get non-lung tissue (e.g., nodule candidates)
    inverted_mask = ~final_lung

    return inverted_mask

print("Lung segmentation function defined.")

# === METADATA PROCESSING AND VOI EXPORT (from Milestone1_part1.py) ===
print("Loading metadata...")
try:
    metadata_file_path_excel = root_folder / "Final_Metadata_MultiNodules.xlsx" # Full path to the metadata file
    df_raw = pd.read_excel(metadata_file_path_excel)
    print("Metadata loaded successfully.")
except FileNotFoundError:
    print(f"Error: Metadata file not found at {metadata_file_path_excel}. Please ensure the file exists.")
    df_raw = pd.DataFrame() # Create an empty DataFrame to avoid errors later

if not df_raw.empty:
    print("Raw Data (first 5 rows):")
    print(df_raw.head())

    # Data Cleaning and Preparation
    df_raw.columns = df_raw.columns.str.replace(r'[^\w\s]', '', regex=True).str.replace(' ', '_')
    df_raw.rename(columns={'Nodule_ID_': 'nodule_id', 'Patient_ID': 'patient_id'}, inplace=True)

    # Drop rows with NaN in critical columns
    df_clean = df_raw.dropna(subset=['patient_id', 'nodule_id', 'Diagnosis'])
    df_clean['nodule_id'] = df_clean['nodule_id'].astype(int)

    # One-Hot Encoding for Diagnosis
    df_diagnosis = df_clean.pivot_table(index=['patient_id', 'nodule_id'], columns='Diagnosis', aggfunc='size', fill_value=0).reset_index()
    df_diagnosis.columns.name = None

    # Majority Voting for Diagnosis
    def get_max_voting(row, categories):
        counts = {cat: row[cat] for cat in categories}
        max_count = 0
        max_cat = 'Unknown'
        for cat, count in counts.items():
            if count > max_count:
                max_count = count
                max_cat = cat
            elif count == max_count:
                # Handle ties by choosing a specific category or marking as tie
                if max_cat == 'Benign' and cat == 'Malign':
                    max_cat = cat # Malign overrides Benign in a tie
        return max_cat

    diagnosis_categories = ['Benign', 'Malign']
    df_diagnosis['Diagnosis'] = df_diagnosis.apply(lambda row: get_max_voting(row, diagnosis_categories), axis=1)

    # Map Diagnosis to numerical values
    diagnosis_map = {'Benign': 0, 'Malign': 1}
    df_diagnosis['Diagnosis_value'] = df_diagnosis['Diagnosis'].map(diagnosis_map)

    # Combine with original metadata (if needed for other attributes)
    max_voting = df_diagnosis[['patient_id', 'nodule_id', 'Diagnosis', 'Diagnosis_value']]
    print("Diagnosis assigned using max-voting:")
    print(max_voting.head())

    ### === EXPORT VOIs BASED ON METADATA ===
    print("Saving VOIs to disk...")

    # Ensure image_files_list and nodule_mask_files_list are correctly populated from updated paths
    # These lists are now global from the "Initial NIFTI FILE SCAN" section
    
    for idx, row in max_voting.iterrows():
        patient_id = row['patient_id']
        nodule_id = str(row['nodule_id'])

        # Find corresponding mask file based on updated nodule_mask_files_list
        mask_pattern = re.compile(rf"{patient_id}_R_{nodule_id}\.(nii|nii\.gz)")
        mask_file = next((f for f in nodule_mask_files_list if mask_pattern.match(f)), None)

        if mask_file == None:
            print(f"❌ Mask not found for {patient_id}, Nodule {nodule_id}")
            continue

        mask_path = nodule_mask_folder / mask_file # Use updated nodule_mask_folder
        
        # Find corresponding image file based on updated image_files_list
        image_file = next((f for f in image_files_list if f.startswith(patient_id)), None)
        if image_file == None:
            print(f"❌ CT image not found for {patient_id}")
            continue

        image_path = image_folder / image_file # Use updated image_folder

        try:
            image_data, image_affine = readNifty(image_path)
            mask_data, _ = readNifty(mask_path)

            # Apply lung segmentation (if applicable, typically on full CT, then nodule extraction)
            # lung_mask = lung_segmentation(image_data) # This would apply to the full image.
            # If `image_data` is already a cropped VOI, apply segmentation as needed.

            save_voi(image_data, mask_data, patient_id, nodule_id, voi_output_folder, image_affine)

        except Exception as e:
            print(f"Error processing {patient_id}, Nodule {nodule_id}: {e}")
        else:
            print("Skipping VOI processing as metadata DataFrame is empty.")

# 1. Use a classic standard pipeline over intensity volumes.
print(df_clean.iloc[0][0])
from BasicSegmentation import BasicSegmentation
data_folder = r'C:\Users\sansg\OneDrive\Escriptori\Assignment 2'
segmenter = BasicSegmentation(df_clean.iloc[0][0], data_folder=voi_base_dir)

segmenter.visualize_original()
segmenter.preprocess()
segmenter.binarize()
# Create the binary mask
binary_mask = segmenter.binarize()
plt.imshow(binary_mask[:, :, 0], cmap='gray')
plt.title('Canal 0')
plt.show()

segmenter.visualize_morphology_slice(slice_idx=5, se_size=3)


segmenter.plot_histogram()
segmenter.visualize_segmentation()
segmenter.postprocess()

# Apply segmentation
pred_mask = ct_image.iloc[0]
pred_mask = segmenter.binarize()
print(pred_mask[0])
gt_mask = ct_data.iloc[0]
print("pred_mask shape:", np.asarray(pred_mask).shape)
print("gt_mask shape:", np.asarray(gt_mask).shape)


# --- New Function for Conceptual Radiomic Feature Visualization ---
def generate_conceptual_radiomic_viz():
    """
    Generates a conceptual visualization of a segmented nodule VOI
    and a magnified pixel grid to illustrate GLCM feature quantification.
    """

    # (a) Simulate a Segmented Nodule VOI (Simplified 2D representation)
    # This is a conceptual image of a nodule with some internal texture variation.
    # In a real scenario, this would be your `binary_mask` from segmentation,
    # then used to extract intensities from the original CT image.
    nodule_voi = np.zeros((50, 50), dtype=int)
    # Create a roughly circular "nodule"
    center_x, center_y = 25, 25
    radius = 18
    for i in range(nodule_voi.shape[0]):
        for j in range(nodule_voi.shape[1]):
            if (i - center_x)**2 + (j - center_y)**2 < radius**2:
                # Assign some varying intensity values to simulate texture
                nodule_voi[i, j] = np.random.randint(50, 200) # Simulate CT intensity range

    # Add some specific patterns for better conceptual GLCM illustration
    # Example: a brighter spot in the center
    nodule_voi[20:30, 20:30] = np.random.randint(180, 250, size=(10,10))
    # Add a diagonal gradient
    for i in range(nodule_voi.shape[0]):
        for j in range(nodule_voi.shape[1]):
            if nodule_voi[i,j] > 0: # Only within the nodule
                nodule_voi[i, j] += int((i + j) / 2 * 1.5) % 30


    # (b) Magnified view showing pixel intensity relationships for GLCM
    # Select a small, representative region from the simulated nodule
    magnified_region = nodule_voi[22:27, 22:27] # Example 5x5 pixel area

    # --- Plotting ---
    plt.figure(figsize=(14, 7))

    # Subplot 1: Segmented Nodule VOI
    ax1 = plt.subplot(1, 2, 1)
    # Use a colormap suitable for intensity, 'viridis' or 'gray' are good options
    im1 = ax1.imshow(nodule_voi, cmap='gray', origin='lower')
    ax1.set_title('(a) Segmented Nodule VOI (Conceptual)', fontsize=14)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Intensity Value (HU)')


    # Subplot 2: Magnified Pixel Grid for GLCM
    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(magnified_region, cmap='gray', origin='lower')
    ax2.set_title('(b) Magnified Pixel Grid for GLCM', fontsize=14)
    ax2.set_xticks(np.arange(-.5, magnified_region.shape[1], 1), minor=True)
    ax2.set_yticks(np.arange(-.5, magnified_region.shape[0], 1), minor=True)
    ax2.grid(which='minor', color='red', linestyle='-', linewidth=0.5)
    ax2.tick_params(which='minor', size=0) # Hide minor tick marks but keep grid
    ax2.set_xticks([]) # Hide major ticks
    ax2.set_yticks([]) # Hide major ticks


    # Add pixel values as text annotations
    for i in range(magnified_region.shape[0]):
        for j in range(magnified_region.shape[1]):
            text_color = 'red' if magnified_region[i, j] < 128 else 'blue' # Contrast text color
            ax2.text(j, i, magnified_region[i, j], ha='center', va='center',
                     color=text_color, fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout to make space for text
    plt.show()

    print("\n--- Conceptual Explanation for Figure 2a ---")
    print("Figure 2a conceptually illustrates how radiomic features, specifically GLCM, are derived.")
    print("Part (a) shows a simulated 2D segmented nodule (Volume of Interest or VOI). This represents the output of the segmentation step, where the nodule has been isolated from the surrounding CT scan data.")
    print("\nPart (b) is a magnified view of a small pixel grid from within this segmented nodule. Each number in the grid represents the intensity value (e.g., in Hounsfield Units) of that pixel.")
    print("\nGLCM features are computed by analyzing the co-occurrence of pixel intensity values at various distances and angles. For example, to calculate a feature like 'Contrast':")
    print("- The algorithm looks at pairs of pixels (e.g., a pixel and its neighbor one unit to the right, or one unit diagonally).")
    print("- It counts how often a pixel with intensity 'X' is found adjacent to a pixel with intensity 'Y'.")
    print("- For 'Contrast', it would sum the squared differences of these intensity pairs across the entire region, weighted by their frequency.")
    print("\nBy quantifying these relationships, GLCM features (like Contrast, Homogeneity, Energy, Dissimilarity, Correlation) provide numerical descriptors of the texture (e.g., smoothness, coarseness, regularity) within the nodule. These numerical features are then fed into a machine learning model for classification.")
    print("This visualization helps to understand that radiomics transforms visual texture into quantifiable data points.")

# --- End of New Function ---

# --- Call the new conceptual radiomic visualization function ---
print("\n--- Generating Conceptual Radiomic Feature Visualization ---")
generate_conceptual_radiomic_viz()


# ==============================================================================
# --- MILESTONE 2: RADIOMICS FEATURE EXTRACTION ---
# Note: The provided Milestone2_part2.py file was empty.
# If you have specific code for radiomics feature extraction, you would add it here.
# A common library for this is PyRadiomics.
# ==============================================================================
# import SimpleITK as sitk
# from radiomics import featureextractor

# Example placeholder for radiomics feature extraction if you had code:
# def extract_radiomics_features(image_path, mask_path):
#     extractor = featureextractor.RadiomicsFeatureExtractor()
#     image = sitk.ReadImage(str(image_path))
#     mask = sitk.ReadImage(str(mask_path))
#     result = extractor.execute(image, mask)
#     return result

print("\n--- Milestone 2: Radiomics Feature Extraction ---")
print("Note: The provided `Milestone2_part2.py` was an empty file.")
print("If you have code for radiomics feature extraction (e.g., using PyRadiomics), it would go here.")
print("It would typically take the exported VOIs from Milestone 1 as input.")


# ==============================================================================
# --- MILESTONE 2: DATA EXPLORATION AND CLUSTERING ---
# ==============================================================================
print("\n--- Milestone 2: Data Exploration and Clustering ---")
print("Loading data for exploration...")
metadata_file_path = root_folder / "MetadatabyNoduleMaxVoting.xlsx"

df = pd.read_excel(metadata_file_path)
print("✅ Data imported! Printing the first 5 elements")
print(df.head())
    
# Data Preprocessing for Clustering
categorical_cols_to_encode = [
   'Malignancy', 'Calcification', 'InternalStructure', 'Lobulation',
   'Margin', 'Sphericity', 'Spiculation', 'Subtlety', 'Texture'
]

# Convert specified columns to 'category' dtype if they exist and are 'object' or numerical
for col in categorical_cols_to_encode:
    if col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype('category')

df_encoded = pd.get_dummies(df, columns=categorical_cols_to_encode, prefix=categorical_cols_to_encode)

features_for_clustering = [col for col in df_encoded.columns if
                           col not in ['patient_id', 'nodule_id', 'seriesuid', 'Diagnosis',
                                           'Diagnosis_value', 'Malignancy_value', 'len_mal_details'] and
                               not col.startswith(('coord', 'bbox'))]

if 'diameter_mm' in df_encoded.columns and 'diameter_mm' not in features_for_clustering:
        features_for_clustering.append('diameter_mm')

existing_features = [f for f in features_for_clustering if f in df_encoded.columns]
X_clustering = df_encoded[existing_features].copy()

# Handle missing values in X_clustering (if any) before scaling
if X_clustering.isnull().sum().sum() > 0:
        print("\n⚠️ Warning: Missing values detected in X_clustering. Filling with median/mode.")
        for col in X_clustering.columns:
            if X_clustering[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X_clustering[col]):
                    X_clustering[col] = X_clustering[col].fillna(X_clustering[col].median())
                else:
                    X_clustering[col] = X_clustering[col].fillna(X_clustering[col].mode()[0])

print(f"Features for clustering: {existing_features}")
print(X_clustering.head())

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clustering)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_clustering.columns)
print("Features scaled.")

# --- Prepare df_oh and df_filtered2 for refactored clustering function ---
df_oh = X_scaled_df # X_scaled_df is the one-hot encoded, scaled data

# Create df_filtered2 (numerical features from original df) for comparison
df_filtered2 = df.select_dtypes(include=np.number).copy()

# Drop identifier columns if they are numerical but not features
id_cols_to_drop_from_num = ['patient_id', 'nodule_id', 'seriesuid']
df_filtered2 = df_filtered2.drop(columns=[col for col in id_cols_to_drop_from_num if col in df_filtered2.columns and col in df_filtered2.columns], errors='ignore')

# Handle missing values in df_filtered2
if df_filtered2.isnull().sum().sum() > 0:
    print("\n⚠️ Warning: Missing values detected in df_filtered2. Filling with median.")
    for col in df_filtered2.columns:
        if df_filtered2[col].isnull().any():
            df_filtered2[col] = df_filtered2[col].fillna(df_filtered2[col].median())

# Scale df_filtered2 separately for clustering with numerical features
scaler_numerical = StandardScaler()
df_filtered2_scaled = scaler_numerical.fit_transform(df_filtered2)
df_filtered2_scaled_df = pd.DataFrame(df_filtered2_scaled, columns=df_filtered2.columns)

print("\nShape of df_oh (one-hot encoded & scaled data):", df_oh.shape)
print("Shape of df_filtered2 (original numerical data):", df_filtered2.shape)
print("Shape of df_filtered2_scaled_df (scaled numerical data):", df_filtered2_scaled_df.shape)

# --- Refactored Clustering Function ---
def perform_and_evaluate_clustering(data_df, data_name, silhouette_metric='euclidean'):
    """
       Performs clustering with various algorithms and evaluates them using Silhouette Score.

        Args:
            data_df (pd.DataFrame): The DataFrame to cluster.
            data_name (str): A descriptive name for the data (e.g., "ONE HOT ENCODED DATA").
            silhouette_metric (str): The distance metric for silhouette_score.
        """
    print(f"\n--- Clustering on {data_name} ---")
    check_clusters = [2, 5]
    all_cluster_labels = {}

    for c_number in check_clusters:
        print(f"\nEvaluating with {c_number} clusters:")
        models = {
            "K-Means": KMeans(n_clusters=c_number, random_state=42, n_init=10),
            "Hierarchical (Ward)": AgglomerativeClustering(n_clusters=c_number, linkage='ward'),
            "Hierarchical (Complete)": AgglomerativeClustering(n_clusters=c_number, linkage='complete'),
            "Hierarchical (Average)": AgglomerativeClustering(n_clusters=c_number, linkage='average'),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=5) # Consider tuning eps and min_samples for DBSCAN
            }

        cluster_labels = {}
        silhouette_scores = {}

        for name, model in models.items():
            labels = model.fit_predict(data_df)
            cluster_labels[name] = labels

        # Evaluate Silhouette Score, handling cases where it might not be applicable/valid
        if name != "DBSCAN":
            try:
                        # Silhouette score requires at least 2 clusters and samples > 1
                        if len(np.unique(labels)) > 1 and len(labels) > 1:
                            silhouette_scores[name] = silhouette_score(data_df, labels, metric=silhouette_metric)
                        else:
                            silhouette_scores[name] = float('-inf') # Indicate invalid score
                            print(f"Warning: {name} with {c_number} clusters resulted in less than 2 unique clusters or insufficient samples. Cannot compute Silhouette Score.")
            except ValueError as e:
                        silhouette_scores[name] = float('-inf')
                        print(f"Error computing Silhouette Score for {name} with {c_number} clusters: {e}")
    else: # Special handling for DBSCAN
             # Exclude noise points (-1) when counting clusters for evaluation
                    n_clusters_dbscan = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters_dbscan >= 2 and len(labels) > 1: # Require at least 2 clusters and samples
                        try:
                            # For simplicity, calculating on all points including noise:
                            silhouette_scores[name] = silhouette_score(data_df, labels, metric=silhouette_metric)
                        except ValueError as e:
                            silhouette_scores[name] = float('-inf')
                            print(f"Error computing Silhouette Score for DBSCAN: {e}")
                    else:
                        silhouette_scores[name] = float('-inf')
                        print(f"Warning: DBSCAN resulted in less than 2 meaningful clusters or insufficient samples. Cannot compute Silhouette Score.")

# Print results for the current number of clusters
    print(f"Silhouette Scores ({data_name}, metric='{silhouette_metric}') for {c_number} clusters (Higher = Better Clustering):")
    for name, score in silhouette_scores.items():
        if score != float('-inf'):
            print(f"  {name}: {score:.4f}")
        else:
            print(f"  {name}: N/A (Score could not be computed)")
            all_cluster_labels[c_number] = cluster_labels
    return all_cluster_labels

# --- Perform clustering and evaluation on both datasets ---
oh_labels_array = perform_and_evaluate_clustering(df_oh, "ONE HOT ENCODED DATA (Scaled)", 'manhattan')
labels_array = perform_and_evaluate_clustering(df_filtered2_scaled_df, "NUMERICAL FEATURES (Scaled)", 'chebyshev')


# --- 3. t-SNE Visualization ---
print("\n--- Performing t-SNE Visualization ---")

# Initialize t-SNE with specified parameters
tsne = TSNE(n_components=2, perplexity=70, random_state=41)
# Fit t-SNE to the scaled data and transform it to 2 dimensions
tsne_default = tsne.fit_transform(X_scaled)

# Calculate Trustworthiness
# Trustworthiness measures how well the neighborhood relationships are preserved
# (values closer to 1 are better)
Trst = trustworthiness(X_scaled, tsne_default)
print(f"Trustworthiness of t-SNE embedding: {Trst:.4f}")

# Convert t-SNE results into a DataFrame for easier plotting
tsne_df = pd.DataFrame(tsne_default, columns=['TSNE1', 'TSNE2'])

# Plot t-SNE with color coding by 'Diagnosis' (typically 2-class: Benign/Malign)
if 'Diagnosis' in df.columns:
    # Assign 'Diagnosis' as the label. reset_index(drop=True) is crucial
    # to ensure alignment with tsne_df, especially if df underwent prior filtering.
    tsne_df['Label'] = df['Diagnosis'].reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=tsne_df['TSNE1'], y=tsne_df['TSNE2'], hue=tsne_df['Label'], palette='bright', alpha=0.7)

    plt.title(f"t-SNE Visualization by Diagnosis", fontsize=16)
    plt.xlabel("TSNE1", fontweight='bold')
    plt.ylabel("TSNE2", fontweight='bold')
    plt.legend(title="Diagnosis Label")
    plt.grid(True, linestyle='--', alpha=0.6) # Add a grid for better readability
    plt.show()
    print("t-SNE visualization by Diagnosis generated.")
else:
    print("Warning: 'Diagnosis' column not found in the original DataFrame for t-SNE plotting.")

# Plot t-SNE with color coding by 'Malignancy' (typically 5-class)
if 'Malignancy' in df.columns:
    # Assign 'Malignancy' as the label, ensuring index alignment
    tsne_df['Label'] = df['Malignancy'].reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=tsne_df['TSNE1'], y=tsne_df['TSNE2'], hue=tsne_df['Label'], palette='bright', alpha=0.7)

    plt.title(f"t-SNE Visualization by Malignancy", fontsize=16)
    plt.xlabel("TSNE1", fontweight='bold')
    plt.ylabel("TSNE2", fontweight='bold')
    plt.legend(title="Malignancy Label")
    plt.grid(True, linestyle='--', alpha=0.6) # Add a grid for better readability
    plt.show()
    print("t-SNE visualization by Malignancy generated.")
else:
    print("Warning: 'Malignancy' column not found in the original DataFrame for t-SNE plotting.")


# Hierarchical Clustering (Dendrogram)
print("\nGenerating Hierarchical Clustering Dendrograms...")

# Ensure 'patient_id' and 'nodule_id' exist in df_encoded for dendrogram labels
# Use original df columns as df_encoded might have these as dummies or dropped them
if 'patient_id' in df.columns and 'nodule_id' in df.columns:
    dendrogram_labels = df['patient_id'].astype(str) + '_N' + df['nodule_id'].astype(str)
else:
    print("Warning: 'patient_id' or 'nodule_id' not found in original DataFrame for dendrogram labels. Using default indices.")
    dendrogram_labels = None

def plot_dendrogram(X_data, labels, linkage_method, metric="euclidean"):
        """
        Generates and displays a hierarchical clustering dendrogram for a given linkage method.

        Args:
            X_data (np.ndarray): The scaled feature data.
            labels (pd.Series or None): Labels for the dendrogram leaves.
            linkage_method (str): The linkage algorithm to use (e.g., 'ward', 'complete', 'average', 'single').
            metric (str): The distance metric to use.
        """
        plt.figure(figsize=(20, 10))
        # Ensure X_data is a NumPy array for scipy.cluster.hierarchy functions
        if isinstance(X_data, pd.DataFrame):
            X_data_np = X_data.values
        else:
            X_data_np = X_data

        linked = linkage(X_data_np, method=linkage_method, metric=metric)
        dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True,
                   labels=labels.values if labels is not None else None)
        plt.title(f'Hierarchical Clustering Dendrogram ({linkage_method.capitalize()} Linkage)')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        print(f"Dendrogram ({linkage_method.capitalize()} Linkage) generated.")

# Generate dendrogram for 'ward' linkage (as in previous code)
# Using X_scaled for dendrograms as it's the preprocessed feature set
plot_dendrogram(X_scaled, dendrogram_labels, 'ward')

# Generate dendrogram for 'complete' linkage
plot_dendrogram(X_scaled, dendrogram_labels, 'complete')

# Generate dendrogram for 'average' linkage
plot_dendrogram(X_scaled, dendrogram_labels, 'average')

# Generate dendrogram for 'single' linkage
plot_dendrogram(X_scaled, dendrogram_labels, 'single')

# K-Means Clustering (Elbow Method and Silhouette Score Plot for k)
print("\nPerforming K-Means Clustering (Elbow and Silhouette Method)...")
silhouette_scores_kmeans = []
wcss = [] # Within-cluster sum of squares for Elbow Method
k_range = range(2, 11) # Adjusted range for K-Means

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    cluster_labels = kmeans.labels_
    wcss.append(kmeans.inertia_) # Sum of squared distances of samples to their closest cluster center

    if len(set(cluster_labels)) > 1 and len(cluster_labels) > 1:
        score = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores_kmeans.append(score)
    else:
        silhouette_scores_kmeans.append(0) # or np.nan, or continue

# Plot Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='-')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.xticks(list(k_range))
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
print("Elbow Method plot generated.")

# Plot Silhouette Score for K-Means
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores_kmeans, marker='o')
plt.title('Silhouette Score for K-Means Clustering')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(list(k_range))
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
print("Silhouette Score plot for K-Means generated.")

# Example: Final K-Means clustering with an 'optimal' k (e.g., from plots)
# The user can visually determine the optimal k from the plots
optimal_k = 3 # Placeholder, choose based on Elbow/Silhouette plots
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)
print(f"K-Means clustering performed with k={optimal_k}. Cluster distribution:")
print(df['kmeans_cluster'].value_counts())


# Correlation Analysis
print("\n--- Performing Correlation Analysis ---")
# Ensure relevant columns are numerical for correlation calculation
# Using df_encoded which contains all features including one-hot encoded ones
if 'Diagnosis_value' in df_encoded.columns and 'Malignancy_value' in df_encoded.columns:
# Select only numerical columns from df_encoded for correlation
    numerical_features_for_corr = df_encoded.select_dtypes(include=np.number)
    corr_matrix = numerical_features_for_corr.corr()

    diagnosis_corr = corr_matrix['Diagnosis_value'].sort_values(ascending=False)
    print("\nCorrelation with Diagnosis_value:")
    print(diagnosis_corr)

    malig_corr = corr_matrix['Malignancy_value'].sort_values(ascending=False)
    print("\nCorrelation with Malignancy_value:")
    print(malig_corr)

    # Set figure size
    plt.figure(figsize=(12, 10)) # Adjusted size for better readability
    # Plot the heatmap
    sns.heatmap(corr_matrix,
                    cmap='coolwarm',
                    annot=False, # Set to False for very large matrices, or select top N for display
                    fmt=".2f",
                    square=True,
                    cbar_kws={"shrink": 0.8},
                    linewidths=0.5)
    plt.title("Correlation Matrix Heatmap (Numerical & One-Hot Encoded Features)")
    plt.tight_layout()
    plt.show()

else:
    print("Diagnosis_value or Malignancy_value not found in df_encoded for correlation analysis.")
    print("\nConclusion on data exploration:")
    print("The diameter is often a significant factor. Subtlety, Spiculation, and Lobulation also play important roles.")

   


# ==============================================================================
# --- MILESTONE 3: TRAINING A CLASSIFIER ON GLCM FEATURES ---
# ==============================================================================

def run_classification_pipeline(X, y, groups, feature_set_name):
    """
    Runs the classification pipeline (scaling, randomized search, cross-validation)
    for a given feature set.
    """
    print(f"\n--- Running Classification for {feature_set_name} ---")

    # Data Preprocessing (Standardization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"{feature_set_name} scaled.")

    # Stratified Group K-Fold for Cross-Validation
    if groups is not None and len(np.unique(groups)) > 1: # Ensure groups are valid for StratifiedGroupKFold
        sgkf_local = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        print(f"Groups for cross-validation based on 'seriesuid' for {feature_set_name}: {len(np.unique(groups))} unique groups.")
    else:
        sgkf_local = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print(f"Using StratifiedKFold for {feature_set_name} as 'seriesuid' groups are not applicable or insufficient.")
        groups = None # Ensure groups is None if not used by StratifiedGroupKFold

    # Model: Support Vector Classifier
    svc = SVC(random_state=42)

    # Randomized Search for Hyperparameter Tuning
    print(f"\nPerforming Randomized Search for SVC hyperparameters for {feature_set_name}...")
    param_distributions = {
        'C': loguniform(1e-1, 1e2),
        'gamma': loguniform(1e-4, 1e-1),
        'kernel': ['rbf', 'linear']
    }

    random_search = RandomizedSearchCV(svc, param_distributions, n_iter=100, cv=sgkf_local,
                                       scoring='accuracy', random_state=42, n_jobs=-1, verbose=0) # verbose=0 to reduce output

    random_search.fit(X_scaled, y, groups=groups if groups is not None else None)

    print(f"Best parameters for {feature_set_name} from Randomized Search: {random_search.best_params_}")
    print(f"Best cross-validation accuracy for {feature_set_name} from Randomized Search: {random_search.best_score_:.4f}")

    best_svc_random = random_search.best_estimator_

    # Cross-validation with best estimator
    print(f"\nPerforming Cross-Validation with the best SVC estimator for {feature_set_name}...")
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='binary'),
        'recall': make_scorer(recall_score, average='binary'),
        'f1': make_scorer(f1_score, average='binary')
    }

    results = cross_validate(best_svc_random, X_scaled, y, cv=sgkf_local,
                             scoring=scoring, n_jobs=-1, return_train_score=False,
                             groups=groups if groups is not None else None)

    print(f"\nCross-Validation Results for {feature_set_name}:")
    for metric, values in results.items():
        if metric.startswith('test_'):
            print(f"{metric.replace('test_', '').capitalize()} (Mean +/- Std): {np.mean(values):.4f} +/- {np.std(values):.4f}")

    cv_scores = results['test_accuracy']
    mean_score = np.mean(cv_scores)
    std_error = st.sem(cv_scores)
    confidence = 0.95
    ci_lower, ci_upper = st.t.interval(confidence, len(cv_scores)-1, loc=mean_score, scale=std_error)

    print(f"Mean Score for {feature_set_name}: {mean_score:.4f}")
    print(f"95% Confidence Interval for {feature_set_name}: ({ci_lower:.4f}, {ci_upper:.4f})")
    print(f"The metrics for {feature_set_name} are quite similar to the random search.")


print("\n--- Milestone 3: Training and Comparison of Classifiers ---")

# --- Process GLCM Features ---
print("\n--- Processing GLCM Features ---")
print("Loading GLCM data from 'slice_glcm1d.npz'...")
try:
    data_glcm = np.load("slice_glcm1d.npz", allow_pickle=True)
    df_features_glcm = pd.DataFrame(data_glcm['slice_features'])
    df_meta_glcm = pd.DataFrame(data_glcm['slice_meta'])

    if 'label' in df_meta_glcm.columns:
        y_glcm = df_meta_glcm['label'].values
    else:
        print("Warning: 'label' column not found in GLCM metadata. Assuming all labels are 0 for now.")
        y_glcm = np.zeros(len(df_features_glcm))

    X_glcm = df_features_glcm.values
    groups_glcm = df_meta_glcm['seriesuid'].values if 'seriesuid' in df_meta_glcm.columns else None

    run_classification_pipeline(X_glcm, y_glcm, groups_glcm, "GLCM Features")

except FileNotFoundError:
    print("Error: 'slice_glcm1d.npz' not found. Cannot run GLCM pipeline.")
except Exception as e:
    print(f"An error occurred during GLCM pipeline: {e}")

# --- Process Pre-trained Network Features (Placeholder) ---
print("\n--- Attempting to process Pre-trained Network Features ---")
print("Note: This section requires an external file containing pre-extracted features from a deep learning model.")
print("If you have extracted these features, place the 'pre_trained_network_features.npz' file in the root_folder.")

pre_trained_features_file = root_folder / "pre_trained_network_features.npz" # Assuming it's in the root_folder
try:
    print(f"Attempting to load pre-trained network features from '{pre_trained_features_file}'...")
    data_pretrained = np.load(pre_trained_features_file, allow_pickle=True)
    
    # Assuming the .npz file contains 'features' for X and 'meta' for metadata (including 'label' and 'seriesuid')
    df_features_pretrained = pd.DataFrame(data_pretrained['features'])
    df_meta_pretrained = pd.DataFrame(data_pretrained['meta'])

    if 'label' in df_meta_pretrained.columns:
        y_pretrained = df_meta_pretrained['label'].values
    else:
        print("Warning: 'label' column not found in pre-trained features metadata. Assuming all labels are 0 for now.")
        y_pretrained = np.zeros(len(df_features_pretrained))

    X_pretrained = df_features_pretrained.values
    groups_pretrained = df_meta_pretrained['seriesuid'].values if 'seriesuid' in df_meta_pretrained.columns else None

    # Ensure y and groups have consistent lengths and are not empty
    if len(X_pretrained) == 0 or len(y_pretrained) == 0 or len(X_pretrained) != len(y_pretrained):
        print("Error: Mismatch or empty data for pre-trained features and labels. Skipping comparison.")
    elif groups_pretrained is not None and len(X_pretrained) != len(groups_pretrained):
        print("Error: Mismatch in number of samples between pre-trained features and groups. Skipping comparison.")
    else:
        run_classification_pipeline(X_pretrained, y_pretrained, groups_pretrained, "Pre-trained Network Features")

except FileNotFoundError:
    print(f"Info: '{pre_trained_features_file}' not found. Skipping pre-trained network feature comparison.")
except Exception as e:
    print(f"An error occurred during pre-trained network feature pipeline: {e}")

print("\n--- End of the Script ---")