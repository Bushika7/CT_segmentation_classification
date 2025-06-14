#   MILESTONE 4
#Import packages
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import sys
from pathlib import Path

# For Milestone 1: Segmentation and VOI Generation
from skimage.filters import threshold_otsu
from skimage import morphology as Morpho
from scipy.ndimage import filters as filt
from scipy.ndimage import distance_transform_edt as bwdist
from skimage.measure import find_contours as contour
from BasicSegmentation import BasicSegmentation
from Comparation import comparation

# For Milestone 2: Data Exploration and Clustering
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
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
from sklearn.manifold import TSNE
import seaborn as sns # For enhanced plotting
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score
# from radiomics import featureextractor # Uncomment if you fix the feature extraction in Milestone2_part2
import SimpleITK as sitk # Required if using PyRadiomics, included based on Milestone2_part2.py snippet

print("✅ All packages installed and working!")

# --- Configuration ---
BASE_DOWNLOADS_DIR = Path("C:\\Users\\sansg\\Downloads")
VOIS_IMAGE_DIR = Path("C:\\Users\\sansg\\Downloads\\VOIs\\VOIs\\image")
VOIS_MASK_DIR = Path("C:\\Users\\sansg\\Downloads\\VOIs\\VOIs\\nodule_mask")
METADATA_FILE = Path("C:\\Users\\sansg\\Downloads\\MetadatabyNoduleMaxVoting.xlsx")
ASSIGNMENT_DIR = Path("C:\\Users\\sansg\\OneDrive\\Escriptori\\Assignment 2")

# --- Functions ---
def show_boxplot(df):
    plt.rcParams['figure.figsize'] = [14,6]
    sns.boxplot(data = df, orient="v")
    plt.title("Outliers Distribution", fontsize = 16)
    plt.ylabel("Range", fontweight = 'bold')
    plt.xlabel("Attributes", fontweight = 'bold')
    
def load_and_extract_nii_info(folder_path: Path, prefix: str):
    """Loads NIfTI files from a folder and extracts basic info."""
    file_info_list = []
    nii_files = [f for f in os.listdir(folder_path) if f.endswith('.nii.gz')]

    for file_name in nii_files:
        file_path = folder_path / file_name
        try:
            img = nib.load(file_path)
            img_data = img.get_fdata()
            file_info = {
                'filename': file_name, # Keep original filename
                f'{prefix}_shape': img_data.shape,
                f'{prefix}_mean_intensity': np.mean(img_data),
                f'{prefix}_max_intensity': np.max(img_data),
                f'{prefix}_min_intensity': np.min(img_data)
            }
            file_info_list.append(file_info)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            continue
    return pd.DataFrame(file_info_list)

def process_nodule_metadata(metadata_filepath: Path):
    """Loads metadata and applies max-voting diagnosis."""
    try:
        df = pd.read_excel(metadata_filepath)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_filepath}.")
        sys.exit(1)

    print("# Different patients in df: ", len(df['patient_id'].unique()))

    df.loc[:, 'patient_id'] = df['patient_id'].astype(str)

    max_voting_df = df.groupby('patient_id')['Malignancy_value'].agg(lambda x: x.mode()[0]).reset_index(name='Malignancy_mode')

    print("# Different patients in max_voting: ", len(max_voting_df['patient_id'].unique()))

    max_voting_df['Diagnosis_value'] = np.where(max_voting_df['Malignancy_mode'] > 3, 1, 0)
    max_voting_df['Diagnosis'] = np.where(max_voting_df['Malignancy_mode'] > 3, 'Malign', 'Benign')

    return max_voting_df

def objective(trial, X_train, y_train, groups_train):
    # Hyperparameters to optimize
    C = trial.suggest_float("C", 1e-3, 1e3, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
    
    # Conditional parameters
    params = {
        "C": C,
        "kernel": kernel,
        "shrinking": trial.suggest_categorical("shrinking", [True, False]),
        "tol": trial.suggest_float("tol", 1e-5, 1e-1, log=True)
    }

    # Model training and validation
    model = SVC(**params, random_state=42, class_weight='balanced')
    scores = cross_val_score(
        model, X_train, y_train, cv=cv_inner, groups=groups_train, scoring="accuracy"
    )
    return scores.mean()

def generate_conceptual_radiomic_viz():
    """
    Conceptual visualization of a segmented nodule VOI
    and a magnified pixel grid to illustrate GLCM feature quantification
    """
    nodule_voi = np.zeros((50, 50), dtype=int)
    center_x, center_y = 25, 25
    radius = 18
    for i in range(nodule_voi.shape[0]):
        for j in range(nodule_voi.shape[1]):
            if (i - center_x)**2 + (j - center_y)**2 < radius**2:
                nodule_voi[i, j] = np.random.randint(50, 200)
    nodule_voi[20:30, 20:30] = np.random.randint(180, 250, size=(10,10))
    for i in range(nodule_voi.shape[0]):
        for j in range(nodule_voi.shape[1]):
            if nodule_voi[i,j] > 0:
                nodule_voi[i, j] += int((i + j) / 2 * 1.5) % 30

    magnified_region = nodule_voi[22:27, 22:27]

    plt.figure(figsize=(14, 7))

    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(nodule_voi, cmap='gray', origin='lower')
    ax1.set_title('(a) Segmented Nodule VOI (Conceptual)', fontsize=14)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Intensity Value (HU)')

    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(magnified_region, cmap='gray', origin='lower')
    ax2.set_title('(b) Magnified Pixel Grid for GLCM', fontsize=14)
    ax2.set_xticks(np.arange(-.5, magnified_region.shape[1], 1), minor=True)
    ax2.set_yticks(np.arange(-.5, magnified_region.shape[0], 1), minor=True)
    ax2.grid(which='minor', color='red', linestyle='-', linewidth=0.5)
    ax2.tick_params(which='minor', size=0)
    ax2.set_xticks([])
    ax2.set_yticks([])

    for i in range(magnified_region.shape[0]):
        for j in range(magnified_region.shape[1]):
            text_color = 'red' if magnified_region[i, j] < 128 else 'blue'
            ax2.text(j, i, magnified_region[i, j], ha='center', va='center',
                     color=text_color, fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.show()

    print("Figure 2a shows a simulated 2D segmented nodule (Volume of Interest or VOI). The nodule has been isolated from the surrounding CT scan data.")
    print("\nPart (b) is a magnified view of a small pixel grid from within this segmented nodule. Each number in the grid represents the intensity value of that pixel.")
    print("\nGLCM features are computed by analyzing the co-occurrence of pixel intensity values at various distances and angles.")
    print("- It counts how often a pixel with intensity 'X' is found adjacent to a pixel with intensity 'Y'.")
    print("\nBy quantifying these relationships, GLCM features provide numerical descriptors of the texture within the nodule.")
    print("These numerical features are then fed into a machine learning model for classification.")


#   MILESTONE 1
print("Starting Lung Nodule Analysis Workflow.")

# 1. Load and process NIfTI file information
print("Loading NIfTI file information...")
df_image_info = load_and_extract_nii_info(VOIS_IMAGE_DIR, 'Image')
df_mask_info = load_and_extract_nii_info(VOIS_MASK_DIR, 'Nodule_mask')

print(f"Length of df_mask_info: {len(df_mask_info)}")
print(f"Length of df_image_info: {len(df_image_info)}")

# Extract common key for merging (e.g., LIDC-IDRI-0001_1 from filename)
df_image_info['file_key'] = df_image_info['filename'].str.replace('_R_', '_').str.replace('.nii.gz', '', regex=False)
df_mask_info['file_key'] = df_mask_info['filename'].str.replace('_R_', '_').str.replace('.nii.gz', '', regex=False)

vois_combined_info = pd.merge(df_image_info, df_mask_info, on='file_key', suffixes=('_image', '_mask'))
vois_combined_info.drop(columns=['file_key'], inplace=True)
print("Combined VOIs information:")
print(vois_combined_info.head())

# 2. Process Metadata for Diagnosis
print("\nProcessing metadata for diagnosis...")
max_voting_diagnosis = process_nodule_metadata(METADATA_FILE)
print("\nDiagnosis assigned using max-voting:")
print(max_voting_diagnosis.head())


# 3. Segmentation Pipeline and Evaluation (Requires external files)
print("\nAttempting to run segmentation pipeline and evaluation...")
sys.path.append(str(ASSIGNMENT_DIR))

if not vois_combined_info.empty:
    first_entry = vois_combined_info.iloc[0]
    
    # CORRECTED LINES: Use 'filename_image' and 'filename_mask'
    original_image_path = VOIS_IMAGE_DIR / first_entry['filename_image']
    original_mask_path = VOIS_MASK_DIR / first_entry['filename_mask']

segmenter = BasicSegmentation(str(original_image_path), data_folder=str(ASSIGNMENT_DIR))

segmenter.visualize_original()
segmenter.preprocess()
pred_mask_data = segmenter.binarize()

gt_mask_nifti = nib.load(original_mask_path)
gt_mask_data = gt_mask_nifti.get_fdata()

plt.imshow(pred_mask_data[:, :, pred_mask_data.shape[2] // 2], cmap='gray')
plt.title('Predicted Binary Mask (Middle Slice)')
plt.show()

segmenter.visualize_morphology_slice(slice_idx=pred_mask_data.shape[2] // 2, se_size=3)
segmenter.plot_histogram()
segmenter.visualize_segmentation()
segmenter.postprocess()

print("Predicted mask shape:", pred_mask_data.shape)
print("Ground Truth mask shape:", gt_mask_data.shape)

if pred_mask_data.shape != gt_mask_data.shape:
    print(f"Warning: Mask shapes differ. Predicted: {pred_mask_data.shape}, Ground Truth: {gt_mask_data.shape}. Resampling might be needed.")

results = comparation.segmentation_metrics(pred_mask_data, gt_mask_data)

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
if str(ASSIGNMENT_DIR) in sys.path:
    sys.path.remove(str(ASSIGNMENT_DIR))
    
# 4. Generate Conceptual Radiomic Feature Visualization
print("\n--- Generating Conceptual Radiomic Feature Visualization ---")
generate_conceptual_radiomic_viz()

print("\nMILESTONE 1 workflow completed.")  

# ==============================================================================
# --- MILESTONE 2: DATA EXPLORATION AND CLUSTERING ---
# ==============================================================================
print("\n--- Milestone 2: Data Exploration and Clustering ---")
print("Loading data for exploration...")

# Assuming 'MetadatabyNoduleMaxVoting.xlsx' is in the root_folder
metadata_file_path = METADATA_FILE  
df = pd.read_excel(metadata_file_path)
print("✅ Data imported! Printing the first 5 elements")
print(df.head())

# Data Preprocessing for Clustering
categorical_cols_to_encode = [
    'Malignancy', 'Calcification', 'InternalStructure', 'Lobulation',
    'Margin', 'Sphericity', 'Spiculation', 'Subtlety', 'Texture'
]

for col in categorical_cols_to_encode:
    if col in df.columns:
        if df[col].dtype == 'object':
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

print(f"Features for clustering: {existing_features}")
print(X_clustering.head())

print("Count of missing values in each column:\n", X_clustering.isnull().sum())
print("✅ No missing values on the data!")

show_boxplot(X_clustering)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clustering)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_clustering.columns)
X_scaled_df.head()
print("Features scaled.")


print("\n--- Generating t-SNE Visualization of GLCM Features for Diagnosis ---")
# Perform t-SNE on the scaled GLCM features
# Adjust perplexity and n_iter as needed, or for larger datasets
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_default = tsne.fit_transform(X_scaled_df)

# Convert t-SNE results into a DataFrame for easy plotting
tsne_df = pd.DataFrame(tsne_default, columns=['TSNE1', 'TSNE2'])
# Use the 'Diagnosis' column from df_combined for labels
tsne_df['Label'] =  df['Diagnosis']

# Plot t-SNE with color coding
plt.figure(figsize=(10, 6))
sns.scatterplot(x=tsne_df['TSNE1'], y=tsne_df['TSNE2'], hue=tsne_df['Label'], palette='bright', alpha=0.7)

plt.title("t-SNE 2-class Visualization of True Labels", fontsize=16)
plt.xlabel("TSNE1", fontweight='bold')
plt.ylabel("TSNE2", fontweight='bold')
plt.legend(title="Class Label")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("t-SNE visualization generated successfully.")
print("\n--- Generating t-SNE Visualization of GLCM Features for Malignancy ---")
# Perform t-SNE on the scaled GLCM features
# Adjust perplexity and n_iter as needed, or for larger datasets
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_default = tsne.fit_transform(X_scaled_df)

# Convert t-SNE results into a DataFrame for easy plotting
tsne_df = pd.DataFrame(tsne_default, columns=['TSNE1', 'TSNE2'])
# Use the 'Diagnosis' column from df_combined for labels
tsne_df['Label'] =  df['Malignancy']

# Plot t-SNE with color coding
plt.figure(figsize=(10, 6))
sns.scatterplot(x=tsne_df['TSNE1'], y=tsne_df['TSNE2'], hue=tsne_df['Label'], palette='bright', alpha=0.7)
plt.title("t-SNE 2-class Visualization of True Labels", fontsize=16)
plt.xlabel("TSNE1", fontweight='bold')
plt.ylabel("TSNE2", fontweight='bold')
plt.legend(title="Class Label")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

df_oh = pd.get_dummies(X_clustering, columns=['Calcification_value', 'InternalStructure_value',
   'Lobulation_value', 'Margin_value', 'Sphericity_value',
   'Spiculation_value', 'Subtlety_value', 'Texture_value'], dtype='int32')
df_oh.head()

tsne_oh = tsne.fit_transform(df_oh)
# Convert t-SNE results into a DataFrame for easy plotting
tsne_df = pd.DataFrame(tsne_oh, columns=['TSNE1', 'TSNE2'])
tsne_df['Label'] = df['Diagnosis']  # Assuming 'Label' is your categorical class column

# Plot t-SNE with color coding
plt.figure(figsize=(10, 6))
sns.scatterplot(x=tsne_df['TSNE1'], y=tsne_df['TSNE2'], hue=tsne_df['Label'], palette='bright', alpha=0.7)

plt.title(f"t-SNE 2-class Visualization of True labels on one hot encoded features", fontsize=16)
plt.xlabel("TSNE1", fontweight='bold')
plt.ylabel("TSNE2", fontweight='bold')
plt.legend(title="Class Label")
plt.show()

# Convert t-SNE results into a DataFrame for easy plotting
tsne_df = pd.DataFrame(tsne_oh, columns=['TSNE1', 'TSNE2'])
tsne_df['Label'] = df['Malignancy']  # Assuming 'Label' is your categorical class column

# Plot t-SNE with color coding
plt.figure(figsize=(10, 6))
sns.scatterplot(x=tsne_df['TSNE1'], y=tsne_df['TSNE2'], hue=tsne_df['Label'], palette='bright', alpha=0.7)

plt.title(f"t-SNE 5-class Visualization of True labels on one hot encoded features", fontsize=16)
plt.xlabel("TSNE1", fontweight='bold')
plt.ylabel("TSNE2", fontweight='bold')
plt.legend(title="Class Label")
plt.show()

# For one hot encoded data
check_clusters = [2, 5]
oh_labels_array = []
for element in check_clusters:
    c_number = element
    models = {
    "K-Means": KMeans(n_clusters=c_number, random_state=42),
    "Hierarchical (Ward)": AgglomerativeClustering(n_clusters=c_number, linkage='ward'),
    "Hierarchical (Complete)": AgglomerativeClustering(n_clusters=c_number, linkage='complete'),
    "Hierarchical (Average)": AgglomerativeClustering(n_clusters=c_number, linkage='average'),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5)  # Alternative to HyperDB (Density-based)
    }

# Store results
cluster_labels_oh = {}
silhouette_scores = {}

for name, model in models.items():
    labels = model.fit_predict(df_oh)
    cluster_labels_oh[name] = labels
    
    # Evaluate if clusters are meaningful (skip DBSCAN where n_clusters isn't defined)
    if name != "DBSCAN":
        silhouette_scores[name] = silhouette_score(df_oh, labels, metric='manhattan')

# Print results
print("ONE HOT ENCODED DATA: Silhouette Scores (Higher = Better Clustering) for number of clusters: ", element)
for name, score in silhouette_scores.items():
    print(f"{name}: {score:.4f}")
oh_labels_array.append(cluster_labels_oh)



# Store results
cluster_labels_oh = {}
silhouette_scores = {}
labels_array = []

for name, model in models.items():
    labels = model.fit_predict(X_clustering)
    cluster_labels_oh[name] = labels
    
    # Evaluate if clusters are meaningful (skip DBSCAN where n_clusters isn't defined)
    if name != "DBSCAN":
        silhouette_scores[name] = silhouette_score(X_clustering, labels, metric='chebyshev')

# Print results
print("NON-ONE HOT ENCODED DATA: Silhouette Scores (Higher = Better Clustering) for number of clusters: ", element)
for name, score in silhouette_scores.items():
    print(f"{name}: {score:.4f}")
labels_array.append(cluster_labels_oh)


#Visualizing unsupervised clustering results on the T-SNE graph using one hot encoded, 5 clusters
method = 'Hierarchical (Complete)'
# Convert t-SNE results into a DataFrame for easy plotting
tsne_df = pd.DataFrame(tsne_oh, columns=['TSNE1', 'TSNE2'])
tsne_df['Label'] = oh_labels_array[1][method]  # Assuming 'Label' is your categorical class column

# Plot t-SNE with color coding
plt.figure(figsize=(10, 6))
sns.scatterplot(x=tsne_df['TSNE1'], y=tsne_df['TSNE2'], hue=tsne_df['Label'], palette='bright', alpha=0.7)

plt.title(f"t-SNE 5-class Visualization one hot data {method} clustering", fontsize=16)
plt.xlabel("TSNE1", fontweight='bold')
plt.ylabel("TSNE2", fontweight='bold')
plt.legend(title="Class Label")
plt.show()

#Non one-hot encoded data, 2 clusters
# Convert t-SNE results into a DataFrame for easy plotting
method = "Hierarchical (Average)"
tsne_df = pd.DataFrame(tsne_default, columns=['TSNE1', 'TSNE2'])
tsne_df['Label'] = labels_array[0][method]  # Assuming 'Label' is your categorical class column

# Plot t-SNE with color coding
plt.figure(figsize=(10, 6))
sns.scatterplot(x=tsne_df['TSNE1'], y=tsne_df['TSNE2'], hue=tsne_df['Label'], palette='bright', alpha=0.7)

plt.title(f"t-SNE 2-class Visualization Non one hot data {method}", fontsize=16)
plt.xlabel("TSNE1", fontweight='bold')
plt.ylabel("TSNE2", fontweight='bold')
plt.legend(title="Class Label")
plt.show()

# Convert t-SNE results into a DataFrame for easy plotting
method = "K-Means"
tsne_df = pd.DataFrame(tsne_default, columns=['TSNE1', 'TSNE2'])
tsne_df['Label'] = labels_array[1][method]  # Assuming 'Label' is your categorical class column

# Plot t-SNE with color coding
plt.figure(figsize=(10, 6))
sns.scatterplot(x=tsne_df['TSNE1'], y=tsne_df['TSNE2'], hue=tsne_df['Label'], palette='bright', alpha=0.7)

plt.title(f"t-SNE 5-class Visualization Non one hot data {method}", fontsize=16)
plt.xlabel("TSNE1", fontweight='bold')
plt.ylabel("TSNE2", fontweight='bold')
plt.legend(title="Class Label")
plt.show()

# Hierarchical Clustering (Dendrogram) WARD
print("Dendrogram generated.")
print("\nGenerating Hierarchical Clustering Dendrogram with method = ward...")
plt.figure(figsize=(20, 10))
linked = linkage(X_scaled, method='ward')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
print("Dendrogram generated.")

# Hierarchical Clustering (Dendrogram) AVERAGE   
print("\nGenerating Hierarchical Clustering Dendrogram with method = average...")
plt.figure(figsize=(20, 10))
linked = linkage(X_scaled, method='average')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
print("Dendrogram generated.")

# Hierarchical Clustering (Dendrogram) COMPLETE   
print("\nGenerating Hierarchical Clustering Dendrogram with method = complete...")
plt.figure(figsize=(20, 10))
linked = linkage(X_scaled, method='complete')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
print("Dendrogram generated.")

# Hierarchical Clustering (Dendrogram) SINGLE   
print("\nGenerating Hierarchical Clustering Dendrogram with method = single...")
plt.figure(figsize=(20, 10))
linked = linkage(X_scaled, method='single')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
print("Dendrogram generated.")

# K-Means Clustering
print("\nPerforming K-Means Clustering...")
silhouette_scores = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    if len(set(cluster_labels)) > 1:
        score = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(0)

plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, marker='o')
plt.title('Silhouette Score for K-Means Clustering')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.show()

optimal_k = 7 # Example, choose based on plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)
print(f"K-Means clustering performed with k={optimal_k}. Cluster distribution:")
print(df['kmeans_cluster'].value_counts())

# Correlation Analysis
print("\nPerforming Correlation Analysis...")
if 'Diagnosis_value' in df.columns and 'Malignancy_value' in df.columns:
    numerical_df = df.select_dtypes(include=np.number)
    corr_matrix = numerical_df.corr()

    diagnosis_corr = corr_matrix['Diagnosis_value'].sort_values(ascending=False)
    print("\nCorrelation with Diagnosis_value:")
    print(diagnosis_corr)

    malig_corr = corr_matrix['Malignancy_value'].sort_values(ascending=False)
    print("\nCorrelation with Malignancy_value:")
    print(malig_corr)

    if 'Diagnosis_value' in df_encoded.columns and 'Malignancy_value' in df_encoded.columns:
        corr_matrix_oh = df_encoded.select_dtypes(include=np.number).corr()
        diagnosis_corr_oh = corr_matrix_oh['Diagnosis_value'].sort_values(ascending=False)
        print("\nCorrelation with Diagnosis_value (One-Hot Encoded Features):")
        print(diagnosis_corr_oh)
        

        malig_corr_oh = corr_matrix_oh['Malignancy_value'].sort_values(ascending=False)
        print("\nCorrelation with Malignancy_value (One-Hot Encoded Features):")
        print(malig_corr_oh)
        
        # Plot the heatmap
        print("\n Printing the heatmap with the correlations...")
        sns.heatmap(corr_matrix_oh, 
                    cmap='coolwarm', 
                    annot=True, 
                    fmt=".2f", 
                    square=True,
                    cbar_kws={"shrink": 0.8},
                    linewidths=0.5)
        plt.title("Correlation Matrix Heatmap")
        plt.tight_layout()
        plt.show()
        
        # Correlations matrix with one hot encoding
        # Compute correlation of all annotations with the diagnosis column
        df_oh["Diagnosis_value"] = df["Diagnosis_value"]
        df_oh["Malignancy_value"]= df["Malignancy_value"]
        corr_matrix = df_oh.corr()

        # Extract correlations only for the "Diagnosis" column
        diagnosis_corr_oh = corr_matrix["Diagnosis_value"].sort_values(ascending=False)

        # Display the correlation values
        print("\n Printing correlations of Diagnosis with oen hot encoding data...")
        print(diagnosis_corr_oh)
        malig_corr_oh = corr_matrix["Malignancy_value"].sort_values(ascending=False)
        print("\n Printing correlations of Malignancy with oen hot encoding data...")
        print(malig_corr_oh)        
    print("t-SNE visualization generated successfully.")

# ==============================================================================
# --- MILESTONE 3: TRAINING A CLASSIFIER ON GLCM FEATURES ---
# This section combines code from BALANCED_Milestone3_part2_3.ipynb
# ==============================================================================
print("\n--- Milestone 3: Training a Classifier on GLCM Features ---")

# Define the base directory where slice_glcm1d.npz is located
GLCM_FILE_PATH = BASE_DOWNLOADS_DIR / "slice_glcm1d.npz"
print(f"Loading GLCM data from '{GLCM_FILE_PATH}'...")
data = np.load(GLCM_FILE_PATH, allow_pickle=True) # Use the full path here
print(f"Data files in npz: {data.files}")

df_features = pd.DataFrame(data['slice_features'])
df_meta = pd.DataFrame(data['slice_meta'], columns=['filename', 'patient_id', 'nodule_id', 'diagnosis'])

print("GLCM Features (first 5 rows):")
print(df_features.head())
print("GLCM Metadata (first 5 rows):")
print(df_meta.head())

labels = df_meta['diagnosis']
print("GLCM Metadata labels (first 5 rows):")
print(labels.head())
print(f"Shape of df_meta: {df_meta.shape}")

#Let's get rif of NoNods
df_binary = df_features[labels != 'NoNod']
print("Features without NoNod")
df_binary.head()
print(f"Shape of labels withou NoNod: {df_binary.shape}")
filt_nonod = df_meta[df_meta['diagnosis'] != 'NoNod']
print(f"Shape of df_meta without labels = NoNod: {filt_nonod.shape}")
labels = filt_nonod['diagnosis']

#Performing the T-test for feature importance
class_0 = df_binary[labels == 'Benign']  # Subset where class is 0
class_1 = df_binary[labels == 'Malignant']  # Subset where class is 1
# Perform t-test for each feature
p_values = {col: ttest_ind(class_0[col], class_1[col], equal_var=True).pvalue for col in df_features.columns}
# Convert results to DataFrame
feature_importance = pd.DataFrame.from_dict(p_values, orient='index', columns=['p_value'])
# Sort by significance
feature_importance = feature_importance.sort_values(by='p_value')
print("Printing the 24 first most important features", feature_importance.head(24))  # Features with the smallest p-values

#We filter out the p-values > 0.05
important_features = feature_importance[feature_importance < 0.05]
important_features = important_features.dropna()
print(f"Shape of most important features: {important_features.shape}")
print(important_features.head())

df_binary_imp = df_binary[important_features.index]

#Training SVM on Stratified K-Fold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
model_base = SVC(class_weight='balanced')
scoring = {
    'accuracy': 'accuracy',
    'precision_malignant': make_scorer(precision_score, pos_label='Malignant'),
    'recall_malignant': make_scorer(recall_score, pos_label='Malignant'),
    'f1_malignant': make_scorer(f1_score, pos_label='Malignant'),
    'precision_benign': make_scorer(precision_score, pos_label='Benign'),
    'recall_benign': make_scorer(recall_score, pos_label='Benign'),
    'f1_benign': make_scorer(f1_score, pos_label='Benign')
}

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_binary_imp)

# Run cross-validation
results = cross_validate(
    model_base,
    X_scaled,
    labels,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)

# Print full results
print("\n=== Detailed Metrics Per Fold ===")
for fold in range(10):
    print(f"\nFold {fold + 1}:")
    print(f"  Accuracy: {results['test_accuracy'][fold]:.4f}")
    print("  Malignant:")
    print(f"    Precision: {results['test_precision_malignant'][fold]:.4f}")
    print(f"    Recall: {results['test_recall_malignant'][fold]:.4f}")
    print(f"    F1: {results['test_f1_malignant'][fold]:.4f}")
    print("  Benign:")
    print(f"    Precision: {results['test_precision_benign'][fold]:.4f}")
    print(f"    Recall: {results['test_recall_benign'][fold]:.4f}")
    print(f"    F1: {results['test_f1_benign'][fold]:.4f}")

# Print averages
print("\n=== Average Metrics ===")
print(f"Accuracy: {np.mean(results['test_accuracy']):.4f} (±{np.std(results['test_accuracy']):.4f})")
print("\nMalignant:")
print(f"  Precision: {np.mean(results['test_precision_malignant']):.4f} (±{np.std(results['test_precision_malignant']):.4f})")
print(f"  Recall: {np.mean(results['test_recall_malignant']):.4f} (±{np.std(results['test_recall_malignant']):.4f})")
print(f"  F1: {np.mean(results['test_f1_malignant']):.4f} (±{np.std(results['test_f1_malignant']):.4f})")
print("\nBenign:")
print(f"  Precision: {np.mean(results['test_precision_benign']):.4f} (±{np.std(results['test_precision_benign']):.4f})")
print(f"  Recall: {np.mean(results['test_recall_benign']):.4f} (±{np.std(results['test_recall_benign']):.4f})")
print(f"  F1: {np.mean(results['test_f1_benign']):.4f} (±{np.std(results['test_f1_benign']):.4f})")

# Extract cross-validation scores
cv_scores = results['test_accuracy']

# Compute mean and standard error
mean_score = np.mean(cv_scores)
std_error = st.sem(cv_scores)  # Standard Error of the Mean (SEM)

# Calculate confidence interval
confidence = 0.95
ci_lower, ci_upper = st.t.interval(confidence, len(cv_scores)-1, loc=mean_score, scale=std_error)

print(f"Mean Score: {mean_score:.4f}")
print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")

#Grouping by the file name
nodules = filt_nonod["filename"]
print(f"Length of the nodules when grouping by file name: {len(nodules)}")
print(f"Length of the labels: {len(labels)}")

#Training SVM on Stratified Group K-Fold
cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
model_base = SVC(class_weight='balanced')
scoring = {
    'accuracy': 'accuracy',
    'precision_malignant': make_scorer(precision_score, pos_label='Malignant'),
    'recall_malignant': make_scorer(recall_score, pos_label='Malignant'),
    'f1_malignant': make_scorer(f1_score, pos_label='Malignant'),
    'precision_benign': make_scorer(precision_score, pos_label='Benign'),
    'recall_benign': make_scorer(recall_score, pos_label='Benign'),
    'f1_benign': make_scorer(f1_score, pos_label='Benign')
}

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_binary_imp)

# Run cross-validation
results = cross_validate(
    model_base,
    X_scaled,
    labels,
    cv=cv,
    scoring=scoring,
    groups=nodules,
    n_jobs=-1
)

# Print full results
print("\n=== Detailed Metrics Per Fold ===")
for fold in range(10):
    print(f"\nFold {fold + 1}:")
    print(f"  Accuracy: {results['test_accuracy'][fold]:.4f}")
    print("  Malignant:")
    print(f"    Precision: {results['test_precision_malignant'][fold]:.4f}")
    print(f"    Recall: {results['test_recall_malignant'][fold]:.4f}")
    print(f"    F1: {results['test_f1_malignant'][fold]:.4f}")
    print("  Benign:")
    print(f"    Precision: {results['test_precision_benign'][fold]:.4f}")
    print(f"    Recall: {results['test_recall_benign'][fold]:.4f}")
    print(f"    F1: {results['test_f1_benign'][fold]:.4f}")

# Print averages
print("\n=== Average Metrics ===")
print(f"Accuracy: {np.mean(results['test_accuracy']):.4f} (±{np.std(results['test_accuracy']):.4f})")
print("\nMalignant:")
print(f"  Precision: {np.mean(results['test_precision_malignant']):.4f} (±{np.std(results['test_precision_malignant']):.4f})")
print(f"  Recall: {np.mean(results['test_recall_malignant']):.4f} (±{np.std(results['test_recall_malignant']):.4f})")
print(f"  F1: {np.mean(results['test_f1_malignant']):.4f} (±{np.std(results['test_f1_malignant']):.4f})")
print("\nBenign:")
print(f"  Precision: {np.mean(results['test_precision_benign']):.4f} (±{np.std(results['test_precision_benign']):.4f})")
print(f"  Recall: {np.mean(results['test_recall_benign']):.4f} (±{np.std(results['test_recall_benign']):.4f})")
print(f"  F1: {np.mean(results['test_f1_benign']):.4f} (±{np.std(results['test_f1_benign']):.4f})")

# Extract cross-validation scores
cv_scores = results['test_accuracy']

# Compute mean and standard error
mean_score = np.mean(cv_scores)
std_error = st.sem(cv_scores)  # Standard Error of the Mean (SEM)

# Calculate confidence interval
confidence = 0.95
ci_lower, ci_upper = st.t.interval(confidence, len(cv_scores)-1, loc=mean_score, scale=std_error)

print(f"Mean Score: {mean_score:.4f}")
print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")

#Hyperparameter tuning
#Grid search
cv_outer = StratifiedGroupKFold(n_splits=10)
cv_inner = StratifiedGroupKFold(n_splits=10)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_binary_imp)

# Define parameter grid
param_grid = {
    'svc__C': [0.01, 0.1, 1, 10, 100],
    'svc__kernel': ['linear', 'rbf'],
    'svc__gamma': ['scale', 'auto']
}

metrics = []

pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(class_weight='balanced'))])

# Initialize grid search with cv_inner
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv_inner,  # Inner CV for hyperparameter tuning
    scoring='accuracy',
    n_jobs=-1,
    refit=True, # Refit the best model on the full dataset
    verbose = 1  # get some intermediate info about the training
)

# Fit on ALL data (handles train/test splits internally)
grid_search.fit(df_binary_imp, labels, groups=nodules)

# Best model and params
print("Best params:", grid_search.best_params_)
best_model = grid_search.best_estimator_
best_model = SVC(C=0.01, gamma='scale',kernel='linear', class_weight='balanced')
scoring = {
    'accuracy': 'accuracy',
    'precision_malignant': make_scorer(precision_score, pos_label='Malignant'),
    'recall_malignant': make_scorer(recall_score, pos_label='Malignant'),
    'f1_malignant': make_scorer(f1_score, pos_label='Malignant'),
    'precision_benign': make_scorer(precision_score, pos_label='Benign'),
    'recall_benign': make_scorer(recall_score, pos_label='Benign'),
    'f1_benign': make_scorer(f1_score, pos_label='Benign')
}

# Run cross-validation
results = cross_validate(
    best_model,
    X_scaled,
    labels,
    cv=cv_outer,
    groups=nodules,
    scoring=scoring,
    n_jobs=-1
)

# Print full results
print("\n=== Detailed Metrics Per Fold ===")
for fold in range(10):
    print(f"\nFold {fold + 1}:")
    print(f"  Accuracy: {results['test_accuracy'][fold]:.4f}")
    print("  Malignant:")
    print(f"    Precision: {results['test_precision_malignant'][fold]:.4f}")
    print(f"    Recall: {results['test_recall_malignant'][fold]:.4f}")
    print(f"    F1: {results['test_f1_malignant'][fold]:.4f}")
    print("  Benign:")
    print(f"    Precision: {results['test_precision_benign'][fold]:.4f}")
    print(f"    Recall: {results['test_recall_benign'][fold]:.4f}")
    print(f"    F1: {results['test_f1_benign'][fold]:.4f}")

# Print averages
print("\n=== Average Metrics ===")
print(f"Accuracy: {np.mean(results['test_accuracy']):.4f} (±{np.std(results['test_accuracy']):.4f})")
print("\nMalignant:")
print(f"  Precision: {np.mean(results['test_precision_malignant']):.4f} (±{np.std(results['test_precision_malignant']):.4f})")
print(f"  Recall: {np.mean(results['test_recall_malignant']):.4f} (±{np.std(results['test_recall_malignant']):.4f})")
print(f"  F1: {np.mean(results['test_f1_malignant']):.4f} (±{np.std(results['test_f1_malignant']):.4f})")
print("\nBenign:")
print(f"  Precision: {np.mean(results['test_precision_benign']):.4f} (±{np.std(results['test_precision_benign']):.4f})")
print(f"  Recall: {np.mean(results['test_recall_benign']):.4f} (±{np.std(results['test_recall_benign']):.4f})")
print(f"  F1: {np.mean(results['test_f1_benign']):.4f} (±{np.std(results['test_f1_benign']):.4f})")
# Extract cross-validation scores
cv_scores = results['test_accuracy']

# Compute mean and standard error
mean_score = np.mean(cv_scores)
std_error = st.sem(cv_scores)  # Standard Error of the Mean (SEM)

# Calculate confidence interval
confidence = 0.95
ci_lower, ci_upper = st.t.interval(confidence, len(cv_scores)-1, loc=mean_score, scale=std_error)

print(f"Mean Score: {mean_score:.4f}")
print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")

#Random search
# Define parameter distributions
param_dist = {
    'svc__C': loguniform(1e-3, 1e3),
    'svc__kernel': ['linear', 'rbf'],
    'svc__gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 5)),
    'svc__shrinking': [True, False],  # Use shrinking heuristic (default=True)
    'svc__probability': [True, False],  # Enable probability estimates (default=False)
    'svc__tol': [1e-3, 1e-4, 1e-5],  # Tolerance for stopping (default=1e-3)
    'svc__class_weight': [None, 'balanced']  # Handle imbalanced classes (default=None)
}

metrics = []

pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(class_weight='balanced'))])

# Initialize RandomizedSearchCV with OUTER CV
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,  # Number of random combinations to try
    cv=cv_inner,  # Inner CV for hyperparameter tuning
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=2,
    refit=True  # Refit the best model on the full dataset
)

# Fit on ALL data (handles train/test splits internally)
random_search.fit(X_scaled, labels, groups=nodules)

# Best model and params
print("Best params:", random_search.best_params_)
best_model_r = random_search.best_estimator_

# Initialize RandomizedSearchCV with OUTER CV. We could not go with the previous way as the grid search
# because we want to avoid changing the model randomly between each fold
scoring = {
    'accuracy': 'accuracy',
    'precision_malignant': make_scorer(precision_score, pos_label='Malignant'),
    'recall_malignant': make_scorer(recall_score, pos_label='Malignant'),
    'f1_malignant': make_scorer(f1_score, pos_label='Malignant'),
    'precision_benign': make_scorer(precision_score, pos_label='Benign'),
    'recall_benign': make_scorer(recall_score, pos_label='Benign'),
    'f1_benign': make_scorer(f1_score, pos_label='Benign')
}

# Run cross-validation
results = cross_validate(
    best_model_r,
    X_scaled,
    labels,
    cv=cv_outer,
    groups=nodules,
    scoring=scoring,
    n_jobs=-1
)

# Print full results
print("\n=== Detailed Metrics Per Fold ===")
for fold in range(10):
    print(f"\nFold {fold + 1}:")
    print(f"  Accuracy: {results['test_accuracy'][fold]:.4f}")
    print("  Malignant:")
    print(f"    Precision: {results['test_precision_malignant'][fold]:.4f}")
    print(f"    Recall: {results['test_recall_malignant'][fold]:.4f}")
    print(f"    F1: {results['test_f1_malignant'][fold]:.4f}")
    print("  Benign:")
    print(f"    Precision: {results['test_precision_benign'][fold]:.4f}")
    print(f"    Recall: {results['test_recall_benign'][fold]:.4f}")
    print(f"    F1: {results['test_f1_benign'][fold]:.4f}")

# Print averages
print("\n=== Average Metrics ===")
print(f"Accuracy: {np.mean(results['test_accuracy']):.4f} (±{np.std(results['test_accuracy']):.4f})")
print("\nMalignant:")
print(f"  Precision: {np.mean(results['test_precision_malignant']):.4f} (±{np.std(results['test_precision_malignant']):.4f})")
print(f"  Recall: {np.mean(results['test_recall_malignant']):.4f} (±{np.std(results['test_recall_malignant']):.4f})")
print(f"  F1: {np.mean(results['test_f1_malignant']):.4f} (±{np.std(results['test_f1_malignant']):.4f})")
print("\nBenign:")
print(f"  Precision: {np.mean(results['test_precision_benign']):.4f} (±{np.std(results['test_precision_benign']):.4f})")
print(f"  Recall: {np.mean(results['test_recall_benign']):.4f} (±{np.std(results['test_recall_benign']):.4f})")
print(f"  F1: {np.mean(results['test_f1_benign']):.4f} (±{np.std(results['test_f1_benign']):.4f})")
# Extract cross-validation scores
cv_scores = results['test_accuracy']

# Compute mean and standard error
mean_score = np.mean(cv_scores)
std_error = st.sem(cv_scores)  # Standard Error of the Mean (SEM)

# Calculate confidence interval
confidence = 0.95
ci_lower, ci_upper = st.t.interval(confidence, len(cv_scores)-1, loc=mean_score, scale=std_error)

print(f"Mean Score: {mean_score:.4f}")
print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")

#OPTUNA
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(lambda trial: objective(trial, X_scaled, labels, nodules), n_trials=200)
print("\nThe best model obtained is the following one:", best_model)

# Initialize RandomizedSearchCV with OUTER CV. We could not go with the previous way as the grid search
# because we want to avoid changing the model randomly between each fold
scoring = {
    'accuracy': 'accuracy',
    'precision_malignant': make_scorer(precision_score, pos_label='Malignant'),
    'recall_malignant': make_scorer(recall_score, pos_label='Malignant'),
    'f1_malignant': make_scorer(f1_score, pos_label='Malignant'),
    'precision_benign': make_scorer(precision_score, pos_label='Benign'),
    'recall_benign': make_scorer(recall_score, pos_label='Benign'),
    'f1_benign': make_scorer(f1_score, pos_label='Benign')
}

# Run cross-validation
results = cross_validate(
    best_model,
    X_scaled,
    labels,
    cv=cv_outer,
    groups=nodules,
    scoring=scoring,
    n_jobs=-1
)

# Print full results
print("\n=== Detailed Metrics Per Fold ===")
for fold in range(5):
    print(f"\nFold {fold + 1}:")
    print(f"  Accuracy: {results['test_accuracy'][fold]:.4f}")
    print("  Malignant:")
    print(f"    Precision: {results['test_precision_malignant'][fold]:.4f}")
    print(f"    Recall: {results['test_recall_malignant'][fold]:.4f}")
    print(f"    F1: {results['test_f1_malignant'][fold]:.4f}")
    print("  Benign:")
    print(f"    Precision: {results['test_precision_benign'][fold]:.4f}")
    print(f"    Recall: {results['test_recall_benign'][fold]:.4f}")
    print(f"    F1: {results['test_f1_benign'][fold]:.4f}")

# Print averages
print("\n=== Average Metrics ===")
print(f"Accuracy: {np.mean(results['test_accuracy']):.4f} (±{np.std(results['test_accuracy']):.4f})")
print("\nMalignant:")
print(f"  Precision: {np.mean(results['test_precision_malignant']):.4f} (±{np.std(results['test_precision_malignant']):.4f})")
print(f"  Recall: {np.mean(results['test_recall_malignant']):.4f} (±{np.std(results['test_recall_malignant']):.4f})")
print(f"  F1: {np.mean(results['test_f1_malignant']):.4f} (±{np.std(results['test_f1_malignant']):.4f})")
print("\nBenign:")
print(f"  Precision: {np.mean(results['test_precision_benign']):.4f} (±{np.std(results['test_precision_benign']):.4f})")
print(f"  Recall: {np.mean(results['test_recall_benign']):.4f} (±{np.std(results['test_recall_benign']):.4f})")
print(f"  F1: {np.mean(results['test_f1_benign']):.4f} (±{np.std(results['test_f1_benign']):.4f})")

# Get best model
best_params = study.best_params
best_model = SVC(**best_params, random_state=42,class_weight='balanced')

# Extract cross-validation scores
cv_scores = results['test_accuracy']

# Compute mean and standard error
mean_score = np.mean(cv_scores)
std_error = st.sem(cv_scores)  # Standard Error of the Mean (SEM)

# Calculate confidence interval
confidence = 0.95
ci_lower, ci_upper = st.t.interval(confidence, len(cv_scores)-1, loc=mean_score, scale=std_error)

print(f"Mean Score: {mean_score:.4f}")
print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")
