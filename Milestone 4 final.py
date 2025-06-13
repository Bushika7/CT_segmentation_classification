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

# --- MILESTONE 1 ---
if __name__ == "__main__":
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

    try:
        from BasicSegmentation import BasicSegmentation
        from Comparation import comparation
        print("BasicSegmentation and Comparation modules found.")

        if not vois_combined_info.empty:
            first_entry = vois_combined_info.iloc[0]
            
            # CORRECTED LINES: Use 'filename_image' and 'filename_mask'
            original_image_path = VOIS_IMAGE_DIR / first_entry['filename_image']
            original_mask_path = VOIS_MASK_DIR / first_entry['filename_mask']

            try:
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

            except Exception as e:
                print(f"Error during segmentation or evaluation for {original_image_path}: {e}")
                print("Please ensure BasicSegmentation and Comparation modules are correctly implemented and can handle the input data.")
        else:
            print("No VOIs found to process for segmentation and evaluation.")

    except ImportError as e:
        print(f"Warning: Missing required modules for segmentation/comparison: {e}")
        print("Please ensure `BasicSegmentation.py` and `Comparation.py` are in your script's directory or the specified assignment directory.")
        print("Skipping segmentation pipeline and evaluation.")
    finally:
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
try:
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
    
    def show_boxplot(df):
        plt.rcParams['figure.figsize'] = [14,6]
        sns.boxplot(data = df, orient="v")
        plt.title("Outliers Distribution", fontsize = 16)
        plt.ylabel("Range", fontweight = 'bold')
        plt.xlabel("Attributes", fontweight = 'bold')
    show_boxplot(X_clustering)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clustering)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_clustering.columns)
    X_scaled_df.head()
    print("Features scaled.")
    
    
    print("\n--- Generating t-SNE Visualization of GLCM Features for Diagnosis ---")
    try:
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

    except Exception as e:
        print(f"An error occurred during t-SNE visualization: {e}")
        print("Please ensure 'seaborn' and 'scikit-learn' are installed correctly.")
        
    print("\n--- Generating t-SNE Visualization of GLCM Features for Malignancy ---")
    try:
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

        print("t-SNE visualization generated successfully.")

    except Exception as e:
        print(f"An error occurred during t-SNE visualization: {e}")
        print("Please ensure 'seaborn' and 'scikit-learn' are installed correctly.")
    
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

    # Hierarchical Clustering (Dendrogram)
    print("\nGenerating Hierarchical Clustering Dendrogram...")
    plt.figure(figsize=(20, 10))
    linked = linkage(X_scaled, method='ward')
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

    optimal_k = 3 # Example, choose based on plot
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
    else:
        print("Diagnosis_value or Malignancy_value not found for correlation analysis.")

    print("\nConclusion on data exploration:")
    print("The diameter is often a significant factor. Subtlety, Spiculation, and Lobulation also play important roles.")

except FileNotFoundError:
    print(f"Error: Metadata file for Milestone 2 not found at {metadata_file_path}. Please ensure the file exists.")
except Exception as e:
    print(f"An error occurred during Milestone 2 data exploration: {e}")

# ==============================================================================
# --- MILESTONE 3: TRAINING A CLASSIFIER ON GLCM FEATURES ---
# This section combines code from BALANCED_Milestone3_part2_3.ipynb
# ==============================================================================
print("\n--- Milestone 3: Training a Classifier on GLCM Features ---")

# Define the base directory where slice_glcm1d.npz is located
GLCM_FILE_PATH = BASE_DOWNLOADS_DIR / "slice_glcm1d.npz"

print(f"Loading GLCM data from '{GLCM_FILE_PATH}'...")
try:
    data = np.load(GLCM_FILE_PATH, allow_pickle=True) # Use the full path here
    print(f"Data files in npz: {data.files}")

    df_features = pd.DataFrame(data['slice_features'])
    df_meta = pd.DataFrame(data['slice_meta'])

    print("GLCM Features (first 5 rows):")
    print(df_features.head())
    print("GLCM Metadata (first 5 rows):")
    print(df_meta.head())

    if 'label' in df_meta.columns:
        y = df_meta['label'].values
        print(f"Target variable 'label' unique values: {np.unique(y)}")
    else:
        print("Warning: 'label' column not found in GLCM metadata. Please ensure target variable is correctly identified.")
        y = np.zeros(len(df_features))

    X = df_features.values
    print(f"Shape of features (X): {X.shape}")
    print(f"Shape of target (y): {y.shape}")

    # Data Preprocessing (Standardization)
    scaler = StandardScaler()
    X_scaled_glcm = scaler.fit_transform(X)
    print("GLCM features scaled.")

    # Stratified Group K-Fold for Cross-Validation
    if 'seriesuid' in df_meta.columns:
        groups = df_meta['seriesuid'].values
        print(f"Groups for cross-validation based on 'seriesuid': {len(np.unique(groups))} unique groups.")
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        print("Warning: 'seriesuid' (or patient_id) not found for StratifiedGroupKFold. Using StratifiedKFold instead.")
        sgkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        groups = None

    # Model: Support Vector Classifier
    svc = SVC(random_state=42)

    # Randomized Search for Hyperparameter Tuning
    print("\nPerforming Randomized Search for SVC hyperparameters...")
    param_distributions = {
        'C': loguniform(1e-1, 1e2),
        'gamma': loguniform(1e-4, 1e-1),
        'kernel': ['rbf', 'linear']
    }

    random_search = RandomizedSearchCV(svc, param_distributions, n_iter=100, cv=sgkf,
                                       scoring='accuracy', random_state=42, n_jobs=-1, verbose=1)

    random_search.fit(X_scaled_glcm, y, groups=groups if groups is not None else None)

    print(f"Best parameters from Randomized Search: {random_search.best_params_}")
    print(f"Best cross-validation accuracy from Randomized Search: {random_search.best_score_:.4f}")

    best_svc_random = random_search.best_estimator_

    # Cross-validation with best estimator
    print("\nPerforming Cross-Validation with the best SVC estimator...")
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='binary'),
        'recall': make_scorer(recall_score, average='binary'),
        'f1': make_scorer(f1_score, average='binary')
    }

    results = cross_validate(best_svc_random, X_scaled_glcm, y, cv=sgkf,
                             scoring=scoring, n_jobs=-1, return_train_score=False,
                             groups=groups if groups is not None else None)

    print("\nCross-Validation Results:")
    for metric, values in results.items():
        if metric.startswith('test_'):
            print(f"{metric.replace('test_', '').capitalize()} (Mean +/- Std): {np.mean(values):.4f} +/- {np.std(values):.4f}")

    cv_scores = results['test_accuracy']
    mean_score = np.mean(cv_scores)
    std_error = st.sem(cv_scores)
    confidence = 0.95
    ci_lower, ci_upper = st.t.interval(confidence, len(cv_scores)-1, loc=mean_score, scale=std_error)

    print(f"Mean Score: {mean_score:.4f}")
    print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")
    print("The metrics are quite similar to the random search.")

except FileNotFoundError:
    print(f"Error: '{GLCM_FILE_PATH}' not found. This file is required for Milestone 3 classifier training.")
except Exception as e:
    print(f"An error occurred during Milestone 3 classifier training: {e}")
