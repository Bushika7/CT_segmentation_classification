import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GridSearchCV, RandomizedSearchCV, cross_validate, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler
# Removed: from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform
import scipy.stats as st
import optuna
from optuna.samplers import TPESampler

# --- Configuration (Ensure this matches your actual paths) ---
BASE_DOWNLOADS_DIR = Path("C:\\Users\\sansg\\Downloads")

# --- Functions (Re-defining objective for clarity, but assume it's global or passed) ---
def objective(trial, X_train, y_train, groups_train, cv_inner):
    """
    Optuna objective function for hyperparameter optimization of SVC.
    """
    # Hyperparameters to optimize
    C = trial.suggest_float("C", 1e-3, 1e3, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])

    # Conditional parameters for gamma
    if kernel == 'rbf':
        gamma = trial.suggest_float("gamma", 1e-3, 1e3, log=True)
    else:
        gamma = 'scale'

    pipeline_optuna = Pipeline([('scaler', StandardScaler()),
                                ('svc', SVC(C=C, kernel=kernel, gamma=gamma,
                                            random_state=42, class_weight='balanced'))])

    scores = cross_val_score(
        pipeline_optuna, X_train, y_train, cv=cv_inner, groups=groups_train, scoring="accuracy", n_jobs=-1
    )
    return scores.mean()

# ==============================================================================
# --- MILESTONE 4: CLASSIFIER TRAINING ON VGG FEATURES ---
# ==============================================================================
print("\n--- Milestone 4: Training a Classifier on VGG Features ---")

VGG_FILE_PATH = BASE_DOWNLOADS_DIR / "vgg_features.npz"
print(f"Loading VGG data from '{VGG_FILE_PATH}'...")

try:
    vgg_data = np.load(VGG_FILE_PATH, allow_pickle=True)
    print(f"Data files in npz: {vgg_data.files}")
except FileNotFoundError:
    print(f"Error: vgg_features.npz not found at {VGG_FILE_PATH}. Please ensure the path is correct.")
    exit()

vgg_features_array = vgg_data['vgg_features'] if 'vgg_features' in vgg_data.files else vgg_data['features']
vgg_diagnosis_labels = vgg_data['diagnosis']
vgg_ids = vgg_data['id']
flat_ranking_idx = vgg_data['flat_ranking_idx']

# Squeeze the array to remove dimensions of size 1
if vgg_features_array.ndim > 2:
    vgg_features_array = vgg_features_array.squeeze()

print(f"Shape of VGG features array after squeeze: {vgg_features_array.shape}")

vgg_feature_columns = [f'vgg_feature_{i}' for i in range(vgg_features_array.shape[1])]
df_vgg_features = pd.DataFrame(vgg_features_array, columns=vgg_feature_columns)

df_vgg_meta = pd.DataFrame({
    'id': vgg_ids,
    'diagnosis': vgg_diagnosis_labels
})

print("VGG Features (first 5 rows):")
print(df_vgg_features.head())
print("VGG Metadata (first 5 rows):")
print(df_vgg_meta.head())

# Filter out 'NoNod' entries
initial_num_samples = len(df_vgg_meta)
df_vgg_features_filtered = df_vgg_features[df_vgg_meta['diagnosis'] != 'NoNod'].copy()
df_vgg_meta_filtered = df_vgg_meta[df_vgg_meta['diagnosis'] != 'NoNod'].copy()

# Ensure vgg_labels are integers if they are not already (e.g., if they were originally strings '0'/'1')
vgg_labels = df_vgg_meta_filtered['diagnosis'].astype(int).copy()

print(f"Original number of samples: {initial_num_samples}")
print(f"Number of samples after filtering 'NoNod': {len(df_vgg_features_filtered)}")

# --- Corrected section: Directly set pos_label as labels are already numerical (0 or 1) ---
# Based on the VGG Metadata output (e.g., diagnosis: 1),
# we assume 1 corresponds to 'Malignant' (positive class) and 0 to 'Benign'.
pos_label_malignant = 1
pos_label_benign = 0
# --- End of Corrected section ---

# Extract patient_id for StratifiedGroupKFold
df_vgg_meta_filtered['patient_id'] = df_vgg_meta_filtered['id'].apply(lambda x: x.split('_')[0])
vgg_groups = df_vgg_meta_filtered['patient_id']

# Feature Selection using flat_ranking_idx
N_VGG_FEATURES = 500
selected_vgg_indices = flat_ranking_idx[:N_VGG_FEATURES]
df_vgg_selected_features = df_vgg_features_filtered.iloc[:, selected_vgg_indices].copy()

print(f"\nSelected top {N_VGG_FEATURES} VGG features.")
print("Selected VGG Features (first 5 rows):")
print(df_vgg_selected_features.head())

# Define scoring metrics for cross-validation using numerical pos_label
scoring = {
    'accuracy': 'accuracy',
    'precision_malignant': make_scorer(precision_score, pos_label=pos_label_malignant, zero_division=0),
    'recall_malignant': make_scorer(recall_score, pos_label=pos_label_malignant, zero_division=0),
    'f1_malignant': make_scorer(f1_score, pos_label=pos_label_malignant, zero_division=0),
    'precision_benign': make_scorer(precision_score, pos_label=pos_label_benign, zero_division=0),
    'recall_benign': make_scorer(recall_score, pos_label=pos_label_benign, zero_division=0),
    'f1_benign': make_scorer(f1_score, pos_label=pos_label_benign, zero_division=0)
}

# --- SVM Training with Stratified K-Fold (basic, without grouping) ---
print("\n--- Training SVM with Stratified K-Fold (VGG Features) ---")
cv_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_base_vgg_skf = SVC(class_weight='balanced', random_state=42)

scaler_vgg_skf = StandardScaler()
X_vgg_scaled_skf = scaler_vgg_skf.fit_transform(df_vgg_selected_features)

results_vgg_skf = cross_validate(
    model_base_vgg_skf,
    X_vgg_scaled_skf,
    vgg_labels, # Use original numerical labels
    cv=cv_skf,
    scoring=scoring,
    n_jobs=-1
)

print("\n=== Average Metrics (VGG Features - Stratified K-Fold) ===")
print(f"Accuracy: {np.mean(results_vgg_skf['test_accuracy']):.4f} (±{np.std(results_vgg_skf['test_accuracy']):.4f})")
print("\nMalignant:")
print(f"  Precision: {np.mean(results_vgg_skf['test_precision_malignant']):.4f} (±{np.std(results_vgg_skf['test_precision_malignant']):.4f})")
print(f"  Recall: {np.mean(results_vgg_skf['test_recall_malignant']):.4f} (±{np.std(results_vgg_skf['test_recall_malignant']):.4f})")
print(f"  F1: {np.mean(results_vgg_skf['test_f1_malignant']):.4f} (±{np.std(results_vgg_skf['test_f1_malignant']):.4f})")
print("\nBenign:")
print(f"  Precision: {np.mean(results_vgg_skf['test_precision_benign']):.4f} (±{np.std(results_vgg_skf['test_precision_benign']):.4f})")
print(f"  Recall: {np.mean(results_vgg_skf['test_recall_benign']):.4f} (±{np.std(results_vgg_skf['test_recall_benign']):.4f})")
print(f"  F1: {np.mean(results_vgg_skf['test_f1_benign']):.4f} (±{np.std(results_vgg_skf['test_f1_benign']):.4f})")

# --- SVM Training with Stratified Group K-Fold (to avoid patient leakage) ---
print("\n--- Training SVM with Stratified Group K-Fold (VGG Features) ---")
cv_sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
model_base_vgg_sgkf = SVC(class_weight='balanced', random_state=42)

scaler_vgg_sgkf = StandardScaler()
X_vgg_scaled_sgkf = scaler_vgg_sgkf.fit_transform(df_vgg_selected_features)

results_vgg_sgkf = cross_validate(
    model_base_vgg_sgkf,
    X_vgg_scaled_sgkf,
    vgg_labels, # Use original numerical labels
    cv=cv_sgkf,
    scoring=scoring,
    groups=vgg_groups,
    n_jobs=-1
)

print("\n=== Average Metrics (VGG Features - Stratified Group K-Fold) ===")
print(f"Accuracy: {np.mean(results_vgg_sgkf['test_accuracy']):.4f} (±{np.std(results_vgg_sgkf['test_accuracy']):.4f})")
print("\nMalignant:")
print(f"  Precision: {np.mean(results_vgg_sgkf['test_precision_malignant']):.4f} (±{np.std(results_vgg_sgkf['test_precision_malignant']):.4f})")
print(f"  Recall: {np.mean(results_vgg_sgkf['test_recall_malignant']):.4f} (±{np.std(results_vgg_sgkf['test_recall_malignant']):.4f})")
print(f"  F1: {np.mean(results_vgg_sgkf['test_f1_malignant']):.4f} (±{np.std(results_vgg_sgkf['test_f1_malignant']):.4f})")
print("\nBenign:")
print(f"  Precision: {np.mean(results_vgg_sgkf['test_precision_benign']):.4f} (±{np.std(results_vgg_sgkf['test_precision_benign']):.4f})")
print(f"  Recall: {np.mean(results_vgg_sgkf['test_recall_benign']):.4f} (±{np.std(results_vgg_sgkf['test_recall_benign']):.4f})")
print(f"  F1: {np.mean(results_vgg_sgkf['test_f1_benign']):.4f} (±{np.std(results_vgg_sgkf['test_f1_benign']):.4f})")

# --- Hyperparameter Tuning (Grid Search) for VGG Features ---
print("\n--- Hyperparameter Tuning (Grid Search) for VGG Features ---")
cv_outer_vgg_gs = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
cv_inner_vgg_gs = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)

pipeline_vgg_gs = Pipeline([('scaler', StandardScaler()), ('svc', SVC(class_weight='balanced', random_state=42))])

param_grid_vgg_gs = {
    'svc__C': [0.01, 0.1, 1, 10, 100],
    'svc__kernel': ['linear', 'rbf'],
    'svc__gamma': ['scale', 'auto']
}

grid_search_vgg = GridSearchCV(
    pipeline_vgg_gs,
    param_grid_vgg_gs,
    cv=cv_inner_vgg_gs,
    scoring='accuracy',
    n_jobs=-1,
    refit=True,
    verbose=1
)

grid_search_vgg.fit(df_vgg_selected_features, vgg_labels, groups=vgg_groups) # Use original numerical labels

print("\nBest params (Grid Search VGG):", grid_search_vgg.best_params_)
best_model_vgg_gs = grid_search_vgg.best_estimator_

results_vgg_gs_best = cross_validate(
    best_model_vgg_gs,
    df_vgg_selected_features,
    vgg_labels, # Use original numerical labels
    cv=cv_outer_vgg_gs,
    groups=vgg_groups,
    scoring=scoring,
    n_jobs=-1
)

print("\n=== Average Metrics (VGG Features - Grid Search Best Model) ===")
print(f"Accuracy: {np.mean(results_vgg_gs_best['test_accuracy']):.4f} (±{np.std(results_vgg_gs_best['test_accuracy']):.4f})")
print("\nMalignant:")
print(f"  Precision: {np.mean(results_vgg_gs_best['test_precision_malignant']):.4f} (±{np.std(results_vgg_gs_best['test_precision_malignant']):.4f})")
print(f"  Recall: {np.mean(results_vgg_gs_best['test_recall_malignant']):.4f} (±{np.std(results_vgg_gs_best['test_recall_malignant']):.4f})")
print(f"  F1: {np.mean(results_vgg_gs_best['test_f1_malignant']):.4f} (±{np.std(results_vgg_gs_best['test_f1_malignant']):.4f})")
print("\nBenign:")
print(f"  Precision: {np.mean(results_vgg_gs_best['test_precision_benign']):.4f} (±{np.std(results_vgg_gs_best['test_precision_benign']):.4f})")
print(f"  Recall: {np.mean(results_vgg_gs_best['test_recall_benign']):.4f} (±{np.std(results_vgg_gs_best['test_recall_benign']):.4f})")
print(f"  F1: {np.mean(results_vgg_gs_best['test_f1_benign']):.4f} (±{np.std(results_vgg_gs_best['test_f1_benign']):.4f})")


# --- Hyperparameter Tuning (Random Search) for VGG Features ---
print("\n--- Hyperparameter Tuning (Random Search) for VGG Features ---")
param_dist_vgg_rs = {
    'svc__C': loguniform(1e-3, 1e3),
    'svc__kernel': ['linear', 'rbf'],
    'svc__gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 5)),
    'svc__shrinking': [True, False],
    'svc__probability': [True, False],
    'svc__tol': [1e-3, 1e-4, 1e-5],
    'svc__class_weight': [None, 'balanced']
}

random_search_vgg = RandomizedSearchCV(
    pipeline_vgg_gs,
    param_distributions=param_dist_vgg_rs,
    n_iter=20,
    cv=cv_inner_vgg_gs,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=2,
    refit=True
)

random_search_vgg.fit(df_vgg_selected_features, vgg_labels, groups=vgg_groups) # Use original numerical labels

print("\nBest params (Random Search VGG):", random_search_vgg.best_params_)
best_model_vgg_rs = random_search_vgg.best_estimator_

results_vgg_rs_best = cross_validate(
    best_model_vgg_rs,
    df_vgg_selected_features,
    vgg_labels, # Use original numerical labels
    cv=cv_outer_vgg_gs,
    groups=vgg_groups,
    scoring=scoring,
    n_jobs=-1
)

print("\n=== Average Metrics (VGG Features - Random Search Best Model) ===")
print(f"Accuracy: {np.mean(results_vgg_rs_best['test_accuracy']):.4f} (±{np.std(results_vgg_rs_best['test_accuracy']):.4f})")
print("\nMalignant:")
print(f"  Precision: {np.mean(results_vgg_rs_best['test_precision_malignant']):.4f} (±{np.std(results_vgg_rs_best['test_precision_malignant']):.4f})")
print(f"  Recall: {np.mean(results_vgg_rs_best['test_recall_malignant']):.4f} (±{np.std(results_vgg_rs_best['test_recall_malignant']):.4f})")
print(f"  F1: {np.mean(results_vgg_rs_best['test_f1_malignant']):.4f} (±{np.std(results_vgg_rs_best['test_f1_malignant']):.4f})")
print("\nBenign:")
print(f"  Precision: {np.mean(results_vgg_rs_best['test_precision_benign']):.4f} (±{np.std(results_vgg_rs_best['test_precision_benign']):.4f})")
print(f"  Recall: {np.mean(results_vgg_rs_best['test_recall_benign']):.4f} (±{np.std(results_vgg_rs_best['test_recall_benign']):.4f})")
print(f"  F1: {np.mean(results_vgg_rs_best['test_f1_benign']):.4f} (±{np.std(results_vgg_rs_best['test_f1_benign']):.4f})")


# --- Hyperparameter Tuning (Optuna) for VGG Features ---
print("\n--- Hyperparameter Tuning (Optuna) for VGG Features ---")
study_vgg = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study_vgg.optimize(lambda trial: objective(trial, X_vgg_scaled_sgkf, vgg_labels, vgg_groups, cv_inner_vgg_gs), n_trials=20) # Use original numerical labels

print("\nBest params (Optuna VGG):", study_vgg.best_params)
best_model_vgg_optuna = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(C=study_vgg.best_params['C'],
                kernel=study_vgg.best_params['kernel'],
                gamma=study_vgg.best_params['gamma'] if 'gamma' in study_vgg.best_params and study_vgg.best_params['kernel'] == 'rbf' else 'scale',
                class_weight='balanced',
                random_state=42))
])

results_vgg_optuna_best = cross_validate(
    best_model_vgg_optuna,
    df_vgg_selected_features,
    vgg_labels, # Use original numerical labels
    cv=cv_outer_vgg_gs,
    groups=vgg_groups,
    scoring=scoring,
    n_jobs=-1
)

print("\n=== Average Metrics (VGG Features - Optuna Best Model) ===")
print(f"Accuracy: {np.mean(results_vgg_optuna_best['test_accuracy']):.4f} (±{np.std(results_vgg_optuna_best['test_accuracy']):.4f})")
print("\nMalignant:")
print(f"  Precision: {np.mean(results_vgg_optuna_best['test_precision_malignant']):.4f} (±{np.std(results_vgg_optuna_best['test_precision_malignant']):.4f})")
print(f"  Recall: {np.mean(results_vgg_optuna_best['test_recall_malignant']):.4f} (±{np.std(results_vgg_optuna_best['test_recall_malignant']):.4f})")
print(f"  F1: {np.mean(results_vgg_optuna_best['test_f1_malignant']):.4f} (±{np.std(results_vgg_optuna_best['test_f1_malignant']):.4f})")
print("\nBenign:")
print(f"  Precision: {np.mean(results_vgg_optuna_best['test_precision_benign']):.4f} (±{np.std(results_vgg_optuna_best['test_precision_benign']):.4f})")
print(f"  Recall: {np.mean(results_vgg_optuna_best['test_recall_benign']):.4f} (±{np.std(results_vgg_optuna_best['test_recall_benign']):.4f})")
print(f"  F1: {np.mean(results_vgg_optuna_best['test_f1_benign']):.4f} (±{np.std(results_vgg_optuna_best['test_f1_benign']):.4f})")

# ==============================================================================
# --- Overall Best Model Comparison (Add this at the very end of your script) ---
best_glcm_accuracy = 0.8500 # This value should be obtained from your GLCM analysis
best_vgg_accuracy = np.mean(results_vgg_optuna_best['test_accuracy'])

print("\n--- Final Comparison: VGG vs. PyRadiomics GLCM ---")
print(f"Best VGG Features Accuracy (from Optuna): {best_vgg_accuracy:.4f}")
print(f"Best PyRadiomics GLCM Features Accuracy (from GLCM analysis): {best_glcm_accuracy:.4f}")

if best_vgg_accuracy > best_glcm_accuracy:
    print(f"\nOverall, the VGG Features ({best_vgg_accuracy:.4f}) performed better than PyRadiomics GLCM Features ({best_glcm_accuracy:.4f}).")
elif best_glcm_accuracy > best_vgg_accuracy:
    print(f"\nOverall, the PyRadiomics GLCM Features ({best_glcm_accuracy:.4f}) performed better than VGG Features ({best_vgg_accuracy:.4f}).")
else:
    print(f"\nBoth VGG Features and PyRadiomics GLCM Features performed similarly (Accuracy: {best_vgg_accuracy:.4f}).")