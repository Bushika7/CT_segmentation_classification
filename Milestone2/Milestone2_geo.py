# -*- coding: utf-8 -*-
"""
Created on Sun May  4 21:45:59 2025

@author: sansg
"""
### IMPORT PY LIBRARIES
# Python Library 2 manage volumetric data
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from pathlib import Path

print("All packages installed and working!")

# Milestone 2 (Classification): Data Exploration 
#1. Use different unsupervised techniques (eg. hierarchical clustering) and statistical tests to get correlations across radiological descriptions and also detect those annotations more relevant to the diagnosis.
# We import the data from the directory where it is saved in our computer
# dirname = "C:\\Users\\sansg\\Downloads\\MetadatabyNoduleMaxVoting.xlsx"
root_folder = Path(__file__).parent.parent
meta = "MetadatabyNoduleMaxVoting.xlsx"
dirname=Path(root_folder, meta)
df = pd.read_excel(dirname)
print("✅ Data imported! Printing the first 5 elements")
df.head(5)

max_voting = df.groupby(['patient_id']).agg(lambda x: x.mode()[0]).reset_index()
print("✅ Data imported and grouped by patient! Printing the first 5 elements")
max_voting.head(5)

# Some statistical analysis before implementing unsupervised techniques
df.info()
print("Number of cases per", max_voting.groupby('Diagnosis').size())

df_filtered = df.drop(['patient_id', 'seriesuid', 'Diagnosis', 'Malignancy', 'Calcification', 'InternalStructure', 'Lobulation', 'Margin', 'Sphericity', 'Spiculation', 'Subtlety', 'Texture'], axis = 1)
print("✅ Data filtered! Printing the first 5 elements")
print(df_filtered.head(5))

print("Count of missing values in each column:\n", df_filtered.isnull().sum())
print("✅ No missing values on the data!")

def show_boxplot(df):
    plt.rcParams['figure.figsize'] = [14,6]
    sns.boxplot(data = df, orient="v")
    plt.title("Outliers Distribution", fontsize = 16)
    plt.ylabel("Range", fontweight = 'bold')
    plt.xlabel("Attributes", fontweight = 'bold')
show_boxplot(df_filtered)

def remove_outliers(data):
   df = data.copy()
   for col in list(df.columns):
      Q1 = df[str(col)].quantile(0.05)
      Q3 = df[str(col)].quantile(0.95)
      IQR = Q3 - Q1
      lower_bound = Q1 - 1.5*IQR
      upper_bound = Q3 + 1.5*IQR
      df = df[(df[str(col)] >= lower_bound) & (df[str(col)] <= upper_bound)]
   return df
df_cleaned = remove_outliers(df_filtered)
print("✅ Data cleaned of outliers! Printing the first 5 elements")
print(df_cleaned.head(5))
show_boxplot(df_cleaned)


data_scaler = StandardScaler()
scaled_data = data_scaler.fit_transform(df_cleaned)
print("✅ Data scaled! Printing the shape and the data scaled")
print("Now the data shape is the following: ", scaled_data.shape)
print(scaled_data)

# Applying the hierarchical clustering algorithm
# https://www.datacamp.com/tutorial/introduction-hierarchical-clustering-python
# From all the pairwise distances between the items in the two clusters C1 and C2, the single linkage takes the distance between the clusters as the maximum distance. 
complete_clustering = linkage(scaled_data, method="complete", metric="euclidean")
# In the average linkage clustering, the distance between two given clusters C1 and C2 corresponds to the average distances between all pairs of items in the two clusters.
average_clustering = linkage(scaled_data, method="average", metric="euclidean")
# From all the pairwise distances between the items in the two clusters C1 and C2, the single linkage takes the distance between the clusters as the minimum distance
single_clustering = linkage(scaled_data, method="single", metric="euclidean")
print("✅ Data clustered with different methods")
# Complete clustering method
dendrogram(complete_clustering)
plt.title('Complete clustering')
plt.show()
# Average clustering method
dendrogram(average_clustering)
plt.title('Average clustering')
plt.show()
# Single clustering method
dendrogram(single_clustering)
plt.title('Single clustering')
plt.show()

# Correlations using Pearson's correlation
# Define features
feature_cols = [col for col in df_filtered]
# Initialize dictionary
correlations_per_class = {}
# Threshold for strong correlation
correlation_threshold = 0.5 
# Loop through each class
for class_name in df_filtered['Diagnosis_value'].unique():
    binary_target = (df_filtered['Diagnosis_value'] == class_name).astype(int)
    correlations = df_filtered[feature_cols].corrwith(binary_target)
    # Keep only features with correlation above threshold
    strong_correlations = correlations[correlations.abs() > correlation_threshold]
    # Sort them
    strong_correlations = strong_correlations.sort_values(ascending=False)
    correlations_per_class[class_name] = strong_correlations
# Show results
for class_name, correlations in correlations_per_class.items():
    print(f"\nTop Correlated Features for Class: {class_name}")
    # display(correlations.to_frame('Correlation').style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1))

# Perform ANOVA F-test
feature_cols.remove('Diagnosis_value')
print(feature_cols)
X = df_filtered[feature_cols]
y = df_filtered['Diagnosis_value']
f_scores, p_values = f_classif(X, y)
# Create a DataFrame with F-scores
anova_df = pd.DataFrame({
    'Feature': feature_cols,
    'F_score': f_scores,
    'p_value': p_values
}).sort_values(by='F_score', ascending=False)
# Display top 20 features
print(anova_df.head(10))
# Optional: Plot top features by F-score
plt.figure(figsize=(12, 6))
sns.barplot(x='F_score', y='Feature', data=anova_df.head(10), palette='mako')
plt.title('Top 5 Features by ANOVA F-Score')
plt.tight_layout()
plt.show()

# Extract GLCM texture features using the PyRadiomics library
