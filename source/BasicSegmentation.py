# -*- coding: utf-8 -*-
"""
Created on Fri May  2 20:10:25 2025

@author: sansg
"""

# BasicSegmentation.py
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.filters import threshold_otsu
from skimage import morphology as Morpho
from scipy.ndimage import filters as filt
from source.VolumeCutBrowser import VolumeCutBrowser
from source.NiftyIO import readNifty
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter, sobel, laplace


class BasicSegmentation:
    def __init__(self, input_data, data_folder=None):
        # Determine image path
        if isinstance(input_data, str):
            self.image_path = input_data
            filename = os.path.basename(input_data)
        elif hasattr(input_data, "__getitem__") and "filename" in input_data:
            if not data_folder:
                raise ValueError("Provide `data_folder` when using a DataFrame row.")
            filename = input_data["filename"]
            self.image_path = os.path.join(data_folder, "image", filename)
        else:
            raise TypeError(
                "`input_data` must be a filepath or a DataFrame row with `filename`."
            )

        # Build mask path (same filename, different folder)
        mask_folder = os.path.dirname(self.image_path).replace("image", "nodule_mask")
        self.mask_path = os.path.join(mask_folder, filename)

        # Check files exist
        for p in (self.image_path, self.mask_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"File not found: {p}")

        # Load both volumes
        self.data, _ = readNifty(self.image_path)  # CT VOI
        self.gt_mask, _ = readNifty(self.mask_path)  # GT binary mask

        print(f"Loaded image shape: {self.data.shape}")
        print(f"Loaded GT mask shape: {self.gt_mask.shape}")
        if self.data.shape != self.gt_mask.shape:
            raise ValueError("Image and GT mask shapes do not match!")

        # placeholders for processing results
        self.gauss = None
        self.med = None
        self.binarized = None
        self.opened = None
        self.closed = None

    def visualize_original(self):
        VolumeCutBrowser(self.data)

    def preprocess(self, sigma=1, med_size=3):
        print("Applying Gaussian and Median filters...")
        self.gauss = filt.gaussian_filter(self.data, sigma=sigma)
        self.med = filt.median_filter(self.gauss, size=med_size)
        return self.med

    def binarize(self):
        print("Applying Otsu thresholding...")
        Th = threshold_otsu(self.data)
        print(f"Otsu Threshold: {Th}")
        self.binarized = self.data > Th
        return self.binarized

    def postprocess(self, open_size=3, close_size=3):
        print("Applying morphological opening and closing...")
        se_op = Morpho.cube(open_size)
        se_cl = Morpho.cube(close_size)
        self.opened = Morpho.binary_opening(self.binarized, se_op)
        self.closed = Morpho.binary_closing(self.opened, se_cl)
        return self.closed

    def plot_histogram(self):
        print("Plotting intensity histogram...")
        fig, ax = plt.subplots()
        ax.hist(self.data.flatten(), bins=50, edgecolor="k")
        ax.set_title("Intensity Histogram")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Frequency")
        plt.show()
        return fig, ax

    def visualize_segmentation(self):
        VolumeCutBrowser(self.data, IMSSeg=self.binarized)

    def segment_with_kmeans(self, n_clusters=2):
        print("Segmenting with KMeans over filter bank features...")
        # Apply filter banks
        g_filtered = gaussian_filter(self.data, sigma=1)
        s_filtered = sobel(self.data)
        l_filtered = laplace(self.data)
        # Stack features
        features = np.stack([self.data, g_filtered, s_filtered, l_filtered], axis=-1)
        # Flatten for clustering
        flat_features = features.reshape(-1, features.shape[-1])
        # Run KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(flat_features)
        # Reshape back to volume
        self.kmeans_segmentation = labels.reshape(self.data.shape)
        print("KMeans segmentation complete.")
        return self.kmeans_segmentation
