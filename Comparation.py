# -*- coding: utf-8 -*-
"""
Created on Sat May  3 20:39:04 2025

@author: sansg
"""
from skimage import morphology as Morpho
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.filters import threshold_otsu
from skimage import morphology as Morpho
from scipy.ndimage import filters as filt
from VolumeCutBrowser import VolumeCutBrowser
from NiftyIO import readNifty

class comparation():
    def segmentation_metrics(pred_mask, gt_mask):
        pred = np.asarray(pred_mask).flatten().astype(bool)
        gt = np.asarray(gt_mask).flatten().astype(bool)

        TP = np.logical_and(pred, gt).sum()
        FP = np.logical_and(pred, ~gt).sum()
        FN = np.logical_and(~pred, gt).sum()
        TN = np.logical_and(~pred, ~gt).sum()

        dice = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
        jaccard = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

        return {
            'Dice': dice,
            'Jaccard': jaccard,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Precision': precision,
            'Accuracy': accuracy
            }
