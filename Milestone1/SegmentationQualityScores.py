# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 12:09:57 2018

@author: Debora Gil, Guillermo Torres

Quality Measures of an automatic segmentation computed from
a mask of the object (ground truth)
Two types of measures are implemented:
    1. Volumetric (dice, voe, relvoldiff) compute differences and
    similarities between the two volumes. They are similar to precision and
    recall.
    2. Distance-base (AvDist, MxDist) compare volume surfaces
    in terms of distance between segmentation and ground truth.
    Average distances, AvDist, is equivalent to Euclidean distance between
    volumes, while Maximum distance, MxDist, is the infinite norm and detects
    puntual deviations between surfaces

References:
    1. T. Heimann et al, Comparison and Evaluation of Methods for
Liver Segmentation From CT Datasets, IEEE Trans Med Imag, 28(8),2009
"""

import numpy as np

##from scipy.ndimage.morphology import distance_transform_edt as bwdist
from scipy.ndimage import distance_transform_edt as bwdist


def DICE(Seg, GT):
    dice = np.sum(Seg[np.nonzero(GT)]) * 2.0 / (np.sum(Seg) + np.sum(GT))
    return dice

def VOE(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return 1.0 - (intersection / union)
def RelVolDiff(Seg, GT):
    RelVolDiff = (np.sum(Seg) - np.sum(GT)) / np.sum(Seg)
    return RelVolDiff


def DistScores(Seg, GT):
    # Distances to Segmented Volume
    DistSegInt = bwdist(Seg)
    DistSegExt = bwdist(1 - Seg)
    DistSeg = np.maximum(DistSegInt, DistSegExt)
    # Distances to GT Volume
    DistGTInt = bwdist(GT)
    DistGTExt = bwdist(1 - GT)
    DistGT = np.maximum(DistGTInt, DistGTExt)

    BorderSeg = ((DistSegInt < 1) + (DistSegInt > 1)) == 0
    BorderGT = ((DistGTInt < 1) + (DistGTInt > 1)) == 0
    DistAll = np.concatenate((DistSeg[BorderGT], DistGT[BorderSeg]), axis=0)

    DistAvg = np.mean(DistAll)
    DistMx = np.max(DistAll)

    return DistAvg, DistMx
