#! /usr/bin/python3
import numpy as np
import skimage.io as skio
import scipy.ndimage as ndimage
import skimage.filters as filters
import skimage.morphology as morph
from skimage import img_as_ubyte

import pandas as pd


groups = ["control", "gtACR"]
n_embryos = [5, 5]

def get_mask(img):
    threshold = filters.threshold_otsu(img)
    mask = img > threshold
    strel = morph.disk(3)
    strel = strel.reshape([1] + list(strel.shape))
    mask = ndimage.binary_closing(mask, structure=strel)
    mask = ndimage.binary_opening(mask, structure=strel)
    return mask


for idx, g in enumerate(groups):
    for i in range(n_embryos[idx]):
        img = skio.imread("%s_E%d_expression_eCFP.tif" %(g, i+1))
        C1 = img[:, 0, :, :]
        C2 = img[:, 1, :, :]
        mask = get_mask(C1)
        background_c1 = np.mean(C1[~mask])
        background_c2 = np.mean(C2[~mask])
        print(np.mean(C1[mask]-background_c1))
        # print(np.mean(C1[mask]-background_c1)/np.mean(C2[mask]-background_c2))
        skio.imsave("%s_E%d_mask.tif" %(g, i+1), img_as_ubyte(mask))