import matplotlib.pyplot as plt
from skimage.measure import regionprops
import numpy as np

def display_roi_overlay(img, mask, textcolor="white", alpha=0.5):
    props = regionprops(mask)
    mask = np.ma.masked_where(mask==0, mask)
    fig1, ax1 = plt.subplots(figsize=(10, 12))
    ax1.imshow(img, cmap="gray")
    ax1.imshow(mask, interpolation='none', alpha=alpha)
    for idx, obj in enumerate(props):
        centroid = obj.centroid
        ax1.text(centroid[1], centroid[0], str(idx+1), color=textcolor)
    return fig1, ax1