import matplotlib.pyplot as plt
from skimage.measure import regionprops
import numpy as np

def display_roi_overlay(img, mask, textcolor="white", alpha=0.5, ax=None):
    """ Display an image with a labelled integer valued overlay
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 12))
    else:
        fig = ax.figure
        
    props = regionprops(mask)
    mask = np.ma.masked_where(mask==0, mask)
        
    ax.imshow(img, cmap="gray")
    ax.imshow(mask, interpolation='none', alpha=alpha)
    for idx, obj in enumerate(props):
        centroid = obj.centroid
        ax.text(centroid[1], centroid[0], str(idx+1), color=textcolor)
    return fig, ax

def get_line_labels(lns):
    return [l.get_label() for l in lns]

def tile_plots_conditions(condition_list, subplot_size):
    """ Generate subplots for the same graph over a large number of conditions
    INCOMPLETE
    """
    n_rows = int(np.ceil(np.sqrt(len(condition_list))))
    fig1, axes = plt.subplots(n_rows, n_rows, figsize=(subplot_size[0]*n_rows, subplot_size[1]*n_rows))
    if n_rows ==1:
        axes = np.array([axes])
    axes = axes.ravel()
    return fig1, axes