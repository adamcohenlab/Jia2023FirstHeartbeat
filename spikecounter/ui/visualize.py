import matplotlib.pyplot as plt
from matplotlib import patches
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

def plot_scalebars(ax, scalebar_params):
    """ Plot ephys style scalebars
    """
    
    ax.set_axis_off()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bbox = ax.get_window_extent()
    aspect = np.abs(xlim[0] - xlim[1])/np.abs(ylim[0] - ylim[1])*bbox.height/bbox.width

    r1 = patches.Rectangle((scalebar_params["corner_x"], scalebar_params["corner_y"]), \
                           scalebar_params["time_scale"], scalebar_params["thickness"], color="black")
    r2 = patches.Rectangle((scalebar_params["corner_x"], 
                            scalebar_params["corner_y"]-scalebar_params["ampl_scale"]+scalebar_params["thickness"]), \
                           scalebar_params["thickness"]*aspect, scalebar_params["ampl_scale"], color="black")

    offset = scalebar_params["thickness"]*2
    ax.text(scalebar_params["corner_x"], scalebar_params["corner_y"] + offset, "%d s" % scalebar_params["time_scale"], size=scalebar_params["fontsize"])
    ax.text(scalebar_params["corner_x"] - offset*aspect*2 , scalebar_params["corner_y"] + offset - scalebar_params["ampl_scale"], "%.2f F" % scalebar_params["ampl_scale"], size=scalebar_params["fontsize"], rotation=90)
    ax.add_patch(r1)
    ax.add_patch(r2)
    
def stackplot(x, figsize_single=(12,1)):
    offset = np.max(np.max(x, axis=0) - np.min(x, axis=0))
    fig1, ax1 = plt.subplots(figsize=(figsize_single[0],figsize_single[1]*x.shape[1]))
    for i in range(x.shape[1]):
        ax1.plot(x[:,i] + i*offset)
    return fig1, ax1