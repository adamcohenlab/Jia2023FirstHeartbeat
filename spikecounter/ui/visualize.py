import matplotlib.pyplot as plt
from matplotlib import patches
from skimage.measure import regionprops
import numpy as np

def display_roi_overlay(img, mask, textcolor="white", alpha=0.5, ax=None, cmap="gray", mask_cmap="viridis"):
    """ Display an image with a labelled integer valued overlay
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 12))
    else:
        fig = ax.figure
        
    props = regionprops(mask)
    mask = np.ma.masked_where(mask==0, mask)
        
    im = ax.imshow(img, cmap=cmap)
    ax.imshow(mask, interpolation='none', alpha=alpha, cmap=mask_cmap)
    if textcolor is not None:
        for idx, obj in enumerate(props):
            centroid = obj.centroid
            ax.text(centroid[1], centroid[0], str(idx+1), color=textcolor)
    return fig, ax, im

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

def plot_scalebars(ax, scalebar_params, pct_f=False):
    """ Plot ephys style scalebars
    """
    
    ax.set_axis_off()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bbox = ax.get_window_extent()
    aspect = np.abs(xlim[0] - xlim[1])/np.abs(ylim[0] - ylim[1])*bbox.height/bbox.width
    ampl_scale = np.round(scalebar_params["ampl_scale"], decimals=2)

    r1 = patches.Rectangle((scalebar_params["corner_x"], scalebar_params["corner_y"]), \
                           scalebar_params["time_scale"], scalebar_params["thickness"], color="black")
    r2 = patches.Rectangle((scalebar_params["corner_x"], 
                            scalebar_params["corner_y"]-ampl_scale+scalebar_params["thickness"]), \
                           scalebar_params["thickness"]*aspect, ampl_scale , color="black")

    xlabel_offset_x = scalebar_params["xlabel_offset_x"]
    xlabel_offset_y = scalebar_params["xlabel_offset_y"]
    ylabel_offset_y = scalebar_params["ylabel_offset_y"]
    ylabel_offset_x = scalebar_params["ylabel_offset_x"]
    
    if pct_f:
        f_label = r"$%d\%%F$" % (np.round(scalebar_params["ampl_scale"]*100))
    else:
        f_label = r"$%.2f\Delta F/F$" % scalebar_params["ampl_scale"]
    print(f_label)
    
    ax.text(scalebar_params["corner_x"] + xlabel_offset_x, scalebar_params["corner_y"] + xlabel_offset_y, "%ds" % scalebar_params["time_scale"], size=scalebar_params["fontsize"])
    ax.text(scalebar_params["corner_x"] + ylabel_offset_x, scalebar_params["corner_y"] + ylabel_offset_y - scalebar_params["ampl_scale"], \
            f_label, size=scalebar_params["fontsize"], rotation=90)
    ax.add_patch(r1)
    ax.add_patch(r2)
    
def stackplot(y, xvals=None, figsize_single=(12,1), ax=None):
    offset = np.max(np.max(y, axis=1) - np.min(y, axis=1))
    if ax is None:
        fig1, ax = plt.subplots(figsize=(figsize_single[0],figsize_single[1]*y.shape[0]))
    for i in range(y.shape[0]):
        if xvals is None:
            ax.plot(y[i,:] + i*offset)
        else:
            ax.plot(xvals, y[i,:] + i*offset)
    return fig1, ax