import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage.measure import regionprops
import numpy as np

def display_roi_overlay(img, m, textcolor="white", alpha=0.5, ax=None, cmap="gray", mask_cmap="viridis"):
    """ Display an image with a labelled integer valued overlay
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 12))
    else:
        fig = ax.figure
    mask = m.astype(int)
    mask_values = np.unique(mask)
    mask_values = mask_values[mask_values!=0]
    mask = np.ma.masked_where(mask==0, mask)
        
    im = ax.imshow(img, cmap=cmap)
    mask_im = ax.imshow(mask, interpolation='none', alpha=alpha, cmap=mask_cmap, vmin=1, vmax = np.max(mask))
    if textcolor is not None:
        props = regionprops(mask)
        for idx, obj in enumerate(props):
            centroid = obj.centroid
            ax.text(centroid[1], centroid[0], str(mask_values[idx]), color=textcolor)
    return fig, ax, im, mask_im

def get_line_labels(lns):
    """ Get Matplotlib artist labels for 
    """
    return [l.get_label() for l in lns]

def tile_plots_conditions(condition_list, subplot_size, disp_titles=True):
    """ Generate subplots for the same graph over a large number of conditions
    INCOMPLETE
    """
    n_rows = int(np.ceil(np.sqrt(len(condition_list))))
    fig1, axes = plt.subplots(n_rows, n_rows, figsize=(subplot_size[0]*n_rows, subplot_size[1]*n_rows))
    if disp_titles:
        for idx, c in enumerate(condition_list):
            axes.ravel()[idx].set_title(c)
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
    
def stackplot(y, xvals=None, figsize_single=(12,1), ax=None, offset=None, cmap=None, flipud=False):
    if offset is None:
        offset = np.nanmax(np.nanmax(y, axis=1) - np.nanmin(y, axis=1))
    if ax is None:
        fig1, ax = plt.subplots(figsize=(figsize_single[0],figsize_single[1]*y.shape[0]))
    else:
        fig1 = ax.figure
    if cmap is None:
        cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    elif isinstance(cmap, mpl.colors.Colormap):
        cmap = [cmap(i/y.shape[0]) for i in range(y.shape[0])]
    cs = []
    yplot = y
    if flipud:
        yplot = np.flipud(y)
        cmap = np.flip(cmap)
    for i in range(y.shape[0]):
        color = cmap[i%len(cmap)]
        if xvals is None:
            ax.plot(yplot[i,:] + i*offset, color=color)
        else:
            ax.plot(xvals, yplot[i,:] + i*offset, color=color)
        cs.append(color)
    return fig1, ax, cs

def plot_img_scalebar(fig, ax, x0, y0, length_um, thickness_px, pix_per_um = 1, fontsize=9, \
                  color="white", unit="\mu m", yax_direction="down", text_pos="below", scale=0.7,
                 show_label=True):
    rect = patches.Rectangle((x0,y0), length_um*pix_per_um, thickness_px, color=color)
    ax.add_patch(rect)
    
    if show_label:
        plt.draw()
        label = r"$%d \mathrm{%s}$" % (length_um, unit)
        tx = ax.text(x0, y0, label, fontsize=fontsize, color=color)
        bb = tx.get_window_extent(renderer=fig.canvas.renderer)
        transf = ax.transData.inverted()
        bb_datacoords = bb.transformed(transf)
        bb_width = bb_datacoords.x1 - bb_datacoords.x0
        bb_height = bb_datacoords.y1 - bb_datacoords.y0

        x0_text = x0 + ((length_um*pix_per_um)-bb_width)/2

        if yax_direction == "down":
            diff = bb_height*scale
        elif yax_direction == "up":
            diff = -bb_height*scale
        if text_pos == "above":
            diff = -diff
        y0_text = y0 + diff
        tx.set_position((x0_text, y0_text))
        plt.draw()
        
def plot_trace_with_stim_bars(trace, stims, start_y, width, height, dt=1, figsize=(12,4), trace_color="C1", stim_color="blue", scale="axis", scalebar_params=None, axis=None):
    """ Plot a trace with rectangles indicating stimulation
    """
    if axis is None:
        fig1, ax1 = plt.subplots(figsize=(12,4))
    else:
        ax1 = axis
    ax1.plot(np.arange(len(trace))*dt, trace, color=trace_color)
    for st in stims:
        r = patches.Rectangle((st, start_y), width, height, color=stim_color)
        ax1.add_patch(r)
    
    if scale == "axis":
        pass
    elif scale == "bar":
        if scalebar_params is None:
            raise ValueError("scalebar_params required")
        plot_scalebars(ax1, scalebar_params)
    return fig1, ax1