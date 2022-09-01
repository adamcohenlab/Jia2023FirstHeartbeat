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
    if np.any(mask>0):
        mask_im = ax.imshow(mask, interpolation='none', alpha=alpha, cmap=mask_cmap, vmin=1, vmax = max(np.nanmax(mask), 1))
    else:
        mask_im = None
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

def tile_plots_conditions(condition_list, subplot_size, n_rows = None, disp_titles=True):
    """ Generate subplots for the same graph over a large number of conditions
    INCOMPLETE
    """
    n_plots = len(condition_list)
    if n_rows is None:
        n_rows = int(np.ceil(np.sqrt(n_plots)))
        if n_plots% n_rows == 0:
            n_cols = n_plots//n_rows
        else:
            n_cols = n_plots//n_rows + 1
    else:
        n_cols = n_plots//n_rows
    
    fig1, axes = plt.subplots(n_rows, n_cols, figsize=(subplot_size[0]*n_cols, subplot_size[1]*n_rows))
    if n_rows ==1:
        axes = np.array([axes])
    axes = axes.ravel()
    if disp_titles:
        for idx, c in enumerate(condition_list):
            axes[idx].set_title(c)
    return fig1, axes

def plot_scalebars(ax, scalebar_params, time_unit="s", pct_f=False, newline=False):
    """ Plot ephys style scalebars
    """
    
    ax.set_axis_off()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bbox = ax.get_window_extent()
    aspect = np.abs(xlim[0] - xlim[1])/np.abs(ylim[0] - ylim[1])*bbox.height/bbox.width
    ampl_scale = scalebar_params["ampl_scale"]

    r1 = patches.Rectangle((scalebar_params["corner_x"], scalebar_params["corner_y"]), \
                           scalebar_params["time_scale"], scalebar_params["thickness"], color="black")
    r2 = patches.Rectangle((scalebar_params["corner_x"], 
                            scalebar_params["corner_y"]-ampl_scale+scalebar_params["thickness"]), \
                           scalebar_params["thickness"]*aspect, ampl_scale , color="black")

    xlabel_offset_x = scalebar_params["xlabel_offset_x"]
    xlabel_offset_y = scalebar_params["xlabel_offset_y"]
    ylabel_offset_y = scalebar_params["ylabel_offset_y"]
    ylabel_offset_x = scalebar_params["ylabel_offset_x"]
    if "ylabel" in scalebar_params:
        if pct_f:
            f_label = scalebar_params["ylabel"] % (np.round(scalebar_params["ampl_scale"]*100))
        else:
            f_label = scalebar_params["ylabel"] % scalebar_params["ampl_scale"]
    else:
        if pct_f:
            f_label = (r"$%d\%%$" "\n" r"$\Delta F/F$") % (np.round(scalebar_params["ampl_scale"]*100)) 
        else:
            f_label = r"$%.2f\Delta F/F$" % scalebar_params["ampl_scale"]
    print(f_label)
    
    ax.text(scalebar_params["corner_x"] + xlabel_offset_x, scalebar_params["corner_y"] + xlabel_offset_y, "%d%s" % (scalebar_params["time_scale"], time_unit), size=scalebar_params["fontsize"])
    ax.text(scalebar_params["corner_x"] + ylabel_offset_x, scalebar_params["corner_y"] + ylabel_offset_y - scalebar_params["ampl_scale"], \
            f_label, size=scalebar_params["fontsize"], rotation=90, linespacing=0.75)
    ax.add_patch(r1)
    ax.add_patch(r2)
    
def stackplot(y, xvals=None, figsize_single=(12,1), ax=None, offset=None, cmap=None, flipud=False, **plot_args):
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
            ax.plot(yplot[i,:] + i*offset, color=color, **plot_args)
        else:
            ax.plot(xvals, yplot[i,:] + i*offset, color=color, **plot_args)
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
        
def add_stims(ax, stims, start_y, width, height, color="C0"):
    """ Add stim markings to plot
    """
    for st in stims:
        r = patches.Rectangle((st, start_y), width, height, color=stim_color)
        ax1.add_patch(r)
        
def plot_trace_with_stim_bars(trace, stims, start_y, width, height, t=None, figsize=(12,4), trace_color="C1", stim_color="blue", scale="axis", scalebar_params=None, axis=None):
    """ Plot a trace with rectangles indicating stimulation
    """
    if axis is None:
        fig1, ax1 = plt.subplots(figsize=(12,4))
    else:
        ax1 = axis
        fig1 = ax1.get_figure()
    if t is None:
        ax1.plot(np.arange(len(trace)), trace, color=trace_color)
    else:
        ax1.plot(t, trace, color=trace_color)
    
    add_stims(ax1, stims, start_y, width, height, color=stim_color)

    
    if scale == "axis":
        pass
    elif scale == "bar":
        if scalebar_params is None:
            raise ValueError("scalebar_params required")
        plot_scalebars(ax1, scalebar_params)
    return fig1, ax1


def plot_image_roi_traces(masks, figsize_single=(12,4), img_size=4, t=None, traces=None, image=None):
    """ Plot traces of ROIs along with rois and images, - optional manual traces
    """
    n_rois = masks.shape[0]
    
    fig1, axes = plt.subplots(n_rois, 2, figsize=(figsize_single[0], figsize_single[1]*n_rois), \
                              gridspec_kw={"width_ratios": [figsize_single[0]-img_size, img_size]})
    if traces is None:
        traces = np.array([images.image_to_trace(image, mask[idx]) for idx in range(n_rois)])
    if t is None:
        t = traces.shape[1]
    for idx in range(n_rois):
        ax = axes[idx,0]
        ax.plot(t, traces[idx], color="C%d" % idx)
        axes[idx,1].imshow(masks[idx])
        axes[idx,1].set_axis_off()
        
    plt.tight_layout()
    return fig1, axes

def montage_along_axis(all_traces, axis, xvals=None, labels=None):
    """ for the given axis, make plots that scan across it for each value of the other axes
    """ 
    n_traces = all_traces.shape[axis]
    rearranged_traces = np.moveaxis(all_traces, axis, \
                        -2)
    if labels is None:
        labels = np.meshgrid([np.arange(dim) for dim in list(rearranged_traces.shape[:-2])]).ravel()
    rearranged_traces = rearranged_traces.reshape(-1, n_traces, all_traces.shape[-1])
    
    assert rearranged_traces.shape[0] == len(labels)
    figures = []
    axes = []
    for i in range(rearranged_traces.shape[0]):
        fig1, ax1 = plt.subplots(figsize= (10, 1.5*n_traces))
        stackplot(rearranged_traces[i], xvals=xvals, ax=ax1, flipud=True)
        ax1.set_title(labels[i])
        figures.append(fig1)
        axes.append(ax1)
    return figures, axes

def plot_wave_analysis(snr, rd, Tsmoothed, Tsmoothed_dv, divergence, v, title):
    """ Plot quality check plots for activation map generation
    """
    fig1, axes = plt.subplots(2,3, figsize=(14,8))
    plt.axis('off')
    axes = axes.ravel()
    fig1.suptitle(title)
    axes[0].set_title("SNR")
    i0 = axes[0].imshow(snr)
    plt.colorbar(i0, ax=axes[0], label= r"SNR (dB)")


    axes[1].set_title("Smoothed Activation Map")
    i1 = axes[1].imshow(Tsmoothed)
    axes[1].plot(rd[2], rd[3],"kx")
    axes[1].plot(rd[4], rd[5], "wx")
    plt.colorbar(i1, ax=axes[1], label = r"$T_{1/2} (\mathrm{ms})$")

    axes[2].set_title("Smoothed Activation Map (dV/dt)")
    i2 = axes[2].imshow(Tsmoothed_dv)
    plt.colorbar(i2, ax=axes[2], label = r"$T_{1/2} (\mathrm{ms})$")
    axes[3].set_title("Wave Direction")

    stride = 8
    X, Y = np.meshgrid(np.arange(divergence.shape[0]), np.arange(divergence.shape[1]))
    abs_vel = np.linalg.norm(v, axis=0)
    axes[3].quiver(X[::stride,::stride],Y[::stride,::stride], (v[1]/abs_vel)[::stride,::stride],\
                   (v[0]/abs_vel)[::stride,::stride], angles="xy")
    i3 = axes[3].imshow(divergence, \
                        vmin = np.percentile(divergence[~np.isnan(divergence)], 0), \
                        vmax = np.percentile(divergence[~np.isnan(divergence)], 100))
    axes[3].plot(rd[4], rd[5], "rx")
    plt.colorbar(i3, ax=axes[3], label = r"$\nabla \cdot v/|v|$")


    vx_filt = np.copy(v[0])
    vx_filt[np.abs(vx_filt)>np.percentile(np.abs(vx_filt[~np.isnan(vx_filt)]),95)] = np.nan
    vy_filt = np.copy(v[1])
    vy_filt[np.abs(vy_filt)>np.percentile(np.abs(vy_filt[~np.isnan(vy_filt)]),95)] = np.nan
    abs_vel_filt = (vx_filt**2 + vy_filt**2)**0.5

    axes[4].set_title("Wave Speed")
    i4 = axes[4].imshow(abs_vel_filt)
    plt.colorbar(i4, ax=axes[4], label = r"$|v| \mathrm{(\mu m/s)}$")
    plt.axis('on')

    axes[5].set_title("Wave Speed histogram")
    axes[5].hist(abs_vel_filt.ravel(), bins=50, density=True)
    ymin, ymax = axes[5].get_ylim()
    axes[5].text(rd[0], ymax*0.8, "mean = %d" % rd[0])
    axes[5].text(rd[1], ymax*0.9, "median = %d" % rd[1], color="red")
    axes[5].vlines(rd[0], ymin, ymax*0.8, color="black")
    axes[5].vlines(rd[1], ymin, ymax*0.9, color="red")
    axes[5].set_xlabel(r"$|v| \mathrm{(\mu m/s)}$")
    axes[5].set_ylabel("PDF")
    axes[5].get_yaxis().set_visible(False)

    axes[4].set_axis_off()
    axes[3].set_axis_off()
    axes[2].set_axis_off()
    axes[1].set_axis_off()

    plt.tight_layout()
    return fig1, axes