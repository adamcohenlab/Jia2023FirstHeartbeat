""" Functions for analyzing traces of fluorescence or voltage.
"""
import os
from pathlib import Path
from typing import Union, Iterable, Tuple, Collection, Dict
from numpy import typing as npt

import pandas as pd
import numpy as np
from scipy import stats, optimize, signal, interpolate, ndimage
from skimage import morphology

import matplotlib.pyplot as plt
from matplotlib import colors, patches, axes, figure

from statsmodels.stats.diagnostic import lilliefors
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
import igraph
import pywt

from datetime import datetime
from ..ui import visualize
from .. import utils
from . import stats as sstats


plt.style.use(Path(__file__).parent / "../../config" / "bio_pubs_presentation.mplstyle")


def calculate_lags(pks, start_idx, end_idx, ref_indices, plot=True, forward_only=True):
    if len(pks) == 0:
        return np.array([]), np.array([]), np.array([]), 0

    stim_pks = pks[(pks >= start_idx) * (pks < end_idx)]
    mean_freq = len(stim_pks) / (end_idx - start_idx)
    pairwise_diff = np.subtract.outer(stim_pks, ref_indices)
    print(pairwise_diff.shape)
    if forward_only:
        pairwise_diff[pairwise_diff < 0] = np.iinfo(pairwise_diff.dtype).max
    mindist_indices = np.argmin(np.abs(pairwise_diff), axis=1)
    print(mindist_indices)
    mindist = pairwise_diff[np.arange(len(stim_pks)), mindist_indices]

    associated_stims = ref_indices[mindist_indices]

    return stim_pks, mindist, associated_stims


def crosstalk_mask_to_stim_index(
    crosstalk_mask: npt.NDArray[np.bool_], pos: str = "start"
) -> npt.NDArray[np.int64]:
    """Convert detected spikes from crosstalk into an index for spike-triggered averaging.

    Use either upward or downward edge.

    Args:
        crosstalk_mask: 1D array of boolean values indicating whether a spike was detected.
    Returns:
        stims: 1D array of indices corresponding to the start or end of each spike.
    """
    diff_mask = np.diff(crosstalk_mask.astype(int))
    if pos == "start":
        stims = np.argwhere(diff_mask == 1).ravel()
    elif pos == "end":
        stims = np.argwhere(diff_mask == -1).ravel() + 1
    else:
        raise ValueError("pos must be 'start' or 'end'")
    return stims


def find_stim_starts(
    crosstalk_mask: npt.NDArray[np.bool_],
    expected_period: float,
    dt: float = 1,
    tol: float = 0.9,
) -> npt.NDArray[np.floating]:
    """Find the start of a stimulation period based on cross-excitation from blue channel and known
    expected period.

    Args:
        crosstalk_mask: 1D array of boolean values indicating whether a spike was detected.
        expected_period: Expected period between stimulation pulses in seconds.
        dt: Sampling interval in seconds.
        tol: Fraction of expected period to use as tolerance.
    Returns:
        1D array of times in seconds corresponding to the start of each stimulation period.
    """
    stim_locs = crosstalk_mask_to_stim_index(crosstalk_mask, pos="start")
    ts = dt * np.arange(crosstalk_mask.size)
    rising_ts = ts[stim_locs + 1]

    curr_stim = rising_ts[0]
    valid_stims = np.zeros_like(rising_ts)
    valid_stims[0] = 1
    for i in range(1, len(rising_ts)):
        if rising_ts[i] - curr_stim > tol * expected_period:
            valid_stims[i] = 1
            curr_stim = rising_ts[i]
    return rising_ts[valid_stims.astype(bool)]


def plot_trace_with_stim_bars(
    trace,
    stims,
    start_y,
    width,
    height,
    dt=1,
    figsize=(12, 4),
    trace_color="C1",
    stim_color="blue",
    scale="axis",
    scalebar_params=None,
    axis=None,
):
    """Plot a trace with rectangles indicating stimulation"""
    if axis is None:
        fig1, ax1 = plt.subplots(figsize=(12, 4))
    else:
        ax1 = axis
    ax1.plot(np.arange(len(trace)) * dt, trace, color=trace_color)
    for st in stims:
        r = patches.Rectangle((st, start_y), width, height, color=stim_color)
        ax1.add_patch(r)

    if scale == "axis":
        pass
    elif scale == "bar":
        if scalebar_params is None:
            raise ValueError("scalebar_params required")
        visualize.plot_scalebars(ax1, scalebar_params)
    return fig1, ax1


def correct_photobleach(
    trace,
    method="linear",
    nsamps=None,
    plot=False,
    return_params=False,
    invert=False,
    **cost_function_params
):
    """Correct trace for photobleaching"""
    tidx = np.arange(len(trace))
    if method == "linear":
        slope, _, _, _, _ = stats.linregress(tidx, y=trace)
        photobleach = slope * tidx
        corrected_trace = trace - photobleach
    elif method == "localmin":
        """From Hochbaum 2014 Nat. Methods"""
        if nsamps is None:
            raise ValueError("nsamps required if mode is localmin")
        kernel = np.ones(nsamps)
        photobleach = morphology.erosion(trace, kernel)
        photobleach_padded = np.pad(
            photobleach, (nsamps - 1) // 2, mode="mean", stat_length=(nsamps - 1) // 2
        )
        photobleach = signal.convolve(photobleach_padded, kernel / nsamps, mode="valid")
        corrected_trace = trace / photobleach
    elif method == "monoexp":
        tpoints = np.arange(len(trace))

        def expon_below(x, a=1, b=1):
            y = x[0] * np.exp(tpoints * x[1]) + x[2]
            # soft_constraint = a*np.exp(b*(y - trace))
            soft_constraint = b * np.sum((np.maximum(y - trace, 0)) ** a)
            # soft_constraint = a*np.sum(np.log(np.maximum(y-trace,0)+1))
            cost = np.sum((y - trace) ** 2 + soft_constraint)
            return cost

        def expon_above(x, a=1, b=1):
            y = x[0] * np.exp(tpoints * x[1]) + x[2]
            cost = np.sum((y - trace) ** 2 + a * np.exp(b * (trace - y)))
            return cost

        pct1, pct2, med = np.percentile(trace, [5, 85, 50])
        guess_tc = np.log((pct2 - med) / (pct2 - pct1)) / (len(trace) / 2)

        baseline = pct1 - (pct2 - pct1) * 0.5
        p0 = [(pct2 - pct1) * 1.5, guess_tc, pct1]
        if invert:
            p0[2] = np.max(trace)
            res = optimize.minimize(
                lambda x: expon_above(x, **cost_function_params),
                p0,
                bounds=[(0, np.inf), (-np.inf, 0), (-np.inf, np.inf)],
            )
        else:
            res = optimize.minimize(
                lambda x: expon_below(x, **cost_function_params),
                p0,
                bounds=[
                    (pct2 - pct1, np.inf),
                    (p0[1] * 1.5, p0[1] * 0.1),
                    (baseline, med),
                ],
            )
        popt = res.x
        if plot:
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            ax1.scatter(tpoints, trace, s=0.8, alpha=0.5)
            ax1.plot(
                tpoints, popt[0] * np.exp(popt[1] * tpoints) + popt[2], color="red"
            )
            ax1.text(
                10,
                popt[2] + popt[0],
                "%.2E exp(%.2E t) + %.2E" % (popt[0], popt[1], popt[2]),
            )
        photobleach = popt[0] * np.exp(tpoints * popt[1])
        corrected_trace = trace - photobleach
        if return_params:
            return corrected_trace, photobleach, popt
    elif method == "biexp":
        tpoints = np.arange(len(trace))

        def expon_below(x):
            y = x[0] * np.exp(tpoints * x[1]) + x[2] * np.exp(tpoints * x[3]) + x[4]
            cost = np.sum((y - trace) ** 2 + 2 * np.exp(y - trace))
            return cost

        def expon_above(x):
            y = x[0] * np.exp(tpoints * x[1]) + x[2] * np.exp(tpoints * x[3]) + x[4]
            cost = np.sum((y - trace) ** 2 + 2 * np.exp(trace - y))
            return cost

        guess_tc = -(np.percentile(trace, 95) / np.percentile(trace, 5)) / len(trace)
        guess_tc2 = guess_tc * 5
        p0 = [
            (np.max(trace) - np.min(trace)) * 0.95,
            guess_tc,
            (np.max(trace) - np.min(trace)) * 0.05,
            guess_tc2,
            np.min(trace) * 0.8,
        ]
        if invert:
            p0[4] = np.max(trace)
            res = optimize.minimize(
                lambda x: expon_above(x, **cost_function_params),
                p0,
                bounds=[
                    (0, np.inf),
                    (-np.inf, 0),
                    (0, np.inf),
                    (-np.inf, guess_tc),
                    (-np.inf, np.inf),
                ],
            )
        else:
            res = optimize.minimize(
                lambda x: expon_below(x, **cost_function_params),
                p0,
                bounds=[
                    (0, np.inf),
                    (-np.inf, 0),
                    (0, np.inf),
                    (-np.inf, guess_tc),
                    (-np.inf, np.inf),
                ],
            )
        popt = res.x
        if plot:
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            ax1.scatter(tpoints, trace, s=0.8, alpha=0.5)
            ax1.plot(
                tpoints,
                popt[0] * np.exp(popt[1] * tpoints)
                + popt[2] * np.exp(popt[3] * tpoints)
                + popt[4],
                color="red",
            )
            ax1.text(
                10,
                popt[4] + popt[0] + popt[2],
                "%.2E exp(%.2E t) + %.2E exp(%.2E t) + %.2E"
                % (popt[0], popt[1], popt[2], popt[3], popt[4]),
            )
        photobleach = popt[0] * np.exp(tpoints * popt[1]) + popt[2] * np.exp(
            tpoints * popt[3]
        )
        corrected_trace = trace - photobleach
        if return_params:
            return corrected_trace, photobleach, popt
    else:
        raise ValueError("Not implemented")
    return corrected_trace, photobleach


def intensity_to_dff(
    intensity: npt.ArrayLike,
    percentile_threshold: float = 10,
    axis: int = 0,
    moving_average: bool = False,
    window: Union[int, None] = None,
) -> npt.NDArray:
    """Calculate DF/F from intensity counts

    Args:
        intensity: Intensity counts
        percentile_threshold: Percentile threshold for baseline. Defaults to 10.
        axis: Axis to calculate baseline. Defaults to 0.
        moving_average: Use moving average for baseline. Defaults to False.
        window: Window length for moving average. Defaults to None.
    Returns:
        Delta F/F of trace
    """
    if moving_average:
        if window is None:
            raise ValueError("Window length required")
        kernel = np.ones(window) / window
        ma = signal.fftconvolve(intensity, kernel, axes=axis, mode="same")
        dFF = intensity / ma
    else:
        if window is not None:
            raise ValueError("Windowed percentile not yet implemented")

        percentile_10 = np.percentile(intensity, percentile_threshold, axis=axis)[
            ..., None
        ]
        masked_intensity = np.ma.masked_array(intensity, intensity < percentile_10)

        baseline = np.nanmean(masked_intensity, axis=axis)[..., None]
        dFF = (intensity - baseline) / baseline
        dFF = np.ma.getdata(dFF)
    return dFF


def standard_lp_filter(raw, norm_thresh=0.5):
    b, a = signal.butter(5, norm_thresh)
    intensity = signal.filtfilt(b, a, raw)
    mean_freq = 2.0
    # b, a = signal.butter(5, [mean_freq-0.2, mean_freq+0.2], btype="bandstop", fs=10.2)
    # intensity = signal.filtfilt(b, a, intensity)
    return intensity


def analyze_peaks(
    trace: npt.NDArray,
    prominence: Union[str, float] = "auto",
    wlen: int = 400,
    threshold: Union[str, float] = 0,
    f_s: float = 1,
    auto_prom_scale: float = 0.5,
    auto_thresh_scale: float = 0.5,
    auto_thresh_pct: float = 95,
    min_prom: float = 0,
    min_width: Union[int, None] = None,
    max_width: Union[int, None] = None,
    baseline_start: int = 0,
    baseline_duration: int = 1000,
    excl: int = 0,
    return_full: bool = False,
):
    """Analyze peaks within a given trace and return the following statistics, organized in a pandas
    DataFrame.

    Args:
        trace: 1D array of trace to analyze
        prominence: prominence of peaks to detect. If "auto", will use 98th percentile of trace
            times auto_prom_scale. If "snr", will use standard deviation of baseline times
            auto_prom_scale. Defaults to "auto".
        wlen: window length for peak detection. Defaults to 400.
        threshold: threshold for peak detection. If "auto", will use 95th percentile of trace
            times auto_thresh_scale. Defaults to 0.
        f_s: sampling frequency. Defaults to 1.
        auto_prom_scale: scale factor for auto prominence. Defaults to 0.5.
        auto_thresh_scale: scale factor for auto threshold. Defaults to 0.5.
        auto_thresh_pct: percentile for auto threshold. Defaults to 95.
        min_prom: minimum prominence for peak detection. Defaults to 0.
        min_width: minimum width for peak detection. Defaults to None.
        max_width: maximum width for peak detection. Defaults to None.
        baseline_start: start of baseline for SNR calculation. Defaults to 0.
        baseline_duration: duration of baseline for SNR calculation. Defaults to 1000.
        excl: number of points to exclude from peak detection. Defaults to 0.
        return_full: return threshold and prominence parametesr in addition to peak finding results.
            Defaults to False.
    Returns:
        DataFrame of peak statistics
        (Optional) threshold, prominence parameters
    """

    if prominence == "auto":
        p = np.nanpercentile(trace, 98) * auto_prom_scale
        p = max(p, min_prom)
    elif prominence == "snr":
        noise = np.std(
            trace.ravel()[baseline_start : baseline_start + baseline_duration]
        )
        print(noise)
        p = noise * auto_prom_scale
    else:
        p = prominence
    if threshold == "auto":
        t = np.nanpercentile(trace, auto_thresh_pct) * auto_thresh_scale
    else:
        t = threshold

    peaks, properties = signal.find_peaks(
        trace,
        prominence=p,
        height=t,
        wlen=wlen,
        width=(min_width, max_width),
        rel_height=0.5,
    )

    prominences = np.array(properties["prominences"])
    prominences = prominences[peaks > excl * f_s]
    peaks = peaks[peaks > excl * f_s]
    fwhm = signal.peak_widths(trace, peaks, rel_height=0.5, wlen=wlen)[0] / f_s
    isi = np.diff(peaks, append=np.nan) / f_s
    if len(peaks) == 0:
        if return_full:
            return None, p, t
        return None
    res = (
        pd.DataFrame(
            {"peak_idx": peaks, "prominence": prominences, "fwhm": fwhm, "isi": isi}
        )
        .sort_values("peak_idx")
        .reset_index(drop=True)
    )

    if return_full:
        return res, p, t
    else:
        return res


def first_trough_exp_fit(st_traces, before, after, f_s=1):
    """Fit an exponential to a detected peak up to the first trough"""
    if st_traces.shape[0] == 0:
        return pd.Series(
            {"alpha": np.nan, "c": np.nan, "alpha_err": np.nan, "c_err": np.nan}
        )

    sta = np.nanmean(st_traces, axis=0)
    relmin, _ = signal.find_peaks(1 - sta, height=0.5)

    if len(relmin) > 0:
        relmin = relmin[0]
    else:
        relmin = len(sta) - before
    ts = np.tile(np.arange(relmin) / f_s, st_traces.shape[0])
    ys = st_traces[:, before : relmin + before].ravel()

    popt, pcov = optimize.curve_fit(
        lambda x, alpha, c: np.exp(-alpha * x) + c, ts, ys, p0=[0.6, 0.2]
    )

    alpha = popt[0]
    c = popt[1]
    alpha_err = np.sqrt(pcov[0, 0])
    c_err = np.sqrt(pcov[1, 1])

    return pd.Series({"alpha": alpha, "c": c, "alpha_err": alpha_err, "c_err": c_err})


def remove_stim_crosstalk(
    trace,
    method="zscore",
    side="both",
    freq_cutoff=0.1,
    threshold=2,
    plot=False,
    fs=1,
    mode="remove",
    max_width=10,
    lpad=0,
    rpad=0,
    expected_stim_width=None,
    fixed_width_remove=False,
):
    """Remove optical crosstalk from e.g. channelrhodopsin stimulation"""
    # Identify outliers and turn into mask (or directly median filter)
    if method == "zscore":
        zsc = stats.zscore(trace)
        if side == "both":
            mask = np.abs(zsc) > threshold
        elif side == "upper":
            mask = zsc > threshold
        elif side == "lower":
            mask = -zsc > threshold
        else:
            raise ValueError("Invalid value for parameter side")
    elif method == "medfilter":
        return ndimage.median_filter(trace, size=threshold)
    elif method == "peak_detect":
        # Use z-score to avoid mean variation
        zsc = stats.zscore(trace)
        sos = signal.butter(5, freq_cutoff, "hp", fs=fs, output="sos")
        zsc = signal.sosfilt(sos, zsc - np.mean(zsc))
        if side == "both":
            zsc = np.abs(zsc)
        elif side == "upper":
            zsc = zsc
        elif side == "lower":
            zsc = -zsc
        else:
            raise ValueError("Invalid value for parameter side")
        if expected_stim_width is None:
            peaks = np.argwhere(zsc > threshold).ravel()
        else:
            peaks, _ = signal.find_peaks(
                zsc, prominence=threshold, wlen=expected_stim_width + 2
            )
        _, _, lips, rips = signal.peak_widths(zsc, peaks, rel_height=0.5)
        if fixed_width_remove:
            # For the case that the crosstalks are almost as long as the period between true spikes
            lips = peaks - lpad
            rips = peaks + rpad
        else:
            lips = np.floor(lips).astype(int) - lpad
            rips = np.ceil(rips).astype(int) + rpad
        mask = np.zeros_like(trace, dtype=bool)
        for j in range(len(lips)):
            mask[lips[j] - 1 : rips[j] + 1] = True
    else:
        raise ValueError("Method %s not implemented" % method)

    if mode == "remove":
        crosstalk_removed = trace[~mask]
    elif mode == "interpolate":
        crosstalk_removed = interpolate_indices(trace, mask)
    else:
        raise ValueError("Invalid value for parameter mode")
    if plot:
        ts = np.arange(len(trace)) / fs
        _, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(ts, trace)
        ax1.plot(ts[mask], trace[mask], "rx")

    return crosstalk_removed, mask


def interpolate_indices(trace, mask):
    xs = np.arange(len(trace))[~mask]
    ys = trace[~mask]
    interp_f = interpolate.interp1d(xs, ys, kind="cubic", fill_value="extrapolate")
    missing_xs = np.argwhere(mask).ravel()
    missing_ys = interp_f(missing_xs)
    interpolated = np.copy(trace)
    interpolated[mask] = missing_ys
    return interpolated


def get_spike_traces(trace: npt.NDArray[np.floating], 
                     peak_indices: Collection, 
                     bounds: Tuple[int, int],
                     normalize_height: bool = True) -> npt.NDArray[np.floating]:
    """Generate spike-triggered traces of a defined length from a signal trace and known peak
    indices
    
    Args:
        trace: Signal trace
        peak_indices: Indices of peaks in trace
        bounds: Tuple of (before, after) number of samples to include in spike trace
        normalize_height: If True, normalize each spike trace to have a maximum of 1
    Returns:
        spike_traces: Array of spike-triggered traces
    """
    spike_traces = np.ones((len(peak_indices), bounds[0] + bounds[1])) * np.nan
    for pk_idx, pk in enumerate(peak_indices):
        before_pad_length = max(bounds[0] - pk, 0)
        after_pad_length = max(0, pk + bounds[1] - len(trace))
        spike_trace = trace[max(0, pk - bounds[0]) : min(len(trace), pk + bounds[1])]
        spike_trace = np.concatenate(
            [
                np.ones(before_pad_length) * np.nan,
                spike_trace - np.min(spike_trace),
                np.ones(after_pad_length) * np.nan,
            ]
        )
        spike_traces[pk_idx, :] = spike_trace
    if normalize_height:
        spike_traces /= np.nanmax(spike_traces, axis=1)[:, None]
    return spike_traces


def align_fixed_offset(traces, offsets):
    """Align a list of traces according to known relative offsets"""
    max_offset = np.max(offsets)
    aligned_traces = np.nan * np.ones((traces.shape[0], traces.shape[1] + max_offset))
    for i in range(traces.shape[0]):
        start_idx = max_offset - offsets[i]
        aligned_traces[i, start_idx : start_idx + traces.shape[1]] = traces[i, :]
    return aligned_traces


def get_sta(
    trace,
    peak_indices,
    bounds,
    f_s=1,
    normalize_height=True,
    return_std=False,
    use_median=False,
):
    """Generate spike-triggered average from given reference indices (peaks or stimuli)"""
    spike_traces = get_spike_traces(
        trace, peak_indices, bounds, normalize_height
    )

    if len(peak_indices) == 0:
        sta = np.nan * np.ones(before + after)
    else:
        if use_median:
            sta = np.nanmedian(spike_traces, axis=0)
        else:
            sta = np.nanmean(spike_traces, axis=0)
    if return_std:
        ststd = np.nanstd(spike_traces, axis=0)
        return sta, ststd
    return sta


def get_spike_traces_timed(
    trace, sample_times, peak_indices, before_t, after_t, normalize_height=True, f_s=1
):
    """Generate individual spike traces with given sample times to account for missing data"""
    before = int(before_t * f_s)
    after = int(after_t * f_s)
    spike_traces = np.ones((len(peak_indices), before + after)) * np.nan
    for pk_idx, pk in enumerate(peak_indices):
        before_idx = np.argwhere(sample_times >= sample_times[pk] - before_t).ravel()[0]
        after_idx = np.argwhere(sample_times >= sample_times[pk] + after_t).ravel()[0]

        if before_idx > pk - before:
            corrected_before = pk - before_idx - 1
        else:
            corrected_before = before
        if after_idx < pk + after:
            corrected_after = after_idx - pk
        else:
            corrected_after = after

        before_pad_length = max(corrected_before - pk, 0)
        after_pad_length = max(0, pk + corrected_after - len(trace))
        spike_trace = np.concatenate(
            [
                np.ones(before_pad_length) * np.nan,
                trace[max(0, pk - corrected) : min(len(trace), pk + corrected)],
                np.ones(after_pad_length) * np.nan,
            ]
        )
        if normalize_height:
            spike_trace /= np.nanmax(spike_trace)
        spike_traces[pk_idx, :] = spike_trace
    return spike_traces


def spike_match_to_kernel(
    kernel, trace, peak_indices, normalize_height=True, full_output=False
):
    """Compare consecutive spikes in a trace to a known kernel"""
    if normalize_height:
        kernel = (kernel - np.min(kernel)) / (np.max(kernel) - np.min(kernel))
        trace = (trace - np.min(trace)) / (np.max(trace) - np.min(trace))
    corr = signal.correlate(trace, kernel, mode="same")
    local_maxes = []
    for idx, pi in enumerate(peak_indices):
        if idx == len(peak_indices) - 1:
            next_peak = len(corr)
        else:
            next_peak = peak_indices[idx + 1]
        local_maxes.append(np.max(corr[pi:next_peak]))
    if full_output:
        return local_maxes, corr
    else:
        return local_maxes


def analyze_sta(
    trace,
    peak_indices, 
    bounds,
    f_s=1,
    normalize_height=True,
    fitting_function=first_trough_exp_fit,
):
    """Generate spike-triggered average from trace and indices of peaks, as well as associated statistics"""

    spike_traces = get_spike_traces(
        trace, peak_indices, bounds, normalize_height
    )

    if len(peak_indices) == 0:
        sta = np.nan * np.ones(bounds[0] + bounds[1])
        ststd = np.nan * np.ones(bounds[0] + bounds[1])
    else:
        sta = np.nanmean(spike_traces, axis=0)
        ststd = np.nanstd(spike_traces, axis=0)

    sta_stats = fitting_function(spike_traces, bounds[0], bounds[1], f_s=f_s)
    return sta, ststd, sta_stats


def get_peak_statistics(
    df,
    min_peaks=6
) -> pd.Series:
    """Generate statistics on a set of detected peaks
    
    Args:
        df: DataFrame containing peak information
        min_peaks: Minimum number of peaks required to calculate standard deviation
    Returns:
        pandas Series containing statistics on the peaks
    """
    if len(df.shape) == 1:
        mean_prom = np.nan
        std_prom = np.nan
        mean_width = np.nan
        pct95_dff = np.nan
        pct5_dff = np.nan
        max_dff = np.nan
        min_dff = np.nan
        mean_isi = np.nan
        n_peaks = np.nan
    else:
        mean_prom = np.mean(df["prominence"])
        std_prom = np.std(df["prominence"])
        mean_width = np.mean(df["fwhm"])
        try:
            pct95_dff = np.percentile(df["prominence"], 95)
            pct5_dff = np.percentile(df["prominence"], 5)
            max_dff = np.max(df["prominence"])
            min_dff = np.min(df["prominence"])
        except Exception:
            pct95_dff = np.nan
            pct5_dff = np.nan
            max_dff = np.nan
            min_dff = np.nan
        n_peaks = df.shape[0]
        mean_isi = np.mean(df["isi"])
        if n_peaks < min_peaks:
            std_isi = np.nan
            std_width = np.nan
        else:
            std_isi = np.std(df["isi"])
            std_width = np.std(df["fwhm"])

    peak_stats = pd.Series(
        {
            "mean_isi": mean_isi,
            "std_isi": std_isi,
            "mean_prom": mean_prom,
            "std_prom": std_prom,
            "mean_width": mean_width,
            "std_width": std_width,
            "n_peaks": n_peaks,
            "pct95_dff": pct95_dff,
            "pct5_dff": pct5_dff,
            "max_dff": max_dff,
            "min_dff": min_dff,
        }
    )

    return peak_stats


def plot_mean_frequency(spike_stats_by_roi, embryos=[]):
    """Plot mean spike frequency against developmental time"""
    if len(embryos) == 0:
        embryos = spike_stats_by_roi.index.unique()

    fig1, ax1 = plt.subplots(figsize=(10, 10))
    for e in embryos:
        e_data = spike_stats_by_roi.loc[e]
        ax1.plot(e_data["hpf"], e_data["mean_freq"], label="E%d" % e)
    ax1.set_xlabel("Developmental Time (hpf)")
    ax1.set_ylabel("Mean spike frequency (Hz)")

    return fig1, ax1


def plot_isi_cv(spike_stats_by_roi, embryos=[]):
    """Plot ISI coefficient of variation against developmental time"""
    if len(embryos) == 0:
        embryos = spike_stats_by_roi.index.unique()
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    for e in embryos:
        e_data = spike_stats_by_roi.loc[e]
        ax1.plot(e_data["hpf"], e_data["std_isi"] / e_data["mean_isi"], label="E%d")
    ax1.set_xlabel("Developmental Time (hpf)")
    ax1.set_ylabel(r"ISI $\sigma/\mu$")
    return fig1, ax1


def sort_spikes(
    all_spikes, spike_metadata, n_lilli=None, n_neighbors=150, leiden_res=0.05
):
    """Sort spikes by using wavelet transform and Leiden clustering"""
    c = pywt.wavedec(all_spikes, "haar", axis=1)
    c = np.concatenate(c, axis=1)
    coeff = c.reshape((c.shape[0], -1))
    if n_lilli is not None:
        ks = np.apply_along_axis(lilliefors, 0, coeff)[0]
        best_coefficients = np.argsort(ks)
        coeff = coeff[:, best_coefficients[-n_lilli:]]

    features = np.concatenate(
        [
            coeff,
            spike_metadata["amplitude"].to_numpy()[:, np.newaxis],
            spike_metadata["power"].to_numpy()[:, np.newaxis],
        ],
        axis=1,
    )

    ss = StandardScaler()
    norm_features = ss.fit_transform(features)
    adj_matrix = neighbors.kneighbors_graph(norm_features, n_neighbors)
    graph = igraph.Graph.Adjacency(
        (adj_matrix > 0).todense().tolist(), mode="undirected"
    )
    vc = graph.community_leiden(resolution_parameter=leiden_res)
    print(np.max(vc.membership))
    cluster_membership = np.array(vc.membership)
    new_metadata = spike_metadata.copy()
    new_metadata["cluster"] = cluster_membership

    return new_metadata, norm_features


class TimelapseArrayExperiment:
    """Loads and access trace data generated by firefly timelapses and passed through spikecounter pipelines"""

    def __init__(self, data_folder, start_hpf, f_s, block_metadata=None):
        self.data_folder = data_folder
        self.start_hpf = start_hpf
        self.f_s = f_s
        if block_metadata is None:
            self.block_metadata = self._load_block_metadata(
                self.data_folder, self.start_hpf
            )
        else:
            self.block_metadata = self._load_block_metadata(
                block_metadata, self.start_hpf
            )
        self.data_loaded = False
        self.peaks_found = False
        self.t = None
        self.raw = None
        self.dFF = None
        self.dFF_noise = None
        self.missing_data = None
        self.peaks_data = None

        self.hpf_tag = "Hours post fertilization"

    def filter_timepoints(self, timepoints):
        """Throw out timepoints with bad data"""
        self.block_metadata = self.block_metadata.loc[timepoints]

    def preview_trace(self, timepoint, roi):
        """Plot mean intensity of a particular trace"""
        exptname = self.block_metadata["file_name"].loc[timepoint]
        data = (
            pd.read_csv(os.path.join(self.data_folder, "%s_traces.csv" % exptname))
            .set_index("z")
            .loc[0]
            .set_index("region")
        )
        region_data = data.loc[roi]
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(region_data["t"] / self.f_s, region_data["mean_intensity"])
        ax1.set_title("Start Time %.2f" % self.block_metadata["hpf"].loc[timepoint])
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Mean intensity")
        return fig1, ax1

    def load_traces(
        self,
        filter_function=standard_lp_filter,
        timepoints=None,
        per_trace_start=0,
        background_subtract=False,
        scale_lowest_mean=False,
        end_index=0,
        custom_timepoints=[],
        custom_preprocessing_functions=[],
        pb_window=401,
        corr_photobleach=True,
    ):
        """Load and merge traces from individual data CSVs containing time blocks for arrays of embryos."""
        if timepoints is None:
            timepoints = np.arange(0, self.block_metadata.shape[0])

        t = []
        data_blocks = []
        dFF_blocks = []
        dFF_noise_blocks = []
        background_blocks = []

        for idx in timepoints:
            # Load csv file
            exptname = self.block_metadata.loc[idx]["file_name"]
            offset = self.block_metadata.loc[idx]["offset"]
            data = (
                pd.read_csv(os.path.join(self.data_folder, "%s_traces.csv" % exptname))
                .set_index("z")
                .loc[0]
                .set_index("region")
            )

            # Get number of timepoints
            data = data.reset_index().set_index("t")
            if end_index == 0:
                end_index = len(data.index.unique())
            all_time_indices = np.sort(data.index.unique())[:end_index]
            data = data.loc[all_time_indices]

            # Get background of traces (either based on ROI selection, or a level from frames with no excitation associated with each timepoint)
            if background_subtract == "roi":
                background = (
                    pd.read_csv(
                        os.path.join(
                            self.data_folder,
                            "background_traces/%s_traces.csv" % exptname,
                        )
                    )
                    .set_index("z")
                    .loc[0]
                    .set_index("region")
                )
            elif isinstance(background_subtract, int) and background_subtract > 0:
                background = data.loc[all_time_indices[-background_subtract:]]
                background = background.reset_index().set_index("region")

                data = data.loc[all_time_indices[:-background_subtract]]
                all_time_indices = all_time_indices[:-background_subtract]

            # Convert number of timepoints to actual time using offset in seconds and sampling frequency
            t_timepoint = all_time_indices[per_trace_start:] / self.f_s + offset

            # Move back to indexing by ROI
            data = data.reset_index().set_index("region")
            regions = list(data.index.unique("region"))
            #             print(regions)
            #             print(len(t_timepoint))

            # Initialize raw data arrays for the current timepoint
            raw_data_timepoint = np.zeros((len(regions), len(t_timepoint)))
            background_timepoint = np.zeros((len(regions), len(t_timepoint)))

            # Iterate through regions and load into array
            for i, roi in enumerate(regions):
                roi_trace = data.loc[roi]["mean_intensity"][per_trace_start:]
                if idx in custom_timepoints:
                    roi_trace = custom_preprocessing_functions[idx](roi_trace)
                raw_data_timepoint[i, :]
                if background_subtract == "roi":
                    background_timepoint[i, :] = background.loc[roi]["mean_intensity"][
                        per_trace_start:
                    ]
                elif isinstance(background_subtract, int) and background_subtract > 0:
                    roi_trace -= np.mean(background.loc[roi]["mean_intensity"])
                if corr_photobleach:
                    roi_trace, _ = correct_photobleach(
                        roi_trace, method="localmin", nsamps=pb_window
                    )
                raw_data_timepoint[i, :] = roi_trace

            # Load blocks
            t.extend(list(t_timepoint))
            data_blocks.append(raw_data_timepoint)
            background_blocks.append(background_timepoint)

        # Scale to lowest mean to make raw intensities display nicely since blue light intensity fluctuates (don't think this does anything for DF/F)
        if scale_lowest_mean:
            data_means = np.array(
                [
                    np.mean(raw_data_timepoint, axis=1)
                    for raw_data_timepoint in data_blocks
                ]
            )
            scalings = (data_means / np.min(data_means, axis=0)).T
            for idx in range(data_means.shape[0]):
                data_blocks[idx] = data_blocks[idx] / scalings[:, idx][:, np.newaxis]

        # Convert raw intensity to DF/F
        #         print(len(data_blocks))
        for idx in range(len(data_blocks)):
            if background_subtract == "roi":
                background = np.apply_along_axis(
                    filter_function, 1, background_blocks[idx]
                )
            else:
                background = np.zeros_like(data_blocks[idx])
            #             print(data_blocks[idx].shape)
            dFFs = np.apply_along_axis(
                intensity_to_dff, 1, data_blocks[idx] - background
            )
            dFFs_filtered = np.apply_along_axis(filter_function, 1, dFFs)
            dFFs_noise = dFFs - dFFs_filtered
            dFF_blocks.append(dFFs_filtered)
            dFF_noise_blocks.append(dFFs_noise)

        data_blocks = np.concatenate(data_blocks, axis=1)
        dFF_blocks = np.concatenate(dFF_blocks, axis=1)
        dFF_noise_blocks = np.concatenate(dFF_noise_blocks, axis=1)

        # Interpolate to merge all the individual blocks, and mark gaps in the data
        t = np.array(t)
        t_interp = np.arange(np.min(t), np.max(t), step=1 / self.f_s)
        missing_data = np.zeros_like(t_interp, dtype=int)
        for t_idx in range(len(missing_data)):
            nearest_dist = np.min(np.abs(t - t_interp[t_idx]))
            if nearest_dist > 1 / self.f_s:
                missing_data[t_idx] = 1

        data_interp = np.zeros((data_blocks.shape[0], len(t_interp)))
        dFF_interp = np.zeros((dFF_blocks.shape[0], len(t_interp)))
        dFF_noise_interp = np.zeros((dFF_blocks.shape[0], len(t_interp)))

        for roi in range(data_interp.shape[0]):
            f_trace = interpolate.interp1d(t, data_blocks[roi, :])
            data_interp[roi, :] = f_trace(t_interp)
            f_dff = interpolate.interp1d(t, dFF_blocks[roi, :])
            f_dff_noise = interpolate.interp1d(t, dFF_noise_blocks[roi, :])
            dFF_interp[roi, :] = f_dff(t_interp)
            dFF_noise_interp[roi, :] = f_dff_noise(t_interp)

        self.filter_timepoints(timepoints)
        self.t = t_interp
        self.raw = data_interp
        self.dFF = dFF_interp
        self.dFF_noise = dFF_noise_interp
        self.missing_data = missing_data
        self.n_rois = dFF_interp.shape[0]
        self.data_loaded = True

    def timepoint_to_filename(self, timepoint, time="hpf"):
        """Retrieve a raw datafile name based on timepoint"""
        if time == "hpf":
            time_array = self.block_metadata["hpf"].to_numpy()
        elif time == "s":
            time_array = self.block_metadata["offset"].to_numpy()
        else:
            raise Exception("hpf or s time required")
        idx = np.argwhere(timepoint < time_array).ravel()[0] - 1
        return self.block_metadata["file_name"].iloc[idx]

    def analyze_peaks(
        self,
        prominence="auto",
        prefilter=None,
        baseline_start=0,
        baseline_duration=3000,
        auto_prom_scale=0.3,
        **peak_detect_params
    ):
        """Apply scipy detect_peaks on all ROIs and"""
        dfs = []
        for roi in range(self.n_rois):
            try:
                if prefilter is None:
                    trace = self.dFF[roi, :]
                else:
                    trace = prefilter(self.dFF[roi, :])
                if prominence == "snr":
                    if (
                        np.sum(
                            self.dFF_noise[
                                roi, baseline_start : baseline_duration + baseline_start
                            ]
                        )
                        > 0
                    ):
                        noise_level = np.std(
                            self.dFF_noise[
                                roi, baseline_start : baseline_duration + baseline_start
                            ]
                        )
                    else:
                        noise_level = np.std(
                            self.dFF[
                                roi, baseline_start : baseline_duration + baseline_start
                            ]
                        )
                    df = analyze_peaks(
                        self.dFF[roi, :],
                        prominence=noise_level * auto_prom_scale,
                        f_s=self.f_s,
                        **peak_detect_params
                    )
                else:
                    df = analyze_peaks(
                        self.dFF[roi, :],
                        prominence=prominence,
                        f_s=self.f_s,
                        **peak_detect_params
                    )
            except Exception as e:
                print("ROI: %d" % roi)
                raise e
            if df is not None:
                df["t"] = self.t[df["peak_idx"]]
                df["roi"] = roi
                df["hpf"] = df["t"] / 3600 + self.start_hpf
                dfs.append(df)
            else:
                df = pd.DataFrame(
                    [(np.nan, np.nan, np.nan, np.nan, np.nan, roi, np.nan)],
                    columns=[
                        "peak_idx",
                        "prominence",
                        "fwhm",
                        "isi",
                        "t",
                        "roi",
                        "hpf",
                    ],
                )
                dfs.append(df)
        self.peaks_data = pd.concat(dfs, axis=0).set_index("roi")
        self.peaks_found = True

    def get_windowed_peak_stats(
        self,
        window_size: int,
        prominence: Union[float, None] = None,
        height: Union[float, None] = None,
        overlap: float = 0.5,
        isi_stat_min_peaks: int = 7,
        sta_bounds: Union[Tuple[int, int], None] = None
    ) -> Tuple[pd.DataFrame, Union[Dict, None], Union[Dict, None]]:
        """Get ISI statistics averaged over a moving window

        Args:
            window_size: Size of window in indices
            prominence: Minimum prominence of peaks to include. Defaults to None.
            height: Minimum height of peaks to include. Defaults to None.
            overlap: Overlap between windows. Defaults to 0.5.
            isi_stat_min_peaks: Minimum number of peaks to calculate ISI statistics. Defaults to 7.
            sta_bounds: Tuple of start and end times for STA calculation. Defaults to None.
        Returns:
            Tuple of:
                dataframe containing ISI statistics per time window
                dictionaries of average and standard deviation of spike-triggered dFF traces per
                    time window, keyed by embryo
        
        """
        if self.peaks_data is None:
            raise AttributeError("peaks_data not defined. Run analyze_peaks() first")
        if self.dFF is None or self.t is None or self.missing_data is None:
            raise AttributeError("Traces are not defined. Run load_traces() first")

        sta_embryos = {}
        ststd_embryos = {}
        spike_stats_by_roi = []
        segment_edges = self._find_segment_edges()

        for roi in self.peaks_data.index.unique():
            peak_data = self.peaks_data.loc[roi]
            # Apply height and prominence filters to peaks
            filter_mask = np.ones(peak_data.shape[0], dtype=bool)
            if prominence is not None:
                filter_mask = np.bitwise_and(
                    filter_mask,
                    peak_data["prominence"]
                    > np.percentile(peak_data["prominence"], prominence),
                )
            if height is not None:
                filter_mask = np.bitwise_and(
                    filter_mask,
                    self.dFF[roi, :][peak_data["peak_idx"]]
                    > height
                    * np.percentile(self.dFF[roi, :][peak_data["peak_idx"]], 95),
                )
            peak_data = peak_data[filter_mask]
            if len(peak_data.shape) == 1:
                continue

            peak_indices = np.array(peak_data["peak_idx"])
            window_indices = np.arange(
                0, self.dFF.shape[1] - window_size, step=int(window_size - overlap * window_size)
            )
            
            if sta_bounds:
                sta = np.zeros((len(window_indices), sta_bounds[0] + sta_bounds[1]))
                ststd = np.zeros((len(window_indices), sta_bounds[0] + sta_bounds[1]))
            else:
                sta = None
                ststd = None

            for wi_idx, wi in enumerate(window_indices):
                # Select peaks that are within a time window
                mask = (peak_indices >= wi) * (peak_indices < (wi + window_size))
                if ~isinstance(mask, np.ndarray):
                    mask = np.array([mask])
                mask = np.squeeze(mask)

                for edge_pair in segment_edges:
                    if edge_pair[1] >= wi and edge_pair[1] < wi + window_size:
                        left_of_segment_edge = np.argwhere(
                            peak_indices - edge_pair[1] < 0
                        ).ravel()
                        if len(left_of_segment_edge) > 0:
                            last_peak_in_segment = left_of_segment_edge[-1]
                            mask[last_peak_in_segment] = False
                            # return None
                masked_peak_df = peak_data.loc[mask.ravel()]

                # Get peak statistics
                try:
                    roi_spike_stats = get_peak_statistics(
                        masked_peak_df,
                        min_peaks=isi_stat_min_peaks,
                    )
                except Exception as e:
                    print(peak_data)
                    print(roi)
                    print(mask)
                    print(peak_indices)
                    print(wi)
                    raise e

                roi_spike_stats["offset"] = self.t[wi]
                roi_spike_stats["hpf"] = (
                    roi_spike_stats["offset"] / 3600 + self.start_hpf
                )
                roi_spike_stats["roi"] = roi
                roi_spike_stats["mean_freq"] = (
                    roi_spike_stats["n_peaks"]
                    / (window_size - np.sum(self.missing_data[wi : wi + window_size]))
                    * self.f_s
                )

                # Optionally collect statistics on spike-triggered average
                if sta is not None and ststd is not None:
                    locs = np.array(masked_peak_df["peak_idx"])
                    roi_sta, roi_ststd, roi_sta_stats = analyze_sta(self.dFF[roi], locs, sta_bounds)
                    sta[wi_idx, :] = roi_sta
                    ststd[wi_idx, :] = roi_ststd
                    roi_spike_stats = roi_spike_stats.append(roi_sta_stats)

                spike_stats_by_roi.append(roi_spike_stats)

            sta_embryos[roi] = sta
            ststd_embryos[roi] = ststd

        spike_stats_by_roi = pd.DataFrame(spike_stats_by_roi)
        spike_stats_by_roi = spike_stats_by_roi.reset_index(drop=True).set_index("roi")
        return spike_stats_by_roi, sta_embryos, ststd_embryos

    ### Plotting functions ###
    def plot_raw_and_dff(self, roi, figsize=(12, 6), time="s"):
        """Plot DF/F and raw traces over all blocks"""
        t, timeseries_start = self._get_time(time)

        fig1, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()
        ls = []
        ls.extend(ax1.plot(t, self.raw[roi, :], label="Counts"))
        ls.extend(ax2.plot(t, self.dFF[roi, :], color="C1", label=r"$\Delta F/F$"))
        labels = visualize.get_line_labels(ls)

        for offset in timeseries_start:
            ax1.axvline(offset, color="black")

        ax1.set_xlabel("Time (%s)" % time)
        ax1.set_ylabel("Raw")
        ax2.set_xlabel(r"$\Delta F/F$")
        ax2.legend(ls, labels)

        return fig1, [ax1, ax2]

    def plot_spikes(
        self,
        rois: Union[int, Iterable, None] = None,
        n_cols: int = 1,
        figsize: Tuple[int, int] = (12, 4),
        time: str = "s",
        x_lim: Union[Tuple[float, float], None] = None,
    ) -> Tuple[figure.Figure, Collection[axes.Axes]]:
        """Plot spikes found using find_peaks on DF/F

        Args:
            rois: ROI to plot. If None, plot all ROIs
            n_cols: Number of columns in the figure
            figsize: Size of the figure
            time: Time unit to use for the x-axis
            x_lim: Optional limits of the x-axis
        Returns:
            Figure and axes objects

        Raises:
            AttributeError: If no DF/F data has been loaded. Run load_traces() first.
            AttributeError: If no peaks have been found. Run analyze_peaks() first.

        """
        if self.dFF is None:
            raise AttributeError("No DF/F data. Run load_traces() first")
        if self.peaks_data is None:
            raise AttributeError("No peaks found. Run analyze_peaks() first")
        if rois is None:
            rois = np.arange(self.n_rois)
        else:
            rois = list(utils.make_iterable(rois))

        n_rows = int(np.ceil(len(rois) / n_cols))
        fig1, axs = plt.subplots(
            n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows), squeeze=False
        )
        axs = np.array(axs).ravel()
        t, _ = self._get_time(time)

        for idx, ax in enumerate(axs):
            roi = rois[idx]
            roi_peaks = self.peaks_data.loc[roi]
            peak_indices = roi_peaks["peak_idx"]

            ax.plot(t, self.dFF[roi, :])
            ax.plot(t[peak_indices], self.dFF[roi, peak_indices], "rx")
            ax.set_ylabel(r"$\Delta F/F$")
            ax.set_title(f"ROI {roi}")
            if x_lim:
                ax.set_xlim(x_lim)
        
        axs[-1].set_xlabel(f"Time ({time})")
        plt.tight_layout()
        return fig1, axs

    def plot_spikes_with_mask(
        self, mask, rois=None, figsize=(16, 4), img_size=4, time="s"
    ):
        if rois is None:
            rois = np.arange(self.n_rois)
        elif isinstance(rois, int):
            rois = [rois]

        n_rows = len(rois)
        fig1, axes = plt.subplots(
            n_rows,
            2,
            figsize=(figsize[0], figsize[1] * n_rows),
            gridspec_kw={"width_ratios": [figsize[0] - img_size, img_size]},
        )
        t, _ = self._get_time(time)

        for idx in range(len(rois)):
            roi = rois[idx]
            ax = axes[idx, 0]
            if roi in self.peaks_data.index.unique():
                roi_peaks = self.peaks_data.loc[roi]
                try:
                    peak_indices = roi_peaks[["peak_idx"]].values.astype(int)
                except Exception as e:
                    print(roi_peaks[["peak_idx"]])
                    raise e
                ax.plot(t, self.dFF[roi, :])
                if len(peak_indices) > 0 and np.all(peak_indices >= 0):
                    ax.plot(t[peak_indices], self.dFF[roi, peak_indices], "rx")
            ax.set_xlabel("Time (%s)" % time)
            ax.set_ylabel(r"$\Delta F/F$")
            ax.set_title("ROI %d" % roi)

            axes[idx, 1].imshow(mask == (roi + 1))
        plt.tight_layout()
        return fig1, axes

    def plot_peak_quality_metrics(
        self, rois, figsize=(10, 10), ma_window=400, time="hpf"
    ):
        """Plot peak quality metrics:

        Distribution of DF/F of segmented peaks (CDF and PDF), to ensure that cutoffs are appropriate
        DF/F compared to F, to show that peak intensity is independent of raw fluorescence
        Fluorescence of spike and average over time - does ratio of spike to baseline change over time

        """
        t, _ = self._get_time(time)
        fig1, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()

        axes[0].set_xlabel(r"$\Delta F/F$")
        axes[0].set_ylabel("CDF")

        axes[1].set_xlabel(r"$\Delta F/F$")
        axes[1].set_ylabel("PDF")

        axes[2].set_xlabel(r"$\Delta F/F$")
        axes[2].set_ylabel("F")

        axes[3].set_xlabel("Time (%s)" % time)
        axes[3].set_ylabel("F")

        roi_it = utils.convert_to_iterable(rois)
        for idx, roi in enumerate(roi_it):
            roi_peaks = self.peaks_data.loc[roi]
            peak_indices = roi_peaks["peak_idx"]
            # Should I be taking the local percentile here?
            ma = signal.convolve(
                self.raw[roi, :], np.ones(ma_window) / ma_window, "same"
            )

            axes[0].hist(
                self.dFF[roi, peak_indices],
                bins=50,
                cumulative=True,
                density=True,
                alpha=0.5,
            )

            axes[1].hist(self.dFF[roi, peak_indices], bins=50, density=True, alpha=0.5)

            axes[2].scatter(self.dFF[roi, peak_indices], self.raw[roi, peak_indices])

            axes[3].scatter(
                t[peak_indices],
                ma[peak_indices],
                color="C%d" % idx,
                label="E%d" % (roi + 1),
                s=5,
                linewidth=1,
            )
            axes[3].scatter(
                t[peak_indices],
                self.raw[roi, peak_indices],
                color="C%d" % idx,
                marker="+",
                s=5,
                linewidth=1,
            )
        return fig1, axes

    def plot_power_spectra(self, rois, tmin, tmax, figsize=(12, 12)):
        """Plot power spectra using welch and periodogram methods for a given window

        For evaluating noise
        """
        tidx_min = np.argwhere(self.t > tmin)[0][0]
        tidx_max = np.argwhere(self.t > tmax)[0][0]

        fig1, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()

        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("F")

        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel(r"$S_{xx}(f)$")
        axes[1].set_title("Welch")

        axes[2].set_xlabel("Frequency (Hz)")
        axes[2].set_ylabel(r"$S_{xx}(f)$")
        axes[2].set_title("Periodogram")
        roi_it = utils.convert_to_iterable(rois)
        for _, roi in enumerate(roi_it):
            raw = self.raw[roi, tidx_min:tidx_max]
            axes[0].plot(self.t[tidx_min:tidx_max], raw)
            f, pxx = signal.welch(raw - np.mean(raw), fs=self.f_s)
            axes[1].plot(f, pxx)
            f, pxx = signal.periodogram(raw - np.mean(raw), fs=self.f_s)
            axes[2].plot(f, pxx)

        return fig1, axes

    def plot_spike_width_stats(self, rois, figsize=(12, 6), time="hpf"):
        """Plot spike width against developmental time and interspike interval"""
        roi_it = utils.convert_to_iterable(rois)

        fig1, axes = plt.subplots(1, 2, figsize=figsize)
        axes = axes.ravel()

        for _, roi in enumerate(roi_it):
            if time == "hpf":
                t = self.peaks_data.loc[roi]["t"] / 3600 + self.start_hpf
                axes[0].set_xlabel("Developmental Time (hpf)")
            elif time == "s":
                t = self.peaks_data.loc[roi]["t"]
                axes[0].set_xlabel("Time (s)")
            else:
                raise ValueError("Time should be hpf or seconds")

            axes[0].scatter(t, self.peaks_data.loc[roi]["fwhm"])
            axes[1].scatter(
                self.peaks_data.loc[roi]["fwhm"], self.peaks_data.loc[roi]["isi"]
            )

        axes[0].set_ylabel("FWHM (s)")
        axes[1].set_ylabel("ISI (s)")
        axes[1].set_xlabel("FWHM (s)")
        return fig1, axes

    def plot_isi_ks(self, rois, cutoff_times, figsize=(12, 12), time="hpf"):
        """Plot interspike interval Kolmogorov-Smirnov test"""
        roi_it = utils.convert_to_iterable(rois)
        fig1, ax1 = plt.subplots(figsize=figsize)

        for _, roi in enumerate(roi_it):
            if time == "hpf":
                t = self.peaks_data.loc[roi]["t"] / 3600 + self.start_hpf
                ax1.set_xlabel("Developmental Time (hpf)")
            elif time == "s":
                t = self.peaks_data.loc[roi]["t"]
                ax1.set_xlabel("Time (s)")
            t_idx, ks_p = sstats.ks_window(
                self.peaks_data.loc[roi]["isi"], window=50, overlap=0.9
            )
            ax1.plot(t[t_idx], ks_p)

        ax1.axhline(0.05, linestyle="--", color="black", label=r"$p<0.05$")
        ax1.legend()
        ax1.set_xlabel("HPF")
        ax1.set_ylabel(r"Kolmogorov-Smirnov $p$")
        ax1.set_yscale("log")

        return fig1, ax1

    def plot_isi_histograms(self, rois, cutoff_times, figsize=(6, 6), time="hpf"):
        """Plot interspike interval histograms"""
        roi_it = utils.convert_to_iterable(rois)
        roi_it = list(roi_it)
        if len(cutoff_times) != len(roi_it):
            raise ValueError(
                "Length of cutoff times array should be same as length of rois"
            )

        fig1, axes = visualize.tile_plots_conditions(cutoff_times, figsize)
        for idx, roi in enumerate(roi_it):
            if time == "hpf":
                t = self.peaks_data.loc[roi]["t"] / 3600 + self.start_hpf
            elif time == "s":
                t = self.peaks_data.loc[roi]["t"]
            hist_isi = self.peaks_data.loc[roi]["isi"]
            hist_isi = hist_isi[t < cutoff_times[idx]]
            (
                _,
                bins,
                _,
            ) = axes[
                idx
            ].hist(hist_isi, bins=np.linspace(0, 40, 20), density=True)
            axes[idx].set_xlabel("ISI (s)")
            axes[idx].set_ylabel("PDF")

            P = stats.expon.fit(hist_isi, loc=0)
            x = np.linspace(0, np.max(bins), 100)
            y = stats.expon.pdf(x, *P)
            axes[idx].plot(x, y)
            axes[idx].set_title("E%d" % roi)

        return fig1, axes

    def plot_dFF_spectrograms(
        self,
        rois,
        figsize=(8, 6),
        nperseg=1200,
        noverlap=600,
        max_plot_freq=1,
        time="hpf",
    ):
        """Plot spectrograms for each embryo"""
        roi_it = utils.convert_to_iterable(rois)
        roi_it = list(roi_it)
        spectrograms = []
        fig1, axes = visualize.tile_plots_conditions(roi_it, figsize)
        for idx, roi in enumerate(roi_it):
            f, t_s, Sxx = signal.spectrogram(
                self.dFF[roi, :] - np.mean(self.dFF[roi, :]),
                fs=self.f_s,
                nperseg=nperseg,
                noverlap=noverlap,
            )
            if time == "hpf":
                t = t_s / 3600 + self.start_hpf
                axes[idx].set_xlabel("Developmental Time (hpf)")
            elif time == "s":
                t = t_s
                axes[idx].set_xlabel("Time (s)")
            else:
                raise ValueError("Time should be in hpf or seconds")

            img = axes[idx].pcolormesh(
                t,
                f[f < max_plot_freq],
                Sxx[f < max_plot_freq, :],
                cmap="magma",
                norm=colors.LogNorm(vmin=1e-5),
            )
            cbar = fig1.colorbar(
                img, aspect=20, shrink=0.7, label=r"$S_{xx}(f)$", ax=axes[idx]
            )

            axes[idx].set_title("E%d" % roi)
            spectrograms.append(Sxx)

        return fig1, axes, spectrograms

    def _get_time(self, time):
        """Returns time in hpf or seconds as well as start of each timeblock

        Args:
            time: "hpf" or "s"
        Returns:
            t: time in hpf or seconds
            timeseries_start (float): start of each timeblock
        Raises:
            ValueError: if time is not set
            ValueError: if time is not "hpf" or "s"

        """
        if self.t is None:
            raise ValueError("Time not set")
        if time == "s":
            t = self.t
            timeseries_start = self.block_metadata["offset"]
        elif time == "hpf":
            t = self.t / 3600 + self.start_hpf
            timeseries_start = self.block_metadata["hpf"]
        else:
            raise ValueError("Invalid time unit")
        return t, timeseries_start

    def _find_segment_edges(self):
        """ Find the edges of each segment of missing data in the time series
        """
        if self.missing_data is None:
            raise AttributeError("Run load_traces() first")
        missing_edges = self.missing_data[1:] - self.missing_data[:-1]
        rising_edge = np.argwhere(missing_edges == 1).ravel()
        falling_edge = np.argwhere(missing_edges == -1).ravel()
        segment_edges = list(
            zip([0] + list(falling_edge[: len(rising_edge) - 1]), list(rising_edge))
        )
        if len(rising_edge) == len(falling_edge):
            segment_edges.append((falling_edge[-1], len(self.missing_data)))
        return segment_edges

    def _load_block_metadata(self, data_folder, start_hpf):
        """Load metadata from CSV that satisfies a pandas dataframe with the following columns:
        start_time, file_name, condition
        """
        if isinstance(data_folder, str):
            block_metadata = (
                pd.read_csv(os.path.join(data_folder, "experiment_data.csv"))
                .sort_values("start_time")
                .reset_index()
            )
        elif isinstance(data_folder, pd.DataFrame):
            block_metadata = data_folder.sort_values("start_time").reset_index()
        else:
            raise Exception("Invalid type for parameter data_folder")

        if not set(["start_time", "file_name"]).issubset(block_metadata.columns):
            raise Exception("Incorrect file format")
        if "condition" not in block_metadata.columns:
            block_metadata["condition"] = ""

        start_times = [
            datetime.strptime(t, "%H:%M:%S") for t in list(block_metadata["start_time"])
        ]
        offsets = [s - start_times[0] for s in start_times]
        offsets = [o.seconds for o in offsets]
        block_metadata["offset"] = offsets
        block_metadata["hpf"] = start_hpf + block_metadata["offset"] / 3600
        if "index" in block_metadata.columns:
            del block_metadata["index"]
        return block_metadata
