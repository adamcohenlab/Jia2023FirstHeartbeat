import pandas as pd
import numpy as np
import scipy.signal as signal
import scipy.interpolate as interpolate
from scipy import optimize
import matplotlib.pyplot as plt
from datetime import datetime
import os
plt.style.use(os.path.join(os.path.dirname(__file__), "../report.mplstyle"))

def intensity_to_dff(intensity, percentile_threshold=10, moving_average=False, window=None):
    """Calculate DF/F from intensity counts

    """
    if moving_average:
        if window is not None:
            raise Exception("Window length required")
        kernel = np.ones(window)/window
        ma = signal.fftconvolve(intensity, kernel, mode='same')
        dFF = intensity/ma
    else:
        if window is not None:
            raise Exception("Windowed percentile not yet implemented")
    
        baseline = np.mean(intensity[intensity<np.percentile(intensity, percentile_threshold)])
        dFF = (intensity - baseline)/baseline
    return dFF

def standard_lp_filter(raw, norm_thresh=0.5):
    b, a = signal.butter(5, norm_thresh)
    intensity = signal.filtfilt(b, a, raw)
    mean_freq = 2.0
    # b, a = signal.butter(5, [mean_freq-0.2, mean_freq+0.2], btype="bandstop", fs=10.2)
    # intensity = signal.filtfilt(b, a, intensity)
    return intensity

def analyze_peaks(trace, prominence="auto", wlen=400, threshold=0, f_s=1):
    """ Analyze peaks within a given trace and return the following statistics, organized in a pandas DataFrame:

    peak_idx: index where each peak is
    prominence: height above baseline
    fwhm: full-width half-maximum

    """
    if prominence == "auto":
        p = np.percentile(trace, 95)/2
    else:
        p = prominence

    peaks, properties = signal.find_peaks(trace, prominence=p, height=0, wlen=wlen)
    peaks = peaks[trace[peaks]>threshold]
    fwhm = signal.peak_widths(trace, peaks, rel_height=0.5)[0]/f_s
    isi = (peaks[1:] - peaks[:-1])/f_s
    isi = np.append(isi, np.nan)
    prominences = np.array(properties["prominences"])
    prominences = prominences[trace[peaks]>threshold]
    res = pd.DataFrame({"peak_idx": peaks, "prominence": prominences, "fwhm": fwhm, "isi": isi})
    return res

def first_trough_exp_fit(st_traces, before, after, f_s=1):
    """ Fit an exponential to a detected peak up to the first trough

    """
    if st_traces.shape[0] == 0:
        return pd.Series({"alpha":np.nan, "c":np.nan, "alpha_err":np.nan, "c_err":np.nan})
    
    sta = np.nanmean(st_traces, axis=0)
    relmin, _ = signal.find_peaks(1-sta, height=0.5)
    
    if len(relmin) > 0:
        relmin = relmin[0]
    else:
        relmin = len(sta)-before
    ts = np.tile(np.arange(relmin)/f_s,st_traces.shape[0])
    ys = st_traces[:,before:relmin+before].ravel()

    popt, pcov = optimize.curve_fit(lambda x, alpha, c: np.exp(-alpha*x) + c, ts, ys, p0=[0.6, 0.2])

    alpha = popt[0]
    c = popt[1]
    alpha_err = np.sqrt(pcov[0,0])
    c_err = np.sqrt(pcov[1,1])

    return pd.Series({"alpha":alpha, "c":c, "alpha_err":alpha_err, "c_err":c_err})

def analyze_sta(trace, peak_indices, before, after, f_s=1, normalize_height=True, fitting_function=first_trough_exp_fit):
    """ Generate spike-triggered average from trace and indices of peaks, as well as associated statistics

    """
    spike_traces = np.ones((len(peak_indices), before+after))*np.nan
    for pk_idx, pk in enumerate(peak_indices):
        spike_trace = np.concatenate([np.ones(max(before - pk, 0))*np.nan,
                                                trace[max(0, pk-before):min(len(trace),pk+after)],
                                        np.ones(max(0, pk+after - len(trace)))*np.nan])
        if normalize_height:
            spike_trace /= np.nanmax(spike_trace)
        spike_traces[pk_idx,:] = spike_trace
    if len(peak_indices) == 0:
        sta = np.nan*np.ones(before+after)
        ststd = np.nan*np.ones(before+after)
    else:
        sta = np.nanmean(spike_traces, axis=0)
        ststd = np.nanstd(spike_traces, axis=0)

    sta_stats = fitting_function(spike_traces, before, after, f_s=f_s)
    return sta, ststd, sta_stats

def masked_peak_statistics(df, mask, f_s=1, min_peaks=6, sta_stats=False, trace=None, sta_before=None, sta_after=None):
    """ Generate statistics on a set of detected peaks after masking (e.g. subsetting or throwing out bad data)

    """
    if len(mask) != df.shape[0]:
        raise Exception("Mask must be same length as number of peaks")
    
    masked_df = df.loc[mask]

    mean_prom = np.mean(masked_df["prominence"])
    std_prom = np.std(masked_df["prominence"])
    mean_width = np.mean(masked_df["fwhm"])
    
    locs = np.array(masked_df["peak_idx"])
    n_peaks = len(locs)
    mean_isi = np.mean(masked_df["isi"])
    
    if np.sum(mask) < min_peaks:
        std_isi = np.nan
        std_width = np.nan
    else:
        std_isi = np.std(masked_df["isi"])
        std_width = np.std(masked_df["fwhm"])

    peak_stats = pd.Series({"mean_isi": mean_isi, "std_isi": std_isi, "mean_prom": mean_prom, \
        "std_prom": std_prom, "mean_width": mean_width, "std_width": std_width, "n_peaks": n_peaks})
    
    if sta_stats:
        if trace is None or sta_before is None or sta_after is None:
            raise Exception("Trace and duration must be provided for spike triggered average")
        sta, ststd, sta_stats = analyze_sta(trace, locs, sta_before, sta_after)
    else:
        sta = None
        ststd = None
        sta_stats = None

    if sta_stats is not None:
        peak_stats = peak_stats.append(sta_stats)
        
    return peak_stats, sta, ststd

class TimelapseArrayExperiment():
    """ Loads and access trace data generated by firefly timelapses and passed through spikecounter pipelines

    """
    def __init__(self, data_folder, start_hpf, f_s):
        self.data_folder = data_folder
        self.start_hpf = start_hpf
        self.f_s = f_s
        self.block_metadata = self._load_block_metadata(self.data_folder, self.start_hpf)
        self.data_loaded = False
        self.peaks_found = False
        self.t = None
        self.raw = None
        self.dFF = None
        self.missing_data = None
        self.peaks_data = None
    
    def filter_timepoints(self, timepoints):
        """ Throw out timepoints with bad data
        """
        self.block_metadata = self.block_metadata.loc[timepoints]
    
    def preview_trace(self, timepoint, roi):
        """ Plot mean intensity of a particular trace 
        """
        exptname = self.block_metadata["file_name"].loc[timepoint]
        data = pd.read_csv(os.path.join(self.data_folder, "%s_traces.csv" % exptname)).set_index("z").loc[0].set_index("region")
        region_data = data.loc[roi]
        fig1, ax1 = plt.subplots(figsize=(12,6))
        ax1.plot(region_data["t"]/self.f_s, region_data["mean_intensity"])
        ax1.set_title("Start Time %.2f" % self.block_metadata["hpf"].loc[timepoint])
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Mean intensity")
        return fig1, ax1

    def load_traces(self, filter_function=standard_lp_filter, timepoints=None, per_trace_start=0, background_subtract=False, \
        scale_lowest_mean=False, end_index=0, custom_timepoints=[], custom_preprocessing_functions = []):
        """ Load and merge traces from individual data CSVs containing time blocks for arrays of embryos.
        """
        if timepoints is None:
            timepoints = np.arange(0, self.block_metadata.shape[0])

        t = []
        data_blocks = []
        dFF_blocks = []
        background_blocks = []

        for idx in timepoints:
            # Load csv file
            exptname = self.block_metadata["file_name"].loc[idx]
            offset = self.block_metadata["offset"].loc[idx]
            data = pd.read_csv(os.path.join(self.data_folder, "%s_traces.csv" % exptname)).set_index("z").loc[0].set_index("region")
            
            # Get number of timepoints
            data = data.reset_index().set_index("t")
            if end_index == 0:
                end_index = len(data.index.unique())
            all_time_indices = np.sort(data.index.unique())[:end_index]
            data = data.loc[all_time_indices]
            
            # Get background of traces (either based on ROI selection, or a level from frames with no excitation associated with each timepoint)
            if background_subtract=="roi":
                background = pd.read_csv(os.path.join(self.data_folder, "background_traces/%s_traces.csv" % exptname)).set_index("z").loc[0].set_index("region")
            elif isinstance(background_subtract, int):            
                background = data.loc[all_time_indices[-background_subtract:]]
                background = background.reset_index().set_index("region")

                data = data.loc[all_time_indices[:-background_subtract]]
                all_time_indices = all_time_indices[:-background_subtract]

            # Convert number of timepoints to actual time using offset in seconds and sampling frequency
            t_timepoint = all_time_indices[per_trace_start:]/self.f_s + offset
            
            # Move back to indexing by ROI
            data = data.reset_index().set_index("region")
            regions = list(data.index.unique("region"))

            # Initialize raw data arrays for the current timepoint
            raw_data_timepoint = np.zeros((len(regions), len(t_timepoint)))
            background_timepoint = np.zeros((len(regions), len(t_timepoint)))
            
            # Iterate through regions and load into array
            for i, roi in enumerate(regions):
                roi_trace = data.loc[roi]["mean_intensity"][per_trace_start:]
                if idx in custom_timepoints:
                    roi_trace = custom_preprocessing_functions[idx](roi_trace)
                raw_data_timepoint[i,:]
                if background_subtract=="roi":
                    background_timepoint[i,:] = background.loc[roi]["mean_intensity"][per_trace_start:]
                elif isinstance(background_subtract, int):
                    roi_trace -= np.mean(background.loc[roi]["mean_intensity"])
                
                raw_data_timepoint[i,:] = roi_trace
            
            # Load blocks
            t.extend(list(t_timepoint))
            data_blocks.append(raw_data_timepoint)
            background_blocks.append(background_timepoint)
        
        # Scale to lowest mean to make raw intensities display nicely since blue light intensity fluctuates (don't think this does anything for DF/F)
        if scale_lowest_mean:
            data_means = np.array([np.mean(raw_data_timepoint, axis=1) for raw_data_timepoint in data_blocks])
            scalings = (data_means/np.min(data_means, axis=0)).T
            for idx in range(data_means.shape[0]):
                data_blocks[idx] = data_blocks[idx]/scalings[:,idx][:,np.newaxis]
        
        # Convert raw intensity to DF/F
        for idx in range(len(data_blocks)):
            if background_subtract=="roi":
                background=np.apply_along_axis(filter_function, 1, background_blocks[idx])
            else:
                background = np.zeros_like(data_blocks[idx])
            filtered_intensity = np.apply_along_axis(filter_function, 1, data_blocks[idx])
            dFFs = np.apply_along_axis(intensity_to_dff, 1, filtered_intensity-background)
            dFF_blocks.append(dFFs)

        data_blocks = np.concatenate(data_blocks, axis=1)
        dFF_blocks = np.concatenate(dFF_blocks, axis=1)

        # Interpolate to merge all the individual blocks, and mark gaps in the data
        t = np.array(t)
        t_interp = np.arange(np.min(t), np.max(t), step=1/self.f_s)
        missing_data = np.zeros_like(t_interp, dtype=int)
        for t_idx in range(len(missing_data)):
            nearest_dist = np.min(np.abs(t-t_interp[t_idx]))
            if nearest_dist > 1/self.f_s:
                missing_data[t_idx] = 1
        
        data_interp = np.zeros((data_blocks.shape[0], len(t_interp)))
        dFF_interp = np.zeros((dFF_blocks.shape[0], len(t_interp)))

        for roi in range(data_interp.shape[0]):
            f_trace = interpolate.interp1d(t, data_blocks[roi,:])
            data_interp[roi,:] = f_trace(t_interp)
            f_dff = interpolate.interp1d(t, dFF_blocks[roi,:])
            dFF_interp[roi,:] = f_dff(t_interp)
        
        self.filter_timepoints(timepoints)
        self.t = t_interp
        self.raw = data_interp
        self.dFF = dFF_interp
        self.missing_data = missing_data
        self.n_rois = dFF_interp.shape[0]
        self.data_loaded = True
    
    def analyze_peaks(self, prominence="auto", wlen=400, threshold=0):
        dfs = []
        for roi in range(self.n_rois):
            df = analyze_peaks(self.dFF[roi,:], prominence=prominence, wlen=wlen, threshold=threshold, f_s=self.f_s)
            df["t"] = self.t[df["peak_idx"]]
            df["roi"] = roi
            dfs.append(df)
        self.peaks_data = pd.concat(dfs, axis=0).set_index("roi")
        self.peaks_found = True

    def get_windowed_peak_stats(self, window, prominence="auto", overlap=0.5, isi_stat_min_peaks=7, sta_before=0, sta_after=0, wlen=400):
        if not self.peaks_found:
            raise Exception("Peaks not found")

        sta_embryos = {}
        ststd_embryos = {}
        spike_stats_by_roi = []
        segment_edges = self._find_segment_edges()

        for roi in range(self.n_rois):
            peak_data = self.peaks_data.loc[roi]
            peak_indices = np.array(peak_data["peak_idx"])
            window_indices = np.arange(0, self.dFF.shape[1]-window, step=int(window - overlap*window))
            
            sta = np.zeros((len(window_indices), sta_before+sta_after))
            ststd = np.zeros((len(window_indices), sta_before+sta_after))

            for wi_idx, wi in enumerate(window_indices):
                mask = (peak_indices >= wi)*(peak_indices < (wi+window))
                for edge_pair in segment_edges:
                    if edge_pair[1] >= wi and edge_pair[1] < wi+window:
                        nearest_lower_peak = np.argwhere(peak_indices-edge_pair[1] > 0)[0] - 1
                        mask[nearest_lower_peak] = False
                            # return None
                if sta_after+sta_before > 0:
                    sta_stats = True
                else:
                    sta_stats = False
                
                
                # print(wi_idx, np.sum(mask))

                roi_spike_stats, roi_sta, roi_ststd = masked_peak_statistics(peak_data, mask, f_s=self.f_s, \
                    sta_stats=sta_stats, trace=self.dFF[roi,:], sta_before=sta_before, sta_after=sta_after, min_peaks=isi_stat_min_peaks)
                

                roi_spike_stats["offset"] = self.t[wi]
                roi_spike_stats["roi"] = roi
                roi_spike_stats["mean_freq"] = roi_spike_stats["n_peaks"]/(window - np.sum(self.missing_data[wi:wi+window]))*self.f_s

                sta[wi_idx,:] = roi_sta
                ststd[wi_idx,:] = roi_ststd
                spike_stats_by_roi.append(roi_spike_stats)

            sta_embryos[roi] = sta
            ststd_embryos[roi] = ststd
        
        spike_stats_by_roi = pd.DataFrame(spike_stats_by_roi)
        return spike_stats_by_roi, sta_embryos, ststd_embryos

    def plot_raw_and_dff(self, roi):
        return 0
    
    def _find_segment_edges(self):
        missing_edges = self.missing_data[1:] - self.missing_data[:-1]
        rising_edge = np.argwhere(missing_edges==1).ravel()
        falling_edge = np.argwhere(missing_edges==-1).ravel()
        segment_edges = list(zip([0] + list(falling_edge[:len(rising_edge)-1]), list(rising_edge)))
        return segment_edges

    def _load_block_metadata(self, data_folder, start_hpf):
        """ Load metadata from CSV that satisfies a pandas dataframe with the following columns:
        start_time, file_name, condition
        """
        
        block_metadata = pd.read_csv(os.path.join(data_folder, "experiment_data.csv")).sort_values("start_time").reset_index()
        
        if not set(['start_time', 'file_name']).issubset(block_metadata.columns):
            raise Exception("Incorrect file format")
        if 'condition' not in block_metadata.columns:
            block_metadata['condition'] = ""

        start_times = [datetime.strptime(t,"%H:%M:%S") for t in list(block_metadata["start_time"])]
        offsets = [s - start_times[0] for s in start_times]
        offsets = [o.seconds for o in offsets]
        block_metadata["offset"] = offsets
        block_metadata["hpf"] = start_hpf + block_metadata["offset"]/3600
        del block_metadata["index"]
        return block_metadata