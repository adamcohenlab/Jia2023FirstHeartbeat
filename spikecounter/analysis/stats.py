from scipy import stats, interpolate
from sklearn.utils.extmath import randomized_svd
import numpy as np

def ks_window(data, window, overlap=0.5):
    """ Perform Kolmogorov-Smirnov test over windowed portions of some data
    """
    ts = []
    ks_p = []
    for i in np.arange(0, len(data)-window, step=int(window - overlap*window)):
        _, p = stats.kstest(data[i:i+window], 'expon', args=(0, np.mean(data[i:i+window])))
        ts.append(i)
        ks_p.append(p)
    ts = np.array(ts)
    return ts, np.array(ks_p)


### Fitting routines for frequency/CV data to simulated moments
def correct_cov(cov, rs, ksize=5):
    """Correct for low sample number in coefficient of variation calculation from simulations at parameters far from bifurcation. We
    assume that because the spike arrivals are sufficiently rare the CV = 1.
    
    """
    reverse_cov = np.copy(np.flip(cov))
    fliprs = np.flip(rs)[:-1]
    diff = np.diff(reverse_cov)
    try:
        # Starting from high I, check when the CV starts to decrease again (it shouldn't decrease when I is far from bifurcation)
        try:
            idx = np.argwhere((diff<0)*(fliprs<1)).ravel()[0]
        except Exception as e:
            print(fliprs)
            print(diff)
            raise e
    except IndexError:
        # If we can't find an increase find the first time that it's NaN (no peakss detected)
        idx = np.argwhere(np.isnan(reverse_cov)).ravel()[0]-1
    # set CV to 1
    reverse_cov[idx+1:] = 1
    # Moving average smoothing
    reverse_cov[idx-ksize//2:idx+ksize//2] = np.convolve(reverse_cov, np.ones(ksize)/ksize, mode="same")[idx-ksize//2:idx+ksize//2]
    return np.flip(reverse_cov)

def remove_small_windows(cov, min_window_length):
    """ Sometimes there are portions of the data where for various reasons (crap coming into FOV, low SNR), some number of
    peaks are spuriously called before the beat has actually started. This function throws those CV measurements out.
    
    """
    processed = np.copy(cov)
    notnan = (~np.isnan(cov)).astype(int)
    
    # Find runs of datapoints with valid CV
    rising_edge = np.argwhere(np.diff(notnan) == 1).ravel()
    falling_edge = np.argwhere(np.diff(notnan) == -1).ravel()
    if notnan[0] == 1:
        # If the first index has valid CV, we make a fake window edge at -1
        rising_edge = np.concatenate(([-1], rising_edge))
    rising_edge = rising_edge[:len(falling_edge)]
    window_lengths = falling_edge - rising_edge
    for idx, wl in enumerate(window_lengths):
        if wl < min_window_length:
            # Replace short runs with nan
            processed[rising_edge[idx]+1:falling_edge[idx]+1] = np.nan
    return processed

def gen_exptdata(freq_trace, cov_trace, plot=False, n_points_offset=0, min_window_length=10):
    """ Perform preprocessing by removing spuriously detected peaks and downsampling frequency traces to the
    same number of indices as the CV traces.
    
    """
    cov_length = len(cov_trace)
    freq_length = len(freq_trace)
    
    # Remove spuriously detected peaks from experimental CV measurements
    cov_removed_windows = remove_small_windows(cov_trace, min_window_length)
    if plot:
        plt.plot(cov_removed_windows)
    # Find the start and end of the real data (aligned by initiation time)
    last_nan_before_cov = np.argwhere(~np.isnan(cov_removed_windows))[0][0]
    first_nan_after_cov = np.argwhere(np.isnan(cov_removed_windows)).ravel()
    
    try:
        first_nan_after_cov = first_nan_after_cov[np.argwhere(first_nan_after_cov>last_nan_before_cov)[0][0]]
    except IndexError:
        first_nan_after_cov = cov_length

    last_nan_before_f = int(np.rint(freq_length/cov_length)*(last_nan_before_cov-n_points_offset))
    first_nan_after_f = int(np.rint(freq_length/cov_length)*first_nan_after_cov)

    covdata = cov_trace[last_nan_before_cov:first_nan_after_cov]
    
    if n_points_offset > 0:
        covdata = np.concatenate((np.ones(n_points_offset), covdata))
    fdata = freq_trace[last_nan_before_f:first_nan_after_f]
    # Interpolate frequency data to downsample
    freq_interp_f = interpolate.interp1d(np.arange(len(fdata)), fdata, fill_value="extrapolate")
    finterp = freq_interp_f(np.linspace(0, len(fdata), num=len(covdata), endpoint=False))
    xvals = np.arange(len(covdata))
    return covdata, finterp, xvals, last_nan_before_cov

def gen_minfun(xvals_sim, f_sim, cv_sim, xvals_data, f_data, cov_data, scale_factor, rel_weight=1):
    """ Generate function to be minimized in nonlinear optimization.
    """
    ccov = correct_cov(cv_sim, xvals_sim)
    
    # Generate lookup functions for simulated traces
    f_lookup = interpolate.interp1d(xvals_sim, f_sim, fill_value="extrapolate")
    cov_lookup = interpolate.interp1d(xvals_sim, ccov, fill_value="extrapolate")

    # Define function to be minimized
    def minfun(x):
        try:
            # Squared difference between simuled and experimental values. Hard-coded scale factor in x as an order of magnitude estimate to make individual optimization steps better behaved.
            cost = (np.sum(np.power(f_lookup(xvals_data/scale_factor/x[0] - x[1]) - f_data*x[2], 2)) \
                + rel_weight*np.sum(np.power(cov_lookup(xvals_data/scale_factor/x[0] - x[1]) - cov_data, 2)))/len(xvals_data)
        except Exception as e:
            print(xvals.shape)
            print(fdata.shape)
            print(covdata.shape)
            raise e
        return cost
    return minfun, f_lookup, cov_lookup


def denoise_svd(data_matrix, n_pcs, n_initial_components=100, skewness_threshold=0):
    """ SVD a data matrix and reconstruct using the first n singular components
    """
    u, s, v = randomized_svd(data_matrix, n_components=n_initial_components)

    use_pcs = np.zeros_like(s,dtype=bool)
    use_pcs[:n_pcs] = True
    
    skw = np.apply_along_axis(lambda x: stats.skew(np.abs(x)), 1, v)
    use_pcs = use_pcs & (skw > skewness_threshold)
    
    denoised = u[:,use_pcs]@ np.diag(s[use_pcs]) @ v[use_pcs,:]
    return denoised