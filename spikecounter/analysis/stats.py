from scipy import stats, interpolate, optimize
from sklearn import neighbors
from sklearn.utils.extmath import randomized_svd
import numpy as np
import matplotlib.pyplot as plt

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
            # print(fliprs)
            # print(diff)
            raise e
    except IndexError:
        # If we can't find an increase find the first time that it's NaN (no peakss detected)
        idx = np.argwhere(np.isnan(reverse_cov)).ravel()[0]-1
    # set CV to 1
    reverse_cov[idx+1:] = 1
    # Moving average smoothing
    # reverse_cov[idx-ksize//2:idx+ksize//2] = np.convolve(reverse_cov, np.ones(ksize)/ksize, mode="same")[idx-ksize//2:idx+ksize//2]
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

def gen_fitfun(xvals_sim, f_sim, cv_sim, x_offset_pre, x_offset_post, scale_factors, relweight=1):
    ### Here covdata and fdata are Nx2 arrays of the experimental data
    
    ccov = correct_cov(cv_sim, xvals_sim)
    
    f_lookup = interpolate.interp1d(xvals_sim, f_sim, fill_value="extrapolate")
    cov_lookup = interpolate.interp1d(xvals_sim, ccov, fill_value="extrapolate")

    def fit_function(params, x=None, dat1=None, dat2=None):
        freq_valid = ~np.isnan(dat1)
        cov_valid = ~np.isnan(dat2)
        x_valid_freq = x[freq_valid]
        x_valid_cov = x[cov_valid]
        model1 = f_lookup(params['x_scale']/scale_factors[0]*(x_valid_freq-x_offset_pre) +\
                          params['x_offset']/scale_factors[2]-x_offset_post)
        model2 = cov_lookup(params['x_scale']/scale_factors[0]*(x_valid_cov-x_offset_pre) +\
                            params['x_offset']/scale_factors[2]-x_offset_post)
        resid1 = (params['amplitude']/scale_factors[1]*dat1[freq_valid] - model1)
        resid2 = (dat2[cov_valid] - model2)*relweight
        return np.concatenate((resid1, resid2))
    return fit_function, f_lookup, cov_lookup

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

def trajectory_variability_kde(x, y, nsamples=100, bandwidth=1, pady=0):
    """ Take a set of measurements x, y (x values do not have to match) and generate
    a mean trajectory with standard deviations using kernel density estimator.
    """
    
    data = np.array([x, y]).T
    kde = neighbors.KernelDensity(bandwidth=bandwidth)
    kde.fit(data)
    
    x_samples = np.linspace(np.min(x), np.max(x), num=nsamples)
    y_samples = np.linspace(np.min(y)-pady, np.max(y)+pady, num=nsamples)
    
    data_samples = np.array([np.tile(x_samples, (nsamples,1)), np.tile(y_samples, (nsamples,1)).T])
    # Calculate densities
    probability_densities = np.zeros((nsamples, nsamples))
    for i in range(nsamples):
#         print(data_samples[:,:,i].T)
        probability_densities[i,:] = np.exp(kde.score_samples(data_samples[:,:,i].T))
#         print(probability_densities)
    # plt.imshow(probability_densities)
    normalization = np.sum(probability_densities, axis=1)
    # Marginalize along y
    mean_y = np.sum(y_samples[np.newaxis,:]*probability_densities, axis=1)/normalization
    var_y = np.sum((y_samples-mean_y)[np.newaxis,:]**2*probability_densities, axis=1)/normalization
    std_y = np.sqrt(var_y)
    
    return x_samples, mean_y, std_y


def multi_regress(data_matrix, traces, regress_dc=True):
    """
    data_matrix : t by x matrix
    traces: n by t matrix
    """
    if len(traces.shape) == 1:
        tr = traces[np.newaxis,:]
    else:
        tr = traces
        
    if regress_dc:
        I = np.concatenate([np.ones((1,data_matrix.shape[0])), tr-tr.mean(axis=1)], axis=0)
    else:
        I = tr
    
    C = data_matrix.T @ I.T @ np.linalg.inv(I @ I.T)
    resid = data_matrix - (C @ I).T
    
    if regress_dc:
        dc = np.outer(I[0,:], C[:,0])
    else:
        dc = np.zeros_like(data_matrix)
    return dc + resid

def fit_sigmoid(xs, ys, fixed_amplitude=None):
    if fixed_amplitude:
        def sigmoid(x, tau, x0):
            return fixed_amplitude/(1 + np.exp(-(x - x0)/tau))
    else:
        def sigmoid(x, tau, x0, a):
            return a/(1 + np.exp(-(x - x0)/tau))
    popt, _ = optimize.curve_fit(sigmoid, xs, ys)
    return popt