import scipy.stats as stats
import numpy as np

def ks_window(data, window, overlap=0.5):
    ts = []
    ks_p = []
    for i in np.arange(0, len(data)-window, step=int(window - overlap*window)):
        _, p = stats.kstest(data[i:i+window], 'expon', args=(0, np.mean(data[i:i+window])))
        ts.append(i)
        ks_p.append(p)
    ts = np.array(ts)
    return ts, np.array(ks_p)