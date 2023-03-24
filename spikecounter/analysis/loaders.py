""" Functions for loading intermediate data structures in analysis
"""
from parse import parse
from scipy import io as scio
import os
import numpy as np

from .. import utils

def load_matlab_simulation(rootdir, file, parse_string="sigma_{sigma:f}_r_{r:f}.mat"):
    """ Load saved data from ODE simulations
    """
    res = parse(parse_string, file)
    if res is not None:
        try:
            matres = scio.loadmat(os.path.join(rootdir, file))
        except Exception as e:
            print(file, e)
            return None
        r = utils.round_rel_deviation(matres['r'][0][0], factor=100)
    #         if r in valid_ticks:
        isi_mu= matres['isi_mu'][0][0]
        isi_std = matres['isi_std'][0][0]
        f = matres['f'][0][0]
        sigma = utils.round_rel_deviation(matres['s'][0][0], factor=100)
        n_peaks = matres['n_peaks'][0][0]
        isis = matres['all_isis'].ravel()
        return (r, sigma, isi_mu, isi_std, f, sigma, n_peaks, isis)
    else:
        return None
    
def load_python_simulation(rootdir, file, parse_string="sigma_{sigma:f}_I_{I:f}.npz"):
    params = parse(parse_string, file)
    if params is not None:
        try:
            res = np.load(os.path.join(rootdir, file))
        except Exception as e:
            print(file, e)
            return None
        
        return list(res.keys()), res