#! /usr/bin/python3
import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import skimage.filters as filters
import skimage.morphology as morph
import scipy.stats as stats
import scipy.signal as signal
import scipy.ndimage as ndimage
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("report.mplstyle")

spike_mask = imread("spikes_opened1.25.tif")
raw_image = imread("test.tif")

n_timepoints = raw_image.shape[0]

mean_threshold = 1
mean_time = np.mean(raw_image, axis=0)

all_spikes_detected = spike_mask.max(axis=0).astype(bool)

variance = np.var(raw_image, axis=0)/mean_time
variance_spikes = variance[all_spikes_detected]
variance_notspikes = variance[~all_spikes_detected]
variance_spikes[variance_spikes > 6] = 5.95
variance_notspikes[variance_notspikes > 6] = 5.95


fig1, ax1 = plt.subplots(figsize=(10,6))
ax1.hist(variance_spikes, label="Spikes", alpha=0.5, bins=np.linspace(0, 6, 60), density=True)
ax1.hist(variance_notspikes, label="Background", alpha=0.5, bins=np.linspace(0, 5, 60), density=True)
ax1.set_xlabel("Coefficient of Variation")
ax1.set_ylabel("PDF")
ax1.legend()
plt.tight_layout()
plt.savefig("variance_distribution.svg")
plt.show()