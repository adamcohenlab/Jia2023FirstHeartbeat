### Downsample and denoise videos
import argparse
import os
import skimage.io as skio
import numpy as np
from sklearn.utils.extmath import randomized_svd
from skimage import transform
from scipy import ndimage

parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="Input file")
parser.add_argument("output_folder", help="Output folder for results", default=None)
parser.add_argument("--scale_factor", help="Scale factor for downsampling", default=2, type=float)
parser.add_argument("--n_pcs", help="Number of PCs to keep", default=50, type=int)
parser.add_argument("--remove_from_start", help="Time indices to trim from start", default=0, type=int)
parser.add_argument("--remove_from_end", help="Time indices to trim from end", default=0, type=int)

args = parser.parse_args()
input_file = args.input_file
output_folder = args.output_folder
scale_factor = args.scale_factor
n_pcs = args.n_pcs
remove_from_start = args.remove_from_start
remove_from_end = args.remove_from_end

filename = os.path.splitext(os.path.basename(input_file))[0]


os.makedirs(os.path.join(output_folder, "denoised"), exist_ok=True)

img = skio.imread(input_file)
print(img.shape)

trimmed = img[remove_from_start:img.shape[0]-remove_from_end]
print(trimmed.shape)

# LP filter then downsample
sigma = scale_factor
if scale_factor > 1:
    os.makedirs(os.path.join(output_folder, "downsampled"), exist_ok=True)
    smoothed = ndimage.gaussian_filter(trimmed, [1,sigma,sigma])
    print(smoothed.shape)
    downsampled = smoothed[:,np.arange(smoothed.shape[1], step=sigma, dtype=int),:]
    print(downsampled.shape)
    downsampled = downsampled[:,:,np.arange(downsampled.shape[2], step=sigma, dtype=int)]
    print(downsampled.shape)

    skio.imsave(os.path.join(output_folder, "downsampled", os.path.basename(input_file)), np.round(downsampled).astype(np.uint16))
else:
    downsampled = trimmed
    
mean_img = downsampled.mean(axis=0)

# Zero the mean over time
t_zeroed = downsampled - mean_img
data_matrix = t_zeroed.reshape((t_zeroed.shape[0], -1))

# correlate to mean traace
mean_trace = data_matrix.mean(axis=1)
corr = np.matmul(data_matrix.T, mean_trace)/np.dot(mean_trace, mean_trace)
resids = data_matrix - np.outer(mean_trace, corr)

# SVD
u, s, v = randomized_svd(resids, n_components=100)

denoised = u[:,:n_pcs]@ np.diag(s[:n_pcs]) @ v[:n_pcs,:]
denoised = denoised.reshape(downsampled.shape)

# Add back DC offset for the purposes of comparing noise to mean intensity
denoised += mean_img
skio.imsave(os.path.join(output_folder, "denoised", os.path.basename(input_file)), np.round(denoised).astype(np.uint16))
