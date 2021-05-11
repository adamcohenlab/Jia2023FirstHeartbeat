import os
import numpy as np
from skimage import transform
from datetime import datetime
import matplotlib.pyplot as plt
from ipywidgets import interact

def extract_experiment_name(input_path):
    folder_names = input_path.split("/")
    if folder_names[-1] == "":
        expt_name = folder_names[-2]
    else:
        expt_name = folder_names[-1]
    expt_name = expt_name.split(".tif")[0]
    return expt_name

def generate_file_list(input_path):
    if os.path.isdir(input_path):
        raw_files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.splitext(f)[1] == ".tif"])
        files = []
        idx = 0
        while idx < len(raw_files):
            if "block" in raw_files[idx]:
                filename = os.path.splitext(os.path.basename(raw_files[idx]))[0].split("_block")[0]
                block = [raw_files[idx]]
                part_of_block = True
                while part_of_block and idx < len(raw_files)-1:
                    idx +=1
                    if filename in raw_files[idx]:
                        block += [raw_files[idx]]
                    else:
                        part_of_block = False
                        idx -= 1
                files += [block]
            else:
                files += [raw_files[idx]]
            idx +=1

    else:
        files = [input_path]
    return files

def make_output_folder(input_path=None, output_path=None, make_folder_from_file=False):
    if output_path is None:
        if os.path.isdir(input_path):
            output_folder = input_path
        else:
            if make_folder_from_file:
                output_folder = os.path.join(os.path.dirname(input_path), os.path.splitext(os.path.basename(input_path))[0])
            else:
                output_folder = os.path.dirname(input_path)
    else:
        output_folder = output_path
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.isdir(output_folder):
        raise Exception("Generated output path is not a folder")
    
    return output_folder

def write_subfolders(output_folder, subfolders):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    for subfolder in subfolders:
        try:
            os.mkdir(os.path.join(output_folder, subfolder))
        except Exception:
            pass

def standardize_n_dims(img, missing_dims=None):
    if missing_dims is not None:
        img = np.expand_dims(img, tuple(missing_dims))
    if len(img.shape) == 6:
        return img[:,0,:,:,:,:]
    n_axes_to_add = 5 - len(img.shape)
    if n_axes_to_add < 1:
        return img
    else:
        print(img.shape)
        print(np.arange(n_axes_to_add))
        return np.expand_dims(img, tuple(list(np.arange(n_axes_to_add))))

def img_to_8bit(img):
    img_8bit = img/np.max(img)*255
    return img_8bit.astype(np.uint8)

def project_y(img, z_to_x_ratio=1):
    print(img.shape)
    max_proj_y = np.expand_dims(img.max(axis=3), 3)
    max_proj_y = np.swapaxes(max_proj_y, 1, 3)
    # print(np.max(max_proj_y[:,:,0,:,:]))
    # print(np.max(max_proj_y[:,:,1,:,:]))
    # quit()
    max_proj_y_rescaled = np.zeros((max_proj_y.shape[0], max_proj_y.shape[1], max_proj_y.shape[2], int(np.round(max_proj_y.shape[3]*z_to_x_ratio)), max_proj_y.shape[4]), dtype=max_proj_y.dtype)
    for t in range(img.shape[0]):
        for c in range(img.shape[2]):
            max_proj_y_rescaled[t,0,c,:,:] = transform.resize(max_proj_y[t,0,c,:,:], (int(np.round(max_proj_y.shape[3]*z_to_x_ratio)), max_proj_y.shape[4]), preserve_range=True, order=3)
    max_proj_y_rescaled = np.flip(max_proj_y_rescaled, axis=3)
    return max_proj_y_rescaled

def project_x(img, z_to_y_ratio=1):
    print(img.shape)
    max_proj_x = np.expand_dims(img.max(axis=4), 4)
    max_proj_x = np.swapaxes(max_proj_x, 1, 4)
    # print(np.max(max_proj_x[:,:,0,:,:]))
    # print(np.max(max_proj_x[:,:,1,:,:]))
    # quit()
    max_proj_x_rescaled = np.zeros((max_proj_x.shape[0], max_proj_x.shape[1], max_proj_x.shape[2], max_proj_x.shape[3], int(np.round(max_proj_x.shape[4]*z_to_y_ratio))), dtype=max_proj_x.dtype)
    for t in range(img.shape[0]):
        for c in range(img.shape[2]):
            max_proj_x_rescaled[t,0,c,:,:] = transform.resize(max_proj_x[t,0,c,:,:], (max_proj_x.shape[3], (int(np.round(max_proj_x.shape[4]*z_to_y_ratio)))), preserve_range=True, order=3)
    return max_proj_x_rescaled

def max_entropy(self, raw_img):
    """
    Implements Kapur-Sahoo-Wong (Maximum Entropy) thresholding method
    Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) "A New Method for Gray-Level Picture Thresholding Using the Entropy
    of the Histogram", Graphical Models and Image Processing, 29(3): 273-285
    M. Emre Celebi
    06.15.2007
    Ported to ImageJ plugin by G.Landini from E Celebi's fourier_0.8 routines
    2016-04-28: Adapted for Python 2.7 by Robert Metchev from Java source of MaxEntropy() in the Autothresholder plugin
    http://rsb.info.nih.gov/ij/plugins/download/AutoThresholder.java
    :param data: Sequence representing the histogram of the image
    :return threshold: Resulting maximum entropy threshold
    """

    # calculate CDF (cumulative density function)
    data, _ = exposure.histogram(raw_img, normalize=True)
    cdf = data.astype(np.float).cumsum()

    # find histogram's nonzero area
    valid_idx = np.nonzero(data)[0]
    first_bin = valid_idx[0]
    last_bin = valid_idx[-1]

    # initialize search for maximum
    max_ent, threshold = 0, 0

    for it in range(first_bin, last_bin + 1):
        # Background (dark)
        hist_range = data[:it + 1]
        hist_range = hist_range[hist_range != 0] / cdf[it]  # normalize within selected range & remove all 0 elements
        tot_ent = -np.sum(hist_range * np.log(hist_range))  # background entropy

        # Foreground/Object (bright)
        hist_range = data[it + 1:]
        # normalize within selected range & remove all 0 elements
        hist_range = hist_range[hist_range != 0] / (cdf[last_bin] - cdf[it])
        tot_ent -= np.sum(hist_range * np.log(hist_range))  # accumulate object entropy

        # find max
        if tot_ent > max_ent:
            max_ent, threshold = tot_ent, it

    return threshold

def datestring_to_epoch(s, fmt="%Y-%m-%dT%H:%M:%S"):
    return datetime.strptime(s, fmt).timestamp()

def transferjob(sourcedir,targetdir):
    ## Adapted from Daniel Scott Eaton Trenchripper (https://github.com/DanielScottEaton/TrenchRipper)
    mkdircmd = "mkdir -p " + targetdir
    rsynccmd = "rsync -r " + sourcedir + "/ " + targetdir
    wrapcmd = mkdircmd + " && " + rsynccmd
    cmd = "sbatch -p transfer -t 0-12:00 --wrap=\"" + wrapcmd + "\""
    os.system(cmd)

def display_zstack(stack, z=0, c="all"):
    def view_image(z, c):
        if c=="all":
            img = stack[z,:,:,:]
        else:
            img = stack[z,:,:,int(c)]
        plt.imshow(img, interpolation="nearest")
        plt.title("Z: %d C: %s" %(z, str(c)))
        plt.show()
    interact(view_image, z=(0,stack.shape[0]-1), c=["all"] + list(np.arange(stack.shape[-1])))