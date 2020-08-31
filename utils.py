import os
import numpy as np

def extract_experiment_name(input_path):
    folder_names = input_path.split("/")
    if folder_names[-1] == "":
        expt_name = folder_names[-2]
    else:
        expt_name = folder_names[-1]
    expt_name = expt_name.split(".tif")[0]
    return expt_name

def write_subfolders(output_folder, subfolders):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    for subfolder in subfolders:
        try:
            os.mkdir(os.path.join(output_folder, subfolder))
        except Exception:
            pass

def standardize_n_dims(img):
    n_axes_to_add = 5 - len(img.shape)
    if n_axes_to_add < 1:
        return img
    else:
        print(img.shape)
        print(np.arange(n_axes_to_add))
        return np.expand_dims(img, tuple(list(np.arange(n_axes_to_add))))