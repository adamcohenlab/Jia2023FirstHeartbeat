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

def generate_file_list(input_path):
    if os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.splitext(f)[1] == ".tif"]
    else:
        files = [input_path]
    return files

def make_output_folder(input_path=None, output_path=None):
    if output_path is None:
        if os.path.isdir(input_path):
            output_folder = input_path
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

def standardize_n_dims(img):
    n_axes_to_add = 5 - len(img.shape)
    if n_axes_to_add < 1:
        return img
    else:
        print(img.shape)
        print(np.arange(n_axes_to_add))
        return np.expand_dims(img, tuple(list(np.arange(n_axes_to_add))))