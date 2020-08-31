#! /usr/bin/python3

import bioformats
import javabridge
import os
import skimage.io as skio
import numpy as np
import tifffile

javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)

input_path = "/mnt/d/Documents/Cohen_Lab/Data/20200826_arclightq239_asap3_comparison"
# metadata_path = 

for f in os.listdir(input_path)[-2:]:
    full_path = os.path.join(input_path, f)
    filename, extname = os.path.splitext(f)
    imagej_metadata = """ImageJ=1.47a
images={nr_images}
channels={nr_channels}
slices={nr_slices}
hyperstack=true
mode=color
loop=false"""

    if not os.path.exists(os.path.join(input_path, filename)):
        os.mkdir(os.path.join(input_path, filename))
    if not os.path.exists(os.path.join(input_path, filename, "maxproj")):
        os.mkdir(os.path.join(input_path, filename, "maxproj"))
    if not os.path.isdir(full_path):
        reader = bioformats.ImageReader(full_path)
        o = bioformats.OMEXML(bioformats.get_omexml_metadata(full_path))
        n_positions = o.get_image_count()
    
        metadata_path = os.path.join(input_path, filename, "%s_metadata.txt" % filename)
        position_metadata = {}
        maxproj_metadata = {}

        for pos in range(n_positions):
            pixel_data = o.image(index=pos).Pixels
            n_t = pixel_data.SizeT
            n_z = pixel_data.SizeZ
            n_x = pixel_data.SizeX
            n_y = pixel_data.SizeY
            position_filename =  "%s_pos%d.tif" % (filename, (pos+1))
            maxproj_filename = "%s_pos%d_MAX.tif" % (filename, (pos+1))
            plane_info = pixel_data.Plane(index=0)
            position_metadata[position_filename] = "(%.2f, %.2f, %.2f)" % (plane_info.PositionX, plane_info.PositionY, plane_info.PositionZ)
            maxproj_metadata[maxproj_filename] = position_metadata[position_filename]
            if n_t <=1:
                curr_pos = np.zeros((n_z, 3, n_y, n_x), dtype=np.uint8)
                for z in range(n_z):
                    curr_slice = np.moveaxis(reader.read(z=z, series=pos, rescale=False), [0,1,2], [2,1,0])
                    curr_pos[z,:,:,:] = curr_slice
                tifffile.imwrite(os.path.join(input_path, filename, "maxproj", "%s_pos%d_MAX.tif" % (filename, (pos+1))), curr_pos.max(axis=0), imagej=True, resolution=(1/pixel_data.PhysicalSizeX, 1/pixel_data.PhysicalSizeY), metadata={'unit': 'um'})
            else:
                curr_pos = np.zeros((n_t, n_z, 2, n_y, n_x), dtype=np.uint8)
                for z in range(n_z):
                    for t in range(n_t):
                        curr_slice = np.moveaxis(reader.read(z=z, series=pos, rescale=False, t=t), [0,1,2], [2,1,0])
                        curr_pos[t,z,:,:,:] = curr_slice
                tifffile.imwrite(os.path.join(input_path, filename, "maxproj", "%s_pos%d_MAX.tif" % (filename, (pos+1))), curr_pos.max(axis=1), imagej=True, resolution=(1/pixel_data.PhysicalSizeX, 1/pixel_data.PhysicalSizeY), metadata={'unit': 'um'})

            
            tifffile.imwrite(os.path.join(input_path, filename, position_filename), curr_pos, imagej=True, resolution=(1/pixel_data.PhysicalSizeX, 1/pixel_data.PhysicalSizeY), metadata={'spacing': pixel_data.PhysicalSizeZ, 'unit': 'um'})

        with open(metadata_path, "w+") as meta_file:
            for key, value in position_metadata.items():
                meta_file.write("%s;;%s\n" % (key, value))
        with open(os.path.join(input_path, filename, "maxproj", "%s_coordinates.txt" % filename), "w+") as meta_file:
            for key, value in maxproj_metadata.items():
                meta_file.write("%s;;%s\n" % (key, value))
    
javabridge.kill_vm()