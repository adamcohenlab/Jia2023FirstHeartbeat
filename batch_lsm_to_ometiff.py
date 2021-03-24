#! /usr/bin/python3

import os
import skimage.io as skio
import numpy as np
import tifffile
import xmltodict
import bioformats
import javabridge
import argparse
import xml.etree.ElementTree as ET
from xmlformatter import Formatter


parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input folder")

args = parser.parse_args()
# Locate the XML file
input_path = args.input

javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)

formatter = Formatter(indent="1", indent_char="\t", preserve=["literal"])


maxproj_dir = os.path.join(input_path, "maxproj")
if not os.path.exists(maxproj_dir):
    os.mkdir(maxproj_dir)

for f in os.listdir(input_path):
    full_path = os.path.join(input_path, f)
    if not os.path.isdir(full_path):
        filename, extname = os.path.splitext(f)
        is_ometiff = False
        if filename[-4:] == ".ome":
            filename = filename[:-4]
            is_ometiff = True
        
        if extname == ".lsm":
            try:
                lsm = tifffile.imread(full_path)
                print(lsm.shape)
                o = bioformats.OMEXML(bioformats.get_omexml_metadata(full_path))
                with open(os.path.join(input_path, "%s_meta.xml" % filename), "w+") as xml_file:
                    xml_file.write(formatter.format_string(o.to_xml()).decode("utf-8"))
                pixel_data = o.image(index=0).Pixels
                # print(dir(pixel_data))
                print(pixel_data.get_SizeC())
                print(pixel_data.get_SizeT())
                print(pixel_data.get_SizeZ())
                # exit()

                if len(lsm.shape) == 6:
                    lsm = lsm[0,:,:,:,:,:]

                if pixel_data.get_SizeZ() == 1 and pixel_data.get_SizeT() > 1:
                    if lsm.shape[1] > 1:
                        lsm = np.swapaxes(lsm, 0, 1)
                    else:
                        lsm = np.swapaxes(lsm, 0, 2)

                    
                elif pixel_data.get_SizeZ() > 1 and pixel_data.get_SizeC() == 1:
                    lsm = np.swapaxes(lsm, 1, 2)
                z_axis = 1
                print("New shape")
                print(lsm.shape)
                tifffile.imsave(os.path.join(input_path, "%s.tif" % filename), lsm, imagej=True, metadata={'spacing': pixel_data.PhysicalSizeZ, 'unit': 'um'}, resolution=(1/pixel_data.PhysicalSizeX, 1/pixel_data.PhysicalSizeY))
            except Exception as e:
                print(e)
                continue
        elif is_ometiff:
            lsm = tifffile.imread(full_path)
            tf = tifffile.TiffFile(full_path)
            tif_tags = {}
            o = bioformats.OMEXML(bioformats.get_omexml_metadata(full_path))
            xml = ET.fromstring(o.to_xml())[0][0].text

            o = bioformats.OMEXML(xml)
            with open(os.path.join(input_path, "%s_meta.xml" % filename), "w+") as xml_file:
                xml_file.write(formatter.format_string(xml).decode("utf-8"))
            # print(ET.tostring(xml[0][0])[:50])
            # o = bioformats.OMEXML(xml=ET.tostring(xml[0]))
            # with open(os.path.join(input_path, "%s_meta.xml" % filename), "w+") as xml_file:
            #     xml_file.write(formatter.format_string(o.to_xml()).decode("utf-8"))
            pixel_data = o.image(index=0).Pixels
            # print(dir(pixel_data))
            # print(pixel_data.get_SizeC())
            # print(pixel_data.get_SizeT())
            # print(pixel_data.get_SizeZ())
            # exit()
            lsm = np.moveaxis(lsm, [0,3,1,2],[0,1,2,3])
            lsm = np.expand_dims(lsm, 1)
            print(lsm.shape)

            tifffile.imsave(os.path.join(input_path, "%s.tif" % filename), lsm, imagej=True)
        elif extname == ".czi":
            # try:
            czi = bioformats.ImageReader(full_path)

            o = bioformats.OMEXML(bioformats.get_omexml_metadata(full_path))
            with open(os.path.join(input_path, "%s_meta.xml" % filename), "w+") as xml_file:
                xml_file.write(formatter.format_string(o.to_xml()).decode("utf-8"))
            pixel_data = o.image(index=0).Pixels

            n_t = pixel_data.get_SizeT()
            n_z = pixel_data.get_SizeZ()
            n_c = pixel_data.get_SizeC()
            n_x = pixel_data.get_SizeX()
            n_y = pixel_data.get_SizeY()

            lsm = np.zeros((n_t, n_z, n_c, n_y, n_x), dtype=np.uint8)
            
            for t in range(n_t):
                for z in range(n_z):
                    for c in range(n_c):
                        lsm[t,z,c,:,:] = czi.read(c=c, t=t, z=z, rescale=False)
            if len(lsm.shape) == 6:
                lsm = lsm[0,:,:,:,:,:]

            z_axis = 1

            tifffile.imsave(os.path.join(input_path, "%s.tif" % filename), lsm, imagej=True, metadata={'spacing': pixel_data.PhysicalSizeZ, 'unit': 'um'}, resolution=(1/pixel_data.PhysicalSizeX, 1/pixel_data.PhysicalSizeY))
            # except Exception as e:
            #     print(e)
            #     continue
javabridge.kill_vm()