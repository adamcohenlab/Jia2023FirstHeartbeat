function batch_vm_to_tiff2(input_folder, output_folder, jog, curr_folder)
    subfolders = dir(input_folder);
    if curr_folder == 1
        fstring = split(strip(input_folder, 'right', '/'), '/');
        folder_name = fstring{size(fstring, 1)};
        if isfile(fullfile(input_folder, "output_data.mat"))
            metadata_to_py(input_folder);
        end
        mov = vm(input_folder).data;
        write_tif_stack3(mov, [output_folder '/' folder_name '.tif'], jog);
    end
    for i=3:length(subfolders)
        sf = [input_folder '/' subfolders(i).name];
        if ~isfolder(sf)
            continue
        end
        mov = vm(sf).data;
        write_tif_stack3(mov, [output_folder '/' subfolders(i).name '.tif'], jog);
    end
end