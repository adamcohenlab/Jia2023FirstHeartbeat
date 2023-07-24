function batch_metadata_to_py(folder)
    filelist = dir(folder)
    for i = 3:length(filelist)
        if isfile(fullfile(folder, filelist(i).name, "output_data.mat"))
            disp(filelist(i).name);
            metadata_to_py(fullfile(folder, filelist(i).name));
        end
    end
end