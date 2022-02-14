function metadata_to_py(folder)
    Device_Data = load(fullfile(folder, 'output_data.mat'));
    Device_Data = Device_Data.Device_Data;
    
    daq = load_device_data(Device_Data, "DAQ");
    dmd_fast = load_device_data(Device_Data, "DMD_fast");
    if isempty(dmd_fast)
        dmd_fast = load_device_data(Device_Data, "DMD");
    end
    camera = load_device_data(Device_Data, "Main Camera");
    dmd_lightcrafter = load_device_data(Device_Data, "DMD_lightcrafter");
    
    
    
    dd_compat_py = struct;
    dd_compat_py.clock_rate = daq.buffered_tasks(1).rate;
    dd_compat_py.task_traces = struct('task_type',{},'traces',{});
    dd_compat_py.frame_counter = daq.Counter_Inputs.data;
    for i = 1:length(daq.buffered_tasks)
        dd_compat_py.task_traces(i).task_type = daq.buffered_tasks(i).task_type;
        traces = daq.buffered_tasks(i).channels;
        for j=1:length(traces)
            dd_compat_py.task_traces(i).traces(j) = ...
            struct('name', traces(j).name, 'values', traces(j).data);
        end
    end
    
    dd_compat_py.dmd_fast = struct;
    dd_compat_py.dmd_fast.target = dmd_fast.Target;
    dd_compat_py.dmd_fast.tform = dmd_fast.tform.T;
    
    invm = dmd_fast.tform.T^-1;
    invm(:,3) = [0;0;1];
    dd_compat_py.dmd_fast.target_image_space = imwarp(dmd_fast.Target, affine2d(invm), ...
        'OutputView', imref2d([2048, 2048]));
    dd_compat_py.camera = struct;
    dd_compat_py.camera.exposuretime = camera.exposuretime;
    dd_compat_py.camera.frames_requested = camera.frames_requested;
    dd_compat_py.camera.roi = camera.ROI;
    
    if ~isempty(dmd_lightcrafter)
       dd_compat_py.dmd_lightcrafter = struct;
       dd_compat_py.dmd_lightcrafter.target = dmd_lightcrafter.Target;
       dd_compat_py.dmd_lightcrafter.tform = dmd_lightcrafter.tform.T;
       
       
        invm = dmd_lightcrafter.tform.T^-1;
        invm(:,3) = [0;0;1];
        dd_compat_py.dmd_lightcrafter.target_image_space = imwarp(dmd_lightcrafter.Target, affine2d(invm), ...
            'OutputView', imref2d([2048, 2048]));
    end
    save(fullfile(folder, 'output_data_py.mat'), 'dd_compat_py', '-v7.3');
end

function device = load_device_data(dd, device_name)
    for i = 1:length(dd)
       if isfield(dd{i}, "Device_Name") && dd{i}.Device_Name == device_name
           device = dd{i};
           return;
       end
    end
    device = {};
end