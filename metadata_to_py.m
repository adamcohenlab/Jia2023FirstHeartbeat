function metadata_to_py(folder)
    Device_Data = load(fullfile(folder, 'output_data.mat'));
    Device_Data = Device_Data.Device_Data;
    
    daq = load_device_data(Device_Data, device_type="DAQ_Session");
    daq = daq{1};
    
    dmds = load_device_data(Device_Data, device_type="DMD");
    cameras = load_device_data(Device_Data, device_type="Cam_Controller");
    confocal_output_data = load_device_data(Device_Data, device_name="BU_2P");
    
    dd_compat_py = struct;
    dd_compat_py.dmds = {};
    dd_compat_py.cameras = {};
    
    if ~isempty(confocal_output_data) && ~isempty(confocal_output_data.outputdata)
        disp(confocal_output_data.outputdata);
        dd_compat_py.confocal = struct;
        dd_compat_py.confocal.galvofbx = confocal_output_data.outputdata.galvofbx;
        dd_compat_py.confocal.galvofby = confocal_output_data.outputdata.galvofby;
        dd_compat_py.confocal.PMT = confocal_output_data.outputdata.PMT;
        dd_compat_py.confocal.stage_position = confocal_output_data.outputdata.stage_position;
        if isfield(confocal_output_data, 'xdata')
            dd_compat_py.confocal.xdata = confocal_output_data.xdata;
            dd_compat_py.confocal.ydata = confocal_output_data.ydata;
            dd_compat_py.confocal.points_per_line = confocal_output_data.points_per_line;
            dd_compat_py.confocal.numlines = confocal_output_data.numlines;
        elseif isfield(confocal_output_data, 'galvox_wfm')
            dd_compat_py.confocal.xdata = confocal_output_data.galvox_wfm;
            dd_compat_py.confocal.ydata = confocal_output_data.galvoy_wfm;
            dd_compat_py.confocal.points_per_line = length(unique(round(confocal_output_data.galvox_wfm, 6)))-1;
            dd_compat_py.confocal.numlines = length(unique(round(confocal_output_data.galvoy_wfm, 6)));
        end
    end
    
    if ~isempty(daq)
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
    end
    for dmd = dmds
       dd_compat_py.dmds{end+1} = dmd_to_py(dmd{:}); 
    end
    
    for camera = cameras
        dd_compat_py.cameras{end+1} = camera_to_py(camera{:});
    end
    
    save(fullfile(folder, 'output_data_py.mat'), 'dd_compat_py', '-v7.3');
end

function devices = load_device_data(dd, options)
    arguments
        dd;
        options.device_type = [];
        options.device_name = [];
    end
    
    device_type = options.device_type;
    device_name = options.device_name;
    
    devices = {};
    for i = 1:length(dd)
       if isfield(dd{i}, "Device_Name") && ~isempty(device_name) ...
               && dd{i}.Device_Name == device_name
           devices = dd{i};
           break
       elseif isfield(dd{i}, "Device_Type") && ...
           ~isempty(strfind(dd{i}.Device_Type,device_type))
           devices{end+1} = dd{i};
       end
    end
end

function py_dmd = dmd_to_py(dmd)
    py_dmd = struct;
    py_dmd.name = char(dmd.Device_Name);
    py_dmd.target = dmd.Target;
    if isempty(dmd.tform)
        py_dmd.tform = dmd.tform;
    else
        py_dmd.tform = dmd.tform.T;

        invm = dmd.tform.T^-1;
        invm(:,3) = [0;0;1];
        py_dmd.target_image_space = imwarp(dmd.Target, affine2d(invm), ...
            'OutputView', imref2d([2048, 2048]));        
    end

end

function py_camera = camera_to_py(camera)
    py_camera = struct;
    py_camera.name = char(camera.Device_Name);
    py_camera.max_size = camera.virtual_sensor_size;
    py_camera.exposuretime = camera.exposuretime;
    py_camera.frames_requested = camera.frames_requested;
    py_camera.roi = camera.ROI;
    if isfield(camera, 'dropped_frames')
        py_camera.dropped_frames = camera.dropped_frames;
    else
        py_camera.dropped_frames = 0;
    end
end