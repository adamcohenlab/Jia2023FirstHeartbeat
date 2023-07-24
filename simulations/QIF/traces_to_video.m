function y_img = traces_to_video(traces, ny, nx, scale_y, scale_x)
    y_img = reshape(traces, size(traces,1), ny, nx);
    y_img = imresize3(y_img, [size(traces,1) uint32(ny*scale_y) uint32(nx*scale_x)]);
    y_img = permute(reshape(y_img, [size(y_img), 1]), [2,3,4,1]);
    y_img = im2uint8((y_img - min(y_img(:)))/(max(y_img(:)) - min(y_img(:))));
end