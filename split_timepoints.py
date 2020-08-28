import skimage.io as io
import numpy as np
stack = io.imread("Arclight_memCherry_Overnight_Timelapse.tif")
print(stack.shape)

timepoints = np.arange(24)

for timepoint in timepoints:
    arclight = stack[timepoint,:,0,:,:]
    mch = stack[timepoint,:,1,:,:]
    
    ratio = arclight/mch
    ratio[np.isnan(ratio)] = 0
    ratio[np.isinf(ratio)] = 0
    ratio = ratio.astype(np.float32)
    io.imsave("ratios/t%d_ratio_raw.tif" % timepoint, ratio)
    io.imsave("arcLight/t%d.tif" % timepoint, arclight)
    io.imsave("memCh/t%d.tif" % timepoint, mch)