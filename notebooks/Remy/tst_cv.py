def contour_to_filled(in_file):
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    import xarray as xr
    from PIL import Image as im

    inpu = xr.open_dataset(in_file)

    out_ = inpu.ivt.values*0.0
    outout = inpu.copy()

    for i in range(len(inpu.time)):
        inloc = inpu.ivt.values[:,i,:]
        image = im.fromarray(inloc*100.0)
    	
        open_cv_image = np.array(image).astype(np.uint8)
	
        th, im_th = cv2.threshold(open_cv_image,80,120,cv2.THRESH_BINARY_INV)
	
        im_floodfill = im_th.copy()
        h,w = im_th.shape[:2]
        mask = np.zeros((h+2,w+2),np.uint8)
        new_image = cv2.floodFill(im_floodfill, mask, (0,0), (255,255,255))
	
        outimg = new_image[1]
        outimg[outimg==0] = 120
        outimg[outimg==255] = 0
        out_[:,i,:] = outimg

    outout.ivt = out_
    return outout

outy = contour_to_filled('2008_crit_p93_historical_cor.nc')
outy.to_netcdf('tstcv.nc')
