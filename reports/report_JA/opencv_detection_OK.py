import xarray as xr
xr.set_options(display_style='html')
import intake
import cftime
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image as im

def compute_ivt1(ivx):
    g = 9.81
    iv_ = -1/g*ivx.qv.integrate(coord='plev')
    return iv_


def contour_to_filled(in_file):
    inpu = in_file

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

    outout.ivt[:] = out_
    return outout


years = np.arange(1990,2015,1).astype(int)
ssp = 'historical'

cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"


col = intake.open_esm_datastore(cat_url)

cat = col.search(source_id=['NorESM2-LM'], experiment_id=[ssp], table_id=['day'], variable_id=['hus','va'], member_id=['r1i1p1f1'])
dset_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})

dataset_list = list(dset_dict.keys())
dset = dset_dict[dataset_list[0]]

q98 = xr.open_dataset('q94_2000.nc')
q98 = q98.rename({'__xarray_dataarray_variable__':'ivt'})

for year in years:
    dset = dset_dict[dataset_list[0]]
    dset = dset.sel(member_id='r1i1p1f1',time=slice(str(year)+"-01-01", str(year)+"-12-31"))
    vas = dset.va
    hus = dset.hus
    plev = dset.plev
    lat_ = hus.lat
    lon_ = hus.lon
    dset['qv'] = dset.va*dset.hus
    qv = dset.qv
    qv = xr.where(qv.plev>=25000,qv,0)
    dset['qv'] = qv
    ivt = compute_ivt1(dset)
    dset.close()
    ivt_ns = ivt.copy()
    ivt_ns = xr.where(ivt_ns.lat<0,-ivt_ns,ivt_ns,True)
    ivt_ns_pos = xr.where(ivt_ns<0,ivt_ns*0,ivt_ns,True)
    excess = ivt_ns_pos-q98
    ivt_ns_pos.close()
    ar_points = xr.where(excess>0,1,0)
    out_ar = ar_points.copy()
    ar_points.close()
    out_ar = out_ar.squeeze()
    out_ar = out_ar.drop_vars(['quantile','member_id','dcpp_init_year'])
    out_loc = np.zeros((out_ar.ivt.shape[0],out_ar.ivt.shape[1],out_ar.ivt.shape[2])).astype(int)
    test_val = out_ar.ivt.values[:]
    for tt in range(len(out_ar.time)):
        df_loc = test_val[:,tt,:]
        ll = plt.contour(df_loc,levels=[0,1])
        plt.close()
        for item in ll.collections:
            for i in item.get_paths():
                v = i.vertices
                crit = abs(np.max(v[:, 1])-np.min(v[:, 1]))
                if (crit>=20):
                    xx=(v[:, 0]).astype(int)
                    yy=(v[:, 1]).astype(int)
                    for (x,y) in zip(xx,yy):
                        out_loc[y,tt,x] = 1

    out_ar.ivt.values = out_loc
    outy = contour_to_filled(out_ar)
    outy['ivt'] = outy.ivt.astype(bool)
    outy.to_netcdf(ssp+'_'+str(year)+'q94.nc')
