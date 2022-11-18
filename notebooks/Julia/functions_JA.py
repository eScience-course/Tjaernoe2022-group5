import xarray as xr
import pandas as pd
import numpy as np


def count_ARs(ds, lat_cut):
    '''
    Counts number of AR at each time step. Returns dataset with added coordinate ar_counts_[lat_cut].
    '''
    if lat_cut<0:
        pole_ds = ds.sel(lat= slice(-90, lat_cut))
    else:
        pole_ds = ds.sel(lat= slice(lat_cut, 90))

    ar_counts = np.zeros(len(pole_ds.time))
    for i,ts in enumerate(pole_ds.time):
        ll = xr.plot.contour(pole_ds.sel(time=ts).ivt, levels=[0.0,1.0])
        plt.close()
        nr_ar = len(ll.collections[0].get_paths())
        ar_counts[i] = nr_ar
    ds[f'ar_counts_{lat_cut}']= (['time'], ar_counts)
    return ds

def circle_for_polar_map(axes):
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    axes.set_boundary(circle, transform=axes.transAxes)
    
def sort_ar_by_aod(aod_ds,ar_ds, poll_lim, clean_lim):
    warnings.simplefilter('ignore', UserWarning)
    aod_ar=aod_ds.where(ar_ds.ivt==1)
    aod_ar['poll_ar_aod'] = xr.DataArray(coords=aod_ar.coords, dims =aod_ar.dims)
    aod_ar['clean_ar_aod'] = xr.DataArray(coords=aod_ar.coords, dims =aod_ar.dims)
    aod_ar['mid_ar_aod'] = xr.DataArray(coords=aod_ar.coords, dims =aod_ar.dims)
    clean_ar_counts = np.zeros(len(aod_ar.time))
    poll_ar_counts = np.zeros(len(aod_ar.time))
    mid_ar_counts = np.zeros(len(aod_ar.time))

    for i,ts in enumerate(aod_ar.time):
        ts_array = aod_ar.sel(time=ts).od550aer
        ll = xr.plot.contourf(ts_array.squeeze(), levels=[0,20])
        plt.close()
        ar_paths = ll.collections[0].get_paths()
        for j,item in enumerate(ar_paths):
            v = item.vertices
            lat = v[:,1]
            lon = v[:,0]
            ar_i_aod = ts_array.sel(lat=slice(np.min(lat), np.max(lat)), lon = slice(np.min(lon), np.max(lon)))
            
            if ar_i_aod.mean(skipna=True).values>poll_lim.values:
                aod_ar['poll_ar_aod'].loc[ts,slice(np.min(lat), np.max(lat)),slice(np.min(lon), np.max(lon))] = ar_i_aod
                poll_ar_counts[i]=poll_ar_counts[i]+1
            elif ar_i_aod.mean(skipna=True).values< clean_lim.values :
                aod_ar['clean_ar_aod'].loc[ts,slice(np.min(lat), np.max(lat)),slice(np.min(lon), np.max(lon))] = ar_i_aod
                clean_ar_counts[i]=clean_ar_counts[i]+1
            else:
                aod_ar['mid_ar_aod'].loc[ts,slice(np.min(lat), np.max(lat)),slice(np.min(lon), np.max(lon))] = ar_i_aod
                mid_ar_counts[i]=mid_ar_counts[i]+1
    aod_ar['clean_ar_counts']= (['time'], clean_ar_counts)
    aod_ar['mid_ar_counts']= (['time'], mid_ar_counts)
    aod_ar['poll_ar_counts']= (['time'], poll_ar_counts)
    return aod_ar