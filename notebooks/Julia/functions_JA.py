import matplotlib.path as mpath
import xarray as xr
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt


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

    
def count_2020_ARs(ds, lat_cut):
    if lat_cut<0:
        pole_ds = ds.sel(lat= slice(-90, lat_cut))
    else:
        pole_ds = ds.sel(lat= slice(lat_cut, 90))
    ar_counts = np.zeros(len(pole_ds.time))
    for i,ts in enumerate(pole_ds.time):
        ar_ts = pole_ds.sel(time = ts).ar_binary_tag.squeeze()
        ar_ts = xr.where(ar_ts>=1,1,0)
        ll = xr.plot.contour(ar_ts, levels=[0.0,1.0])
        plt.close()
        ar_list = np.array([len(p) for p in ll.collections[0].get_paths()])
        ar_counts[i] = len(ar_list[ar_list>20])
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

def plot_hist(flat_vars, seasons, plotting_vars, pole_ds, savename,skipmid=True):
    name_vars = ['Cloud fraction [%]', 'Precipitation [mm/day]', 'Surface temperature [K]', 'Downwelling longwave flux [W/m$^2$]']
    keys = ['Clean','Intermediate','Polluted']
    colors = ['tab:blue', 'tab:green','tab:orange']
    for season in seasons:
        fig = plt.figure(figsize=(12/4*(len(flat_vars)+1),3), dpi=150)
        fig.suptitle('Months: ' + str(season))
        for ivar, var in enumerate(flat_vars):
            ax = plt.subplot(1,len(flat_vars)+1,ivar+1)
            for ikey, key in enumerate(plotting_vars.keys()):
                if skipmid and key == 'mid' :
                    continue
                ds = plotting_vars[key]['ar_masked'].sel(time=(plotting_vars[key]['ar_masked'].time.dt.month.isin(season)))
                data = ds[var].values.reshape(-1,1)
                data = data[~np.isnan(data)]
                ax.hist(data, alpha = 0.6,bins=15, label=keys[ikey], weights=np.zeros_like(data) + 1. / data.size, color=colors[ikey])
            ax.set_xlabel(name_vars[ivar])
            if ivar in [0,1]:
                ax.set_yscale('log')
            if ivar==0:
                ax.legend()
                ax.set_ylabel('Frequency distribution')     
        ax = plt.subplot(1,len(flat_vars)+1,len(flat_vars)+1)
        counts_sum = pole_ds.sel(time=(pole_ds.time.dt.month.isin(season))).sum(dim='time')
        ax.pie([counts_sum[count] for count in ['clean_ar_counts','mid_ar_counts','poll_ar_counts']]
               ,labels = keys, colors = colors, autopct='%1.f%%',shadow=True)
        plt.tight_layout()
        plt.savefig(f'figures/{savename}_{season[0]}.png')  
    
        