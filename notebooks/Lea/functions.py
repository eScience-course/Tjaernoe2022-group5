
def to_nc(indata, lat_test, lon_test, tts, fln):
    from netCDF4 import Dataset
    nc = Dataset(fln, 'w')
    lat_dim = nc.createDimension('lat', indata.shape[1])
    lon_dim = nc.createDimension('lon', indata.shape[2])
    t_dim = nc.createDimension('time', indata.shape[0])
    lat_var = nc.createVariable('lat', np.float64, ('lat'))
    lat_var[:] = lat_test
    lon_var = nc.createVariable('lon', np.float64, ('lon'))
    lon_var[:] = lon_test
    tnd = nc.createVariable('ivt', np.float64, ('time','lat','lon'))
    tnd[:,:,:] = indata
    times = nc.createVariable('time', np.float64, ('time'))
    times[:] = tts
    nc.close()

def compute_ivt(q,v,p):
    import numpy as np
    g = 9.81
    pres_ = np.copy(p)
    pres_[pres_<25000] = np.nan # 900hPa to 300hPa in theory but extended to include enough model levels
    dp_ = np.append(np.diff(pres_),np.nan)
    dp_[np.isnan(dp_)]=0.0
    dp_=np.repeat(dp_[np.newaxis,:],q.shape[0],axis=0)
    dp_=np.repeat(dp_[:,:,np.newaxis],q.shape[2],axis=2)
    dp_=np.repeat(dp_[:,:,:,np.newaxis],q.shape[3],axis=3)
    p_ = pres_
    q_ = q
    v_ = v
    iv_ = -1/g*np.sum(q_*v_*dp_,axis=1)
    return iv_

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
        if len(ll.collections)>1: #You can remove this and next line if you have run it a few times without getting the printout :))
            print('julia was wrong about something, tell her to fix it'+ts)
        nr_ar = len(ll.collections[0].get_paths())
        ar_counts[i] = nr_ar
    ds[f'ar_counts_{lat_cut}']= ar_counts
    return ds