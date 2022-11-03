
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
    pres_[~np.isnan(pres_)] = 1.0
    pres = np.where(pres_==1.0)
    dp_ = np.diff(pres_)
    print(dp_)
    dp_=np.zeros(len(pres[0]))
    for kk in range(len(dp_)):
        dp_[kk] = -p[pres[0][0]+kk+1]+p[pres[0][0]+kk]
    dp_=np.repeat(dp_[np.newaxis,:],q.shape[0],axis=0)
    dp_=np.repeat(dp_[:,:,np.newaxis],q.shape[2],axis=2)
    dp_=np.repeat(dp_[:,:,:,np.newaxis],q.shape[3],axis=3)
    p_ = pres_
    q_ = q[:,pres[0],:,:]
    v_ = v[:,pres[0],:,:]
    iv_ = -1/g*np.sum(q_*v_*dp_,axis=1)
    return iv_
