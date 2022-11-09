import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as st

distrib = st.mielke

params_init =[25.958341361875725, 6.033654880661353, -0.03094463517841417, 0.0628311337711128]

def delta(tmax,fit_params_hist,fit_params_rcp):
    delta = tmax/qm(tmax,fit_params_rcp,fit_params_hist)
    perc = distrib.cdf(tmax,*fit_params_rcp)
    return delta

def qm(tmax,fit_params_,fit_params_era5):
    k,s,loc,scale = fit_params_
    k_,s_,loc_,scale_ = fit_params_era5
    cdf_hist = distrib.cdf(tmax,*fit_params_)
    tmax_qm = distrib.ppf(cdf_hist,*fit_params_era5)
    return tmax_qm

def qdm(tmax,fut==False):
    tqdm = np.zeros(np.shape(tmax))
    ntimes = np.shape(tqdm)[0]
    nlats = np.shape(tqdm)[1]
    nlons = np.shape(tqdm)[2]
    if fut==True:
        delta_loc = delta(tmax.flatten(),params,params_fut)
        loc_qm = qm(tmax.flatten(),params_fut,params_init)
        loc_pm = delta_loc*loc_qm
    else:
        loc_qm = qm(tmax.flatten(),params,params_init)
    tqdm = loc_qm.reshape(ntimes,nlats,nlons)
    tqdm[np.isnan(tqdm)]=0
    tqdm[np.isinf(tqdm)]=0
    return tqdm

# flattened historical data here                                                                             
params = distrib.fit(histo)

# flattened future data here                                                                                 
params = distrib.fit(fut)

ssp_corrected = qdm(sspdata,True)
hist_corrected = qdm(histdata,False)