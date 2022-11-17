def bias_correction(histo,fut,scn):
    import matplotlib.pyplot as plt
    import netCDF4
    import numpy as np
    import pandas as pd
    import xarray as xr
    import scipy.stats as st
    distrib = st.mielke
    params_init =[25.958341361875725, 6.033654880661353, -0.03094463517841417, 0.0628311337711128]
    params = [159.41596223491013, 26.727408136167004, -1.1915471684285064, 2.1257486429240604]
    params_245 = [272.7567596440349, 36.95819190681676, -1.7772972302143981, 2.7113549002526094]
    params_585 = [234.1123886188019, 25.47272752571952, -1.0582716717988345, 1.9722684979237837]
    params_370 = [212.22386932513837, 31.08540546147708, -1.5705302497093339, 2.508484987028508]

    if scn=='245':
        params_fut = params_245
    elif scn=='370':
        params_fut = params_370
    elif scn=='585':
        params_fut = params_585

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

    def qdm(tmax,fut=False):
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


    ssp_corrected = qdm(fut,True)
    hist_corrected = qdm(histo,False)

    return hist_corrected,ssp_corrected