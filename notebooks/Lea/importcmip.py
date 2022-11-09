#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import s3fs
xr.set_options(display_style='html')
import intake
import cftime
import numpy as np
from netCDF4 import Dataset
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy
from IPython import display

s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD", secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0", client_kwargs=dict(endpoint_url="https://rgw.met.no"))

# import AOD 
start_year0 = 2000
end_year0 = 2014
'''start_year = 2035
end_year = 2050'''
start_year2 = 2085
end_year2 = 2099

#historical
s3path = list([
 'escience2022/Remy/od550aer_AERday_NorESM2-LM_historical_r1i1p1f1_gn_20000101-20091231.nc',
 'escience2022/Remy/od550aer_AERday_NorESM2-LM_historical_r1i1p1f1_gn_20100101-20141231.nc',
])
sopenlist=[s3.open(ss) for ss in s3path]
aodh = (xr.open_mfdataset(sopenlist)).sel(time = slice(str(start_year0)+"-01-01", str(end_year0)+"-12-31"))

'''
#ssp245 2035-2050
s3path = list([
 'escience2022/Remy/od550aer_AERday_NorESM2-LM_ssp245_r1i1p1f1_gn_20310101-20401231.nc',
 'escience2022/Remy/od550aer_AERday_NorESM2-LM_ssp245_r1i1p1f1_gn_20410101-20501231.nc'
])
sopenlist=[s3.open(ss) for ss in s3path]
aod245_3550 = (xr.open_mfdataset(sopenlist)).sel(time = slice(str(start_year)+"-01-01", str(end_year)+"-12-31"))
'''
    
#ssp245 2085-2100
s3path = list([
    'escience2022/Remy/od550aer_AERday_NorESM2-LM_ssp245_r1i1p1f1_gn_20810101-20901231.nc',
    'escience2022/Remy/od550aer_AERday_NorESM2-LM_ssp245_r1i1p1f1_gn_20910101-21001231.nc'
])
sopenlist=[s3.open(ss) for ss in s3path]
aod245 = (xr.open_mfdataset(sopenlist)).sel(time = slice(str(start_year2)+"-01-01", str(end_year2)+"-12-31"))

# ssp585 for 2085-2100
s3path = list([
 'escience2022/Remy/od550aer_AERday_NorESM2-LM_ssp585_r1i1p1f1_gn_20810101-20901231.nc',
 'escience2022/Remy/od550aer_AERday_NorESM2-LM_ssp585_r1i1p1f1_gn_20910101-21001231.nc',
])

sopenlist=[s3.open(ss) for ss in s3path]
aod585 = (xr.open_mfdataset(sopenlist)).sel(time = slice(str(start_year2)+"-01-01", str(end_year2)+"-12-31"))

# ssp370
def read_370():
    s3path = list([
        'escience2022/Remy/od550aer_AERday_NorESM2-LM_ssp370_r1i1p1f1_gn_20810101-20901231.nc',
        'escience2022/Remy/od550aer_AERday_NorESM2-LM_ssp370_r1i1p1f1_gn_20910101-21001231.nc',
        'escience2022/Remy/va_day_NorESM2-LM_ssp370_r1i1p1f1_gn_20810101-20901231.nc',
        'escience2022/Remy/va_day_NorESM2-LM_ssp370_r1i1p1f1_gn_20910101-21001231.nc',
        'escience2022/Remy/hus_day_NorESM2-LM_ssp370_r1i1p1f1_gn_20810101-20901231.nc',
        'escience2022/Remy/hus_day_NorESM2-LM_ssp370_r1i1p1f1_gn_20910101-21001231.nc',
    ])

    sopenlist=[s3.open(ss) for ss in s3path]
    bucket370_8500 = (xr.open_mfdataset(sopenlist)).sel(time = slice(str(start_year2)+"-01-01", str(end_year2)+"-12-31"))
    
    return bucket370_8500

