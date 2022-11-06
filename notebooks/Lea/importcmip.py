#!/usr/bin/env python
# coding: utf-8

# In[3]:


import xarray as xr
import s3fs
xr.set_options(display_style='html')
import intake
import cftime
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy
import matplotlib.path as mpath
from functions import compute_ivt,to_nc
from matplotlib import rc,animation
from matplotlib.animation import FuncAnimation
from IPython import display


# In[4]:


cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(cat_url)


# In[5]:


cat = col.search(source_id=['NorESM2-LM'], experiment_id=['ssp245'], table_id=['day'], variable_id=['pr','clt','hus','va','tas'], member_id=['r1i1p1f1'])
cat.df


# In[6]:


dset_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})


# In[7]:


dataset_list = list(dset_dict.keys())


# In[13]:


start_year = 2035
end_year = 2050
cm245 = dset_dict[dataset_list[0]]
#dset = dset.sel(member_id='r1i1p1f1',time=slice("2000-01-01", "2014-12-31"))
cm245_3550 = cm245.sel(time = slice(str(start_year)+"-01-01", str(end_year)+"-12-31"))


# In[15]:


cm245_3550


# In[16]:


# import AOD
s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD", 
                       secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0",
                       client_kwargs=dict(endpoint_url="https://rgw.met.no"))
#s3.ls('escience2022/Remy/')


# In[17]:


# import AOD ssp245 for 2035-2050
s3path = list([
 'escience2022/Remy/od550aer_AERday_NorESM2-LM_ssp245_r1i1p1f1_gn_20310101-20401231.nc',
 'escience2022/Remy/od550aer_AERday_NorESM2-LM_ssp245_r1i1p1f1_gn_20410101-20501231.nc'
])

sopenlist=[s3.open(ss) for ss in s3path]

aod245_3550 = (xr.open_mfdataset(sopenlist)).sel(time = slice(str(start_year)+"-01-01", str(end_year)+"-12-31"))


# In[ ]:





# In[ ]:





# In[ ]:




