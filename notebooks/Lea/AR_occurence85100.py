#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import seaborn as sns
import cartopy as cy
import cartopy.crs as ccrs


# In[ ]:


#b'/home/jovyan/Tjaernoe2022-group5/notebooks/Lea/Tjaernoe2022-group5/notebooks/Lea/20352049_AR_detection.nc'
path = '20852100_AR_detection.nc'
ar8500 = xr.open_dataset(path)


# In[ ]:


#sum of occurence of atmospheric rivers over the whole year
ar8500['ivt'].sum('time').plot()


# In[ ]:


sm = ar8500['ivt'].sum('time')
smarc = sm.sel(lat=slice(50,90))
smant = sm.sel(lat=slice(-90,-50))

