#!/usr/bin/env python
# coding: utf-8

# In[2]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import seaborn as sns
import cartopy as cy
import cartopy.crs as ccrs


# In[4]:


#b'/home/jovyan/Tjaernoe2022-group5/notebooks/Lea/Tjaernoe2022-group5/notebooks/Lea/20352049_AR_detection.nc'
path = '20352049_AR_detection.nc'
ar3550 = xr.open_dataset(path)



