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


# In[5]:


ar3550


# In[16]:


#sum of occurence of atmospheric rivers over the whole year
#ar3550['ivt'].sum('time').plot()


# In[26]:


sm = ar3550['ivt'].sum('time')
smarc = sm.sel(lat=slice(50,90))
smant = sm.sel(lat=slice(-90,-50))


# In[18]:


sm


# In[31]:


f,ax = plt.subplots(dpi=100, figsize =(10,8),
                    subplot_kw={'projection':ccrs.Orthographic(central_latitude=90.0)})
smarc.plot.pcolormesh(
    cmap = plt.get_cmap('Blues'),ax=ax,
    vmin=0, vmax=70,
    cbar_kwargs={
        'label':'AR occurence Frequency', 
        'orientation':'horizontal',
        
    },
    transform=ccrs.PlateCarree(), 
    x='lon',y='lat',
    levels = 8
)
ax.set_title('Sum of occurence 2035-2050')
ax.coastlines()

gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False

ax.add_feature(cy.feature.BORDERS);


# In[41]:





# In[32]:


f,ax = plt.subplots(dpi=100, figsize =(10,8),
                    subplot_kw={'projection':ccrs.Orthographic(central_latitude=-90.0)})

smant.plot.pcolormesh(
    cmap = plt.get_cmap('Blues'),ax=ax,
    vmin=0, vmax=56,
    cbar_kwargs={
        'label':'AR occurence frequency', 
        'orientation':'horizontal',
        
    },
    transform=ccrs.PlateCarree(), 
    x='lon',y='lat',
    levels = 8
)
ax.set_title('Sum of occurences 2035-2050')
ax.coastlines()

gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False

ax.add_feature(cy.feature.BORDERS);


# In[59]:





# In[ ]:




