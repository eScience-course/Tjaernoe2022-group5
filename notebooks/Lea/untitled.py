import xarray as xr
xr.set_options(display_style='html')
import intake
import cftime
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from IPython import display

def readcmip6(source_id, experiment_id, table_id, variable_id, member_id):
cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(cat_url)
cat = col.search(source_id=['NorESM2-LM'], 
                 experiment_id=['ssp245'], 
                 table_id=['day'], 
                 variable_id=['pr','clt','hus','va','tas'], 
                 member_id=['r1i1p1f1'])

dset_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})
a+experiment_id = dset_dict[dataset_list[0]]
return a