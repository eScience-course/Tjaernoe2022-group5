import xarray as xr
import s3fs
xr.set_options(display_style='html')
import intake
import cftime
import numpy as np
from netCDF4 import Dataset
from IPython import display

def read_pangeo(start_year, end_year, experimentid):
    cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    col = intake.open_esm_datastore(cat_url)
    cat = col.search(source_id=['NorESM2-LM'], experiment_id=[experimentid], 
                     table_id=['day'], variable_id=['pr','clt','hus','va','tas'], member_id=['r1i1p1f1'])
    dset_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})
    dataset_list = list(dset_dict.keys())
    start_year_ = start_year
    end_year_ = end_year
    df = dset_dict[dataset_list[0]]
    df_sliced = df.sel(time = slice(str(start_year)+"-01-01", str(end_year)+"-12-31"))
    return df_sliced

def read_to_detect(start_year, end_year, experimentid):
    cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    col = intake.open_esm_datastore(cat_url)
    cat = col.search(source_id=['NorESM2-LM'], experiment_id=[experimentid], 
                     table_id=['day'], variable_id=['hus','va'], member_id=['r1i1p1f1'])
    dset_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})
    dataset_list = list(dset_dict.keys())
    start_year_ = start_year
    end_year_ = end_year
    df = dset_dict[dataset_list[0]]
    df_sliced = df.sel(time = slice(str(start_year)+"-01-01", str(end_year)+"-12-31"))
    return df_sliced