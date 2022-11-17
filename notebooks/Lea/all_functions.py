import xarray as xr
import s3fs
xr.set_options(display_style='html')
import intake
import cftime
import numpy as np
from netCDF4 import Dataset
from IPython import display
import matplotlib.pyplot as plt
import netCDF4
import pandas as pd
import scipy.stats as st

'''


Here you can find 

1. all functions used in this project
    import data from Pangeo and the bucket
    import aod data 
    bias correction function

2. data treatment
    bias correction


'''


''' 1. Functions '''


### import from pangeo ###
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
    df = df.chunk(20)
    df_sliced = df.sel(time = slice(str(start_year)+"-01-01", str(end_year)+"-12-31"))
    return df_sliced

### AOD bias correction function ###

def bias_correction(histo,fut,scn):

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


### import from "the bucket" ###


def read_aod(start_year, end_year, experimentid):
    s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD", 
                           secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0", 
                           client_kwargs=dict(endpoint_url="https://rgw.met.no"))

    if experimentid == 'historical': 

    s3path = list([
     'escience2022/Remy/od550aer_AERday_NorESM2-LM_historical_r1i1p1f1_gn_20000101-20091231.nc',
     'escience2022/Remy/od550aer_AERday_NorESM2-LM_historical_r1i1p1f1_gn_20100101-20141231.nc',
    ])
    sopenlist=[s3.open(ss) for ss in s3path]
    aod = (xr.open_mfdataset(sopenlist)).sel(time = slice(str(start_year)+"-01-01", str(end_year)+"-12-31"))

    elif experimentid == 'ssp245':
        s3path = list([
            'escience2022/Remy/od550aer_AERday_NorESM2-LM_ssp245_r1i1p1f1_gn_20810101-20901231.nc',
            'escience2022/Remy/od550aer_AERday_NorESM2-LM_ssp245_r1i1p1f1_gn_20910101-21001231.nc'
        ])
        sopenlist=[s3.open(ss) for ss in s3path]
        aod = (xr.open_mfdataset(sopenlist)).sel(time = slice(str(start_year)+"-01-01", str(end_year)+"-12-31"))
            
    elif experimentid == 'ssp585':
        s3path = list([
             'escience2022/Remy/od550aer_AERday_NorESM2-LM_ssp585_r1i1p1f1_gn_20810101-20901231.nc',
             'escience2022/Remy/od550aer_AERday_NorESM2-LM_ssp585_r1i1p1f1_gn_20910101-21001231.nc',
        ])

        sopenlist=[s3.open(ss) for ss in s3path]
        aod = (xr.open_mfdataset(sopenlist)).sel(time = slice(str(start_year)+"-01-01", str(end_year)+"-12-31"))
        
    else:
        print('wrong experiment ID. for ssp370 use function read_370()')

    return aod
    
    
def read_370(start_year, end_year):
    s3path = list([
        'escience2022/Remy/od550aer_AERday_NorESM2-LM_ssp370_r1i1p1f1_gn_20810101-20901231.nc',
        'escience2022/Remy/od550aer_AERday_NorESM2-LM_ssp370_r1i1p1f1_gn_20910101-21001231.nc',
        'escience2022/Remy/va_day_NorESM2-LM_ssp370_r1i1p1f1_gn_20810101-20901231.nc',
        'escience2022/Remy/va_day_NorESM2-LM_ssp370_r1i1p1f1_gn_20910101-21001231.nc',
        'escience2022/Remy/hus_day_NorESM2-LM_ssp370_r1i1p1f1_gn_20810101-20901231.nc',
        'escience2022/Remy/hus_day_NorESM2-LM_ssp370_r1i1p1f1_gn_20910101-21001231.nc',
    ])

    sopenlist=[s3.open(ss) for ss in s3path]
    bucket370_8500 = xr.open_mfdataset(sopenlist)).sel(time = slice(str(start_year)+"-01-01", str(end_year)+"-12-31")
    
     return bucket370_8500
                                                       
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
    dp_ = np.append(np.diff(pres_),np.nan)
    dp_[np.isnan(dp_)]=0.0
    dp_=np.repeat(dp_[np.newaxis,:],q.shape[0],axis=0)
    dp_=np.repeat(dp_[:,:,np.newaxis],q.shape[2],axis=2)
    dp_=np.repeat(dp_[:,:,:,np.newaxis],q.shape[3],axis=3)
    p_ = pres_
    q_ = q
    v_ = v
    print(q)
    print(v)
    print(p_)
    iv_ = -1/g*np.sum(q_*v_*dp_,axis=1)
    return iv_

                                                       
def compute_threshold(start_year, end_year, experimentid):
    dset = read_to_detect(start_year,end_year,experimentid)                                                       
    vas = dset.va
    hus = dset.hus
    plev = dset.plev
    lat_ = hus.lat
    lon_ = hus.lon                                                   
    
    ivt = compute_ivt(hus,vas,plev)
    ivt_ns = ivt.copy()
    ivt_ns = xr.where(ivt_ns.lat<0,-ivt_ns,ivt_ns,True) # minus for southern hemisphere (positive toward the pole)
    ivt_ns_pos = xr.where(ivt_ns<0,ivt_ns*0,ivt_ns,True) # negative values = not poleward

    q93 = ivt_ns_pos.chunk(dict(time=-1)).quantile(0.94,'time')   
    q93.to_netcdf('q93_2000.nc')
    return print('finished computing; check if document q94_2000.nc was safed')   
                                                       
                                                       
                                                       
                                                       
'''Data treatment'''

### bias correction ###

aodh_ = read_aod(2000, 2014, 'historical')
aod245_ = read_aod(2085, 2099, 'ssp245')
aod585_ = read_aod(2085, 2099, 'ssp585')

histo = aodh_['od550aer'].values
ssp245 = aod245_['od550aer'].values
ssp585 = aod585_['od550aer'].values
b245 = bias_correction(histo,ssp245, '245')
b585 = bias_correction(histo,ssp585, '585')

aodh['od550aer'] = (['time','lat','lon'], b245[0])
aod245['od550aer'] = (['time','lat','lon'], b245[1])
aod585['od550aer'] = (['time','lat','lon'], b585[1])
                                                       
### merge data sets with aod data Set###
cmh = read_pangeo(2000, 2014, 'historical')
dh = cmh.merge(aodh.drop(['lat_bnds', 'time_bnds', 'lon_bnds']))

cm245 = read_pangeo(2085, 2099, 'ssp245')
d245= cm245.merge(aod245.drop(['lat_bnds', 'time_bnds', 'lon_bnds']))

cm585 = read_pangeo(2085, 2099, 'ssp585')
d585 = cm585.merge(aod585.drop(('lat_bnds', 'time_bnds', 'lon_bnds')))

# data for ssp370 was split in two different databases so here is some extra treatment                                                       
cm370 = read_pangeo(2085, 2099, 'ssp370')
cm370b  = read_370(2085, 2099)
ssp370 = cm370b['od550aer'].values
b370 = bias_correction(histo,ssp370, '370') 
cm370b['od550aer'] = (['time','lat','lon'], b370[1])
d370 = cm370.merge(cm370b.drop(('lat_bnds', 'time_bnds', 'lon_bnds')))

# slice to poles and slice only until pressure levels where AR can be detected
n245 = d245.sel(lat = slice(60,90),plev=slice(100000, 25000))
s245 =d245.sel(lat = slice(-90,-60),plev=slice(100000, 25000))

n370 = d370.sel(lat = slice(60,90),plev=slice(100000, 25000))
s370 =d370.sel(lat = slice(-90,-60),plev=slice(100000, 25000))

n585 = d585.sel(lat = slice(60,90),plev=slice(100000, 25000))
s585 =d585.sel(lat = slice(-90,-60),plev=slice(100000, 25000))

nh = dh.sel(lat = slice(60,90), plev=slice(100000, 25000))
sh =dh.sel(lat = slice(-90,-60), plev=slice(100000, 25000))


## mask data for being inside or outside of a atmospheric river

                                                       
#treat variables for plotting
#integrate needed humidity
int_nh245 =-1*masked_n245['hus'].integrate('plev')
int_sh245 =-1*masked_s245['hus'].integrate('plev')
int_nh370 =-1*masked_n370['hus'].integrate('plev')
int_sh370 =-1*masked_s370['hus'].integrate('plev')
int_nh585 =-1*masked_n585['hus'].integrate('plev')
int_sh585 =-1*masked_s585['hus'].integrate('plev')
int_nhh =-1*masked_nh['hus'].integrate('plev')
int_shh =-1*masked_sh['hus'].integrate('plev')

# flatten and remove na
    # humidity                                                      
int_nh245 = int_nh245.values.flatten() # for plotting with matplotlib: flatten db to array 
int_sh245 = int_sh245.values.flatten()
int_nh370 = int_nh370.values.flatten()
int_sh370 = int_sh370.values.flatten()
int_nh585 = int_nh585.values.flatten()
int_sh585 = int_sh585.values.flatten()
int_nhh = int_nhh.values.flatten()
int_shh = int_shh.values.flatten()
int_nh245 = int_nh245[~np.isnan(int_nh245)] # remove na from dataset to be able to weight distribution
int_sh245 = int_sh245[~np.isnan(int_sh245)]
int_nh370 = int_nh370[~np.isnan(int_nh370)]
int_sh370 = int_sh370[~np.isnan(int_sh370)]
int_nh585 = int_nh585[~np.isnan(int_nh585)]
int_sh585 = int_sh585[~np.isnan(int_sh585)]
int_nhh = int_nhh[~np.isnan(int_nhh)]
int_shh = int_shh[~np.isnan(int_shh)] 
int_hum = pd.DataFrame(data=[int_nh245,int_sh245,int_nh370,int_sh370,int_nh585,int_sh585,int_nhh,int_shh]).T 
int_hum.columns=['int_nh245','int_sh245','int_nh370','int_sh370','int_nh585','int_sh585','int_nhh','int_shh']                                                       
    # AOD
na245 = masked_n245['od550aer'].values.flatten()  # for plotting with matplotlib: flatten db to array and substract average
sa245 = masked_s245['od550aer'].values.flatten()
na370 = masked_n370['od550aer'].values.flatten()
sa370 = masked_s370['od550aer'].values.flatten()
na585 = masked_n585['od550aer'].values.flatten()
sa585 = masked_s585['od550aer'].values.flatten()
nah = masked_nh['od550aer'].values.flatten()
sah = masked_sh['od550aer'].values.flatten()
na245 = na245[~np.isnan(na245)] # remove na from dataset to be able to weight distribution
sa245 = sa245[~np.isnan(sa245)]
na370 = na370[~np.isnan(na370)]
sa370 = sa370[~np.isnan(sa370)]
na585 = na585[~np.isnan(na585)]
sa585 = sa585[~np.isnan(sa585)]
nah = nah[~np.isnan(nah)]
sah = sah[~np.isnan(sah)]
aod = pd.DataFrame(data=[na245,sa245,na370,sa370,na585,sa585,nah,sah]).T 
aod.columns = ['na245','sa245','na370','sa370','na585','sa585','nah','sah']   
                                                      
    # cloud cover
nc245 = masked_n245['clt'].values.flatten()  # for plotting with matplotlib: flatten db to array and substract average
sc245 = masked_s245['clt'].values.flatten()
nc370 = masked_n370['clt'].values.flatten()
sc370 = masked_s370['clt'].values.flatten()
nc585 = masked_n585['clt'].values.flatten()
sc585 = masked_s585['clt'].values.flatten()
nch = masked_nh['clt'].values.flatten()
sch = masked_sh['clt'].values.flatten()
nc245 = na245[~np.isnan(nc245)] # remove na from dataset to be able to weight distribution
sc245 = sa245[~np.isnan(sc245)]
nc370 = na370[~np.isnan(nc370)]
sc370 = sa370[~np.isnan(sc370)]
nc585 = na585[~np.isnan(nc585)]
sc585 = sa585[~np.isnan(sc585)]
nch = nah[~np.isnan(nch)]
sch = sah[~np.isnan(sch)]
cloud = pd.DataFrame(data=[nc245,sc245,nc370,sc370,nc585,sc585,nch,sch]).T 
cloud.columns = ['nc245','sc245','nc370','sc370','nc585','sc585','nch','sch']   
                                                                                                              
    # precipitation                                                      
np245 = masked_n245['pr'].values.flatten()  # for plotting with matplotlib: flatten db to array and substract average
sp245 = masked_s245['pr'].values.flatten()
np370 = masked_n370['pr'].values.flatten()
sp370 = masked_s370['pr'].values.flatten()
np585 = masked_n585['pr'].values.flatten()
sp585 = masked_s585['pr'].values.flatten()
nph = masked_nh['pr'].values.flatten()
sph = masked_sh['pr'].values.flatten()
np245 = np245[~np.isnan(np245)] # remove na from dataset to be able to weight distribution
sp245 = sp245[~np.isnan(sp245)]
np370 = np370[~np.isnan(np370)]
sp370 = sp370[~np.isnan(sp370)]
np585 = np585[~np.isnan(np585)]
sp585 = sp585[~np.isnan(sp585)]
nph = nph[~np.isnan(nph)]
sph = sph[~np.isnan(sph)]
np245 =np245[np245 >0.0000024099] # exclude weird small values
sp245=sp245[sp245>0.0000024099]
np370=np370[np370>0.0000024099]
sp370=sp370[sp370>0.0000024099]
np585=np585[np585>0.0000024099]
sp585=sp585[sp585>0.0000024099]
nph =nph[nph >0.0000024099]
sph=sph[sph>0.0000024099]
np245= np245*60*60*24 # from precipitation "per second" to "per day"
sp245= sp245*60*60*24
np370= np370*60*60*24
sp370= sp370*60*60*24
np585= np585*60*60*24
sp585= sp585 *60*60*24
nph= nph*60*60*24
sph= sph*60*60*24
precip = pd.DataFrame(data=[np245,sp245,np370,sp370,np585,sp585,nph,sph]).T 
precip.columns = ['np245','sp245','np370','sp370','np585','sp585','nph','sph']                                                          
                                                       
    # surface temperature
#create average temperature for research area during modelling period to calculate anomaly
avtn245 = n245['tas'].mean(['time','lat','lon'])
avts245 = s245['tas'].mean(['time','lat','lon'])
avtn370 = n370['tas'].mean(['time','lat','lon'])
avts370 = s370['tas'].mean(['time','lat','lon'])
avtn585 = n585['tas'].mean(['time','lat','lon'])
avts585 = s585['tas'].mean(['time','lat','lon'])
avtnh =nh['tas'].mean(['time','lat','lon'])
avtsh = sh['tas'].mean(['time','lat','lon'])
nt245 = (n245['tas']-avtn245).values.flatten() # for plotting with matplotlib: flatten db to array and substract average
st245 = (s245['tas']-avts245).values.flatten()
nt370 = (n370['tas']-avtn370).values.flatten()
st370 = (s370['tas']-avts370).values.flatten()
nt585 = (n585['tas']-avtn585).values.flatten()
st585 = (s585['tas']-avts585).values.flatten()
nth = (nh['tas']-avtnh).values.flatten()
sth = (sh['tas']-avtsh).values.flatten()
nt245 = nt245[~np.isnan(nt245)] # remove na from dataset to be able to weight distribution
st245 = st245[~np.isnan(st245)]
nt370 = nt370[~np.isnan(nt370)]
st370 = st370[~np.isnan(st370)]
nt585 = nt585[~np.isnan(nt585)]
st585 = st585[~np.isnan(st585)]
nth = nth[~np.isnan(nth)]
tph = sth[~np.isnan(sth)]
temp = pd.DataFrame(data=[nt245,st245,nt370,st370,nt585,st585,nth,sth]).T 
temp.columns = ['nt245','st245','nt370','st370','nt585','st585','nth','sth']   
                                                       
# size for wilcoxon test arctic
   # humidity
nh370sized = int_nh370[np.random.randint(0, len(int_nh370), 10000)]
nh245sized = int_nh245[np.random.randint(0, len(int_nh245), 10000)]
nh585sized = int_nh585[np.random.randint(0, len(int_nh585), 10000)]
nhhsized =int_nhh[np.random.randint(0, len(int_nhh), 10000)]
   # AOD 
na370sized = na370[np.random.randint(0, len(na370), 10000)]
na245sized = na245[np.random.randint(0, len(na245), 10000)]
na585sized = na585[np.random.randint(0, len(na585), 10000)]
nahsized = nah[np.random.randint(0, len(nah), 10000)]   
   # precipitation 
np370sized = np370[np.random.randint(0, len(np370), 10000)]
np245sized = np245[np.random.randint(0, len(np245), 10000)]
np585sized = np585[np.random.randint(0, len(np585), 10000)]
nphsized = nph[np.random.randint(0, len(nph), 10000)]
     # temperature 
nt370sized = nt370[np.random.randint(0, len(nt370), 10000)]
nt245sized = nt245[np.random.randint(0, len(nt245), 10000)]
nt585sized = nt585[np.random.randint(0, len(nt585), 10000)]
nthsized = nth[np.random.randint(0, len(nth), 10000)]                                                     
                                                       
# size for wilcoxon test antarctic
   # humidity
sh370sized = int_sh370[np.random.randint(0, len(int_sh370), 10000)]
sh245sized = int_sh245[np.random.randint(0, len(int_sh245), 10000)]
sh585sized = int_sh585[np.random.randint(0, len(int_sh585), 10000)]
shhsized =int_shh[np.random.randint(0, len(int_shh), 10000)]
   # AOD 
sa370sized = sa370[np.random.randint(0, len(sa370), 10000)]
sa245sized = sa245[np.random.randint(0, len(sa245), 10000)]
sa585sized = sa585[np.random.randint(0, len(sa585), 10000)]
sahsized = sah[np.random.randint(0, len(sah), 10000)]   
   # precipitation 
sp370sized = sp370[np.random.randint(0, len(sp370), 10000)]
sp245sized = sp245[np.random.randint(0, len(sp245), 10000)]
sp585sized = sp585[np.random.randint(0, len(sp585), 10000)]
sphsized = sph[np.random.randint(0, len(sph), 10000)]
     # temperature 
st370sized = st370[np.random.randint(0, len(st370), 10000)]
st245sized = st245[np.random.randint(0, len(st245), 10000)]
st585sized = st585[np.random.randint(0, len(st585), 10000)]
sthsized = sth[np.random.randint(0, len(sth), 10000)]                                                        
                                                       
                                                       
                                                       
                                                       
                                                       
                                                       
                                                       
                                                       
                                                       