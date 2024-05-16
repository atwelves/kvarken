# script to plot heat fluxes from kvarken model

import xarray as xr 
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

sw_t = np.zeros((166))
lw_t = np.zeros((166))
sen_t = np.zeros((166))
lat_t = np.zeros((166))

pathname = '/media/twelves/My Passport/kvarken/may_jul/'

# read in vvel
filename = 'KVARKE_1d_20200501_20201031_surf_T'
ds = xr.open_dataset('{}/{}.nc'.format(pathname,filename))
sw = ds.sw_down.values
lw = ds.lw_down.values
sen = ds.sen_down.values
lat = ds.lat_down.values

sw[sw==0]=np.nan
lw[lw==0]=np.nan
lat[lat==0]=np.nan
sen[sen==0]=np.nan

# average
sw_t[:92] = np.nanmean(np.nanmean(sw,2),1)
lw_t[:92] = np.nanmean(np.nanmean(lw,2),1)
sen_t[:92] = np.nanmean(np.nanmean(sen,2),1)
lat_t[:92] = np.nanmean(np.nanmean(lat,2),1)

pathname = '/media/twelves/My Passport/kvarken/aug_oct/'

# read in vvel
filename = 'KVARK2_1d_20200801_20201031_surf_T'
ds = xr.open_dataset('{}/{}.nc'.format(pathname,filename))
sw = ds.sw_down.values
lw = ds.lw_down.values
sen = ds.sen_down.values
lat = ds.lat_down.values

sw[sw==0]=np.nan
lw[lw==0]=np.nan
lat[lat==0]=np.nan
sen[sen==0]=np.nan

# average
sw_t[92:] = np.nanmean(np.nanmean(sw,2),1)
lw_t[92:] = np.nanmean(np.nanmean(lw,2),1)
sen_t[92:] = np.nanmean(np.nanmean(sen,2),1)
lat_t[92:] = np.nanmean(np.nanmean(lat,2),1)

plt.figure(figsize=(40,20))
plt.plot(sw_t,label='shortwave',linewidth=10)
plt.plot(lw_t,label='longwave',linewidth=10)
plt.plot(sen_t,label='sensible',linewidth=10)
plt.plot(lat_t,label='latent',linewidth=10)
fig=plt.fill_between(np.linspace(0,166,166),sw_t+lw_t+sen_t+lat_t,color=(255/255,237/255,111/255),alpha=0.7)
plt.legend(fontsize=60)
plt.xticks([0,31,61,92,123,153,184],fontsize=0)
plt.yticks(fontsize=60)
plt.ylabel('Surface heat flux (W m$^{-2}$)',fontsize=60)
plt.grid()
plt.savefig('surf_heat.png')
