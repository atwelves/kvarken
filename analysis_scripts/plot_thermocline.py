# script to plot salinity along section through quark

import numpy as np
import xarray as xr
import matplotlib as mpl
from matplotlib import pyplot as plt
import calendar
from calendar import monthrange
import cmocean

tem = np.zeros((166,80,400))

pathname='/media/twelves/My Passport/kvarken/may_jul'

filename='KVARKE_1d_20200501_20201031_grid_T'
ds = xr.open_dataset('{}/{}.nc'.format(pathname,filename))
sec = ds.isel(x_grid_T=250)
print(sec)
tem[:92,:,:] = sec.temp.values
xr.Dataset.close(ds)

pathname='/media/twelves/My Passport/kvarken/aug_oct'

filename='KVARK2_1d_20200801_20201031_grid_T'
ds = xr.open_dataset('{}/{}.nc'.format(pathname,filename))
sec = ds.isel(x_grid_T=250)
print(sec)
tem[92:,:,:] = sec.temp.values
xr.Dataset.close(ds)

thm = np.zeros((5,80,500))
strt=0
for m in range(5,10):
    mlen = monthrange(2020,m)
    mlen = np.squeeze(mlen[1])
    print(mlen)
    thm[m-5,:,50:450] = np.nanmean(tem[strt:strt+mlen,:,:],0)
    # read in OBC
    ds = xr.open_dataset('kvarken/kvarken_south_ts_y2020m{:02d}.nc'.format(m))
    obc = ds.temp.values
    thm[m-5,:40,0:51] = np.flip(np.squeeze(obc[40:,0:51]))
    ds = xr.open_dataset('kvarken/kvarken_north_ts_y2020m{:02d}.nc'.format(m))
    obc = ds.temp.values
    print(np.shape(np.flip(np.squeeze(obc))))
    thm[m-5,:65,449:500] = np.flip(np.squeeze(obc[15:,0:51]))
    thm[thm==0] = np.nan
    plt.figure(figsize=(40,15))
    pcol = plt.contourf(np.linspace(-50,450,500),np.linspace(0,80,80),thm[m-5,:,:],np.linspace(2,18,9),cmap='cmo.thermal')
    #plt.clim(1,18)
    cbar = plt.colorbar()
    plt.gca().invert_yaxis()
    plt.xticks([0,400],['SW','NE'],fontsize=60)
    plt.xlabel('km',fontsize=60)
    plt.yticks(np.linspace(0,80,5),fontsize=60)
    plt.ylabel('Depth (m)',fontsize=60)
    cbar.ax.tick_params(labelsize=60)
    plt.grid(axis='x',linewidth=15,linestyle='--',color='grey')
    plt.savefig('thermo_{}.png'.format(m))
    strt=mlen+strt
