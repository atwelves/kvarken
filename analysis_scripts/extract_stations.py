# script to extract stations from model output, compare to observations

import xarray as xr
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import cmocean 
import datetime
from datetime import datetime
import calendar
from calendar import monthrange

# Read in SST from SYKE dataset

line_arr = np.zeros((1238,1046))

# Read in model grid

ds = xr.open_dataset('quark_bath_show_stations.nc')
mod_lat = ds.Latitude.values
mod_lon = ds.Longitude.values
#bathy   = ds.Bathymetry.values

ds = xr.open_dataset('quark_bath_mod.nc')
bathy = ds.Bathymetry.values

import csv

old_stat = ''

stat_lat = np.zeros((1000))
stat_lon = np.zeros((1000))
stat_sst = np.zeros((1000))
stat_month = np.zeros((1000))
stat_day = np.zeros((1000))

strt = np.array([0,31,61,92,123,153])

with open('SST.csv', newline='') as csvfile:
    next(csvfile)

    sst_reader = csv.reader(csvfile, delimiter=';', quotechar='|')

    count=-1
    for row in sst_reader:
        new_stat = row[5]
        count=count+1
        if (new_stat != old_stat):
            lat = np.float(row[6])
            lon = np.float(row[7])
            obs_lat = np.zeros((400,700)) + lat
            obs_lon = np.zeros((400,700)) + lon
            diff_lat = obs_lat-mod_lat
            diff_lon = obs_lon-mod_lon
            diff_xy  = np.power(diff_lat,2) + np.power( diff_lon*np.cos(np.pi*lat/180)  ,2)
            mtch_pt  = np.unravel_index(diff_xy.argmin(), diff_xy.shape)
        else:
            mtch_pt  = mtch_pt
        if (bathy[mtch_pt] > 0):
            stat_lat[count] = mtch_pt[0]
            stat_lon[count] = mtch_pt[1]
            string = row[0]
            stat_month[count] = 10*np.int(string[5]) + np.int(string[6])
            indx = stat_month[count] - 5
            indx = np.int(indx)
            strt_day = strt[indx]
            days_in = 10*np.int(string[8])+np.int(string[9])
            days_in = np.int(days_in)
            stat_day[count]   = days_in + np.int(strt_day)
            #print(stat_day[count])
            #print(stat_day[count])
            stat_sst[count] = row[2]
            #bathy[mtch_pt] = np.nan
            #print(stat_sst[count])
        old_stat = new_stat

stat_sst[stat_sst==0] = np.nan
stat_month[stat_month==0] = np.nan
#stat_day = np.int(stat_day)
stat_day[stat_sst==0] = np.nan
#plt.figure()
#pcol = plt.pcolormesh(bathy[:,:],cmap='cmo.rain_r')
#plt.clim(0,0.1)
#plt.show()

#plt.figure()
#color_scale = stat_sst/np.nanmax(stat_sst)
#plt.scatter(stat_lon,stat_lat,c=stat_sst,s=100,cmap='cmo.dense')
#cbar=plt.colorbar()
#plt.fontsize=40
#plt.show()

plt.figure(figsize=(18,15))
cmap = plt.get_cmap('cmo.thermal', 5)
plt.scatter(stat_lat,stat_sst,c=stat_month,cmap=cmap,s=500,alpha=0.8)
cbar=plt.colorbar()
plt.xlabel('km',fontsize=40)
plt.xticks(np.linspace(0,400,5),fontsize=40)
plt.ylabel('SST ($\degree$C)',fontsize=40)
plt.yticks(np.linspace(5,20,4),fontsize=40)
plt.clim(5,10)
cbar.ax.tick_params(labelsize=0)
plt.savefig('obs_sst')

# read in model output
mod_sst = np.zeros((1000))
pathname = '/media/twelves/My Passport/kvarken/may_jul/'
filename = 'KVARKE_1d_20200501_20201031_grid_T' 
ds_may = xr.open_dataset('{}/{}.nc'.format(pathname,filename))
pathname = '/media/twelves/My Passport/kvarken/aug_oct/'
filename = 'KVARK2_1d_20200801_20201031_grid_T'
ds_aug = xr.open_dataset('{}/{}.nc'.format(pathname,filename))

for i in range (0,np.size(stat_sst)):
    if(stat_sst[i]>0 and stat_day[i]<92):
        ext = ds_may.isel(deptht=0,y_grid_T=np.int(stat_lat[i]),x_grid_T=np.int(stat_lon[i]))
        inp = ext.temp.values
        mod_sst[i] = inp[np.int(stat_day[i])-1]
        print(mod_sst[i])
    if (stat_sst[i]>0 and stat_day[i]>92 and stat_day[i]<163):
        ext = ds_aug.isel(deptht=0,y_grid_T=np.int(stat_lat[i]),x_grid_T=np.int(stat_lon[i]))
        inp = ext.temp.values
        mod_sst[i] = inp[np.int(stat_day[i])-93]
        print(mod_sst[i])

stat_sst[mod_sst==0] = np.nan

plt.figure(figsize=(18,15))
cmap = plt.get_cmap('cmo.thermal', 5)
plt.scatter(stat_sst,mod_sst,c=stat_month,cmap=cmap,s=500,alpha=0.8)
plt.plot(np.linspace(3,23,20),np.linspace(3,23,20),linewidth=5,color='k',linestyle=':')
cbar=plt.colorbar()
plt.xlabel('observed SST ($\degree$C)',fontsize=40)
plt.ylim(3,23)
plt.xlim(3,23)
plt.xticks(np.linspace(5,20,4),fontsize=40)
plt.ylabel('modelled SST ($\degree$C)',fontsize=40)
plt.yticks(np.linspace(5,20,4),fontsize=40)
plt.clim(5,10)
cbar.ax.tick_params(labelsize=0)
plt.grid()
plt.savefig('comp_sst')

plt.figure(figsize=(18,15))
plt.scatter(stat_sst,mod_sst,c=stat_lat*0.115*np.sqrt(2),cmap='cmo.ice',s=500,alpha=0.8)
plt.plot(np.linspace(3,23,20),np.linspace(3,23,20),linewidth=5,color='k',linestyle=':')
cbar=plt.colorbar()
plt.xlabel('observed SST ($\degree$C)',fontsize=40)
plt.ylim(3,23)
plt.xlim(3,23)
plt.xticks(np.linspace(5,20,4),fontsize=40)
plt.ylabel('modelled SST ($\degree$C)',fontsize=40)
plt.yticks(np.linspace(5,20,4),fontsize=40)
#plt.clim(0,400)
cbar.ax.tick_params(labelsize=40)
plt.grid()
plt.savefig('comp_sst_lat')

