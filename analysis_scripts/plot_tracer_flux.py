# script to plot quantity of river tracer entering and leaving domain

import xarray as xr
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

knfl = np.zeros((184,80,700))
unfl = np.zeros((184,80,700))
ksfl = np.zeros((184,80,700))
usfl = np.zeros((184,80,700))

secday = 86400

pathname = '/media/twelves/My Passport/kvarken/may_jul/'

# read in vvel
filename = 'KVARKE_1d_20200501_20201031_grid_T'
ds = xr.open_dataset('{}/{}.nc'.format(pathname,filename))
north = ds.isel(y_grid_V=390)
v_north = north.vo.values

# area of northern boundary
a_north = np.size(v_north[np.abs(v_north>0)])*115*np.sqrt(2)/92
print(a_north)

# set to nan if < 0
v_north[v_north<=0]=np.nan

# read in umeå
filename = 'KVARKE_1d_20200501_20201031_tracer'
ds = xr.open_dataset('{}/{}.nc'.format(pathname,filename))
north = ds.isel(y=390)
u_north = north.Umeå.values
# correct units
u_north = u_north/(115*np.sqrt(2))

# read in kyrö
filename = 'KVARKE_1d_20200501_20201031_tracer'
ds = xr.open_dataset('{}/{}.nc'.format(pathname,filename))
north = ds.isel(y=390)
k_north = north.Kyro.values
# correct units
k_north = k_north/(115*np.sqrt(2))

# multiply to get northern fluxes

k_north_flux = np.multiply(k_north,v_north)
print(np.nansum(k_north_flux))
u_north_flux = np.multiply(u_north,v_north)
print(np.nansum(u_north_flux))

knfl[:92,:,:] = k_north_flux
unfl[:92,:,:] = u_north_flux

pathname = '/media/twelves/My Passport/kvarken/aug_oct/'

# read in vvel
filename = 'KVARK2_1d_20200801_20201031_grid_T'
ds = xr.open_dataset('{}/{}.nc'.format(pathname,filename))
north = ds.isel(y_grid_V=390)
v_north = north.vo.values

# area of northern boundary
a_north = np.size(v_north[np.abs(v_north>0)])*115*np.sqrt(2)/92
print(a_north)

# set to nan if < 0
v_north[v_north<=0]=np.nan

# read in umeå
filename = 'KVARK2_1d_20200801_20201031_tracer'
ds = xr.open_dataset('{}/{}.nc'.format(pathname,filename))
north = ds.isel(y=390)
u_north = north.Umeå.values
# correct units
u_north = u_north/(115*np.sqrt(2))

# read in kyrö
filename = 'KVARK2_1d_20200801_20201031_tracer'
ds = xr.open_dataset('{}/{}.nc'.format(pathname,filename))
north = ds.isel(y=390)
k_north = north.Kyro.values
# correct units
k_north = k_north/(115*np.sqrt(2))

# multiply to get northern fluxes

k_north_flux = np.multiply(k_north,v_north)
print(np.nansum(k_north_flux))
u_north_flux = np.multiply(u_north,v_north)
print(np.nansum(u_north_flux))

knfl[92:166,:,:] = k_north_flux
unfl[92:166,:,:] = u_north_flux

plt.figure(figsize=(40,10))
plin = plt.plot(1000*np.nanmean(np.nanmean(unfl,2),1)*a_north,'k',linewidth=5,label='Umeå')
plin = plt.plot(1000*np.nanmean(np.nanmean(knfl,2),1)*a_north,'r',linewidth=5,label='Kyrö')
plin = plt.plot(np.linspace(0,165,165),np.tile(115*np.sqrt(2),165),'c:',linewidth=10,label='_nolegend')
plt.legend(fontsize=60)
plt.grid()
plt.ylabel('Tracer flux (g s⁻¹)',fontsize=60)
plt.title('North-Eastern boundary',fontsize=80)
plt.xticks([0,31,61,92,123,153],fontsize=0)
plt.xlim(0,165)
plt.yticks(fontsize = 60)

plt.savefig('ne_tracer_fluxes.png')

pathname = '/media/twelves/My Passport/kvarken/may_jul/'

# read in vvel
filename = 'KVARKE_1d_20200501_20201031_grid_T'
ds = xr.open_dataset('{}/{}.nc'.format(pathname,filename))
south = ds.isel(y_grid_V=10)
v_south = south.vo.values

# area of northern boundary
a_south = np.size(v_south[np.abs(v_south>0)])*115*np.sqrt(2)/92
print(a_south)

# set to nan if < 0
v_south[v_south>=0]=np.nan

# read in umeå
filename = 'KVARKE_1d_20200501_20201031_tracer'
ds = xr.open_dataset('{}/{}.nc'.format(pathname,filename))
south = ds.isel(y=10)
u_south = south.Umeå.values
u_south = u_south/(115*np.sqrt(2))

# read in kyrö
filename = 'KVARKE_1d_20200501_20201031_tracer'
ds = xr.open_dataset('{}/{}.nc'.format(pathname,filename))
south = ds.isel(y=10)
k_south = south.Kyro.values
k_south = k_south/(115*np.sqrt(2))

# multiply to get northern fluxes

k_south_flux = np.multiply(k_south,v_south)
print(np.nansum(k_south_flux))
u_south_flux = np.multiply(u_south,v_south)
print(np.nansum(u_south_flux))

ksfl[:92,:,:] = k_south_flux
usfl[:92,:,:] = u_south_flux

pathname = '/media/twelves/My Passport/kvarken/aug_oct/'

# read in vvel
filename = 'KVARK2_1d_20200801_20201031_grid_T'
ds = xr.open_dataset('{}/{}.nc'.format(pathname,filename))
south = ds.isel(y_grid_V=10)
v_south = south.vo.values

# area of northern boundary
a_south = np.size(v_south[np.abs(v_south>0)])*115*np.sqrt(2)/92
print(a_south)

# set to nan if < 0
v_south[v_south>=0]=np.nan

# read in umeå
filename = 'KVARK2_1d_20200801_20201031_tracer'
ds = xr.open_dataset('{}/{}.nc'.format(pathname,filename))
south = ds.isel(y=10)
u_south = south.Umeå.values
u_south = u_south/(115*np.sqrt(2))

# read in kyrö
filename = 'KVARK2_1d_20200801_20201031_tracer'
ds = xr.open_dataset('{}/{}.nc'.format(pathname,filename))
south = ds.isel(y=10)
k_south = south.Kyro.values
k_south = k_south/(115*np.sqrt(2))

# multiply to get northern fluxes

k_south_flux = np.multiply(k_south,v_south)
print(np.nansum(k_south_flux))
u_south_flux = np.multiply(u_south,v_south)
print(np.nansum(u_south_flux))

ksfl[92:166,:,:] = k_south_flux
usfl[92:166,:,:] = u_south_flux

plt.figure(figsize=(40,10))
plin = plt.plot(-1000*np.nanmean(np.nanmean(usfl,2),1)*a_south,'k',linewidth=5,label='Umeå')
plin = plt.plot(-1000*np.nanmean(np.nanmean(ksfl,2),1)*a_south,'r',linewidth=5,label='Kyrö')
plin = plt.plot(np.linspace(0,165,165),np.tile(115*np.sqrt(2),165),'c:',linewidth=10,label='_nolegend')
plt.xticks([0,31,61,92,123,153],fontsize=0)
plt.legend(fontsize=60)
plt.xlim(0,165)
plt.grid()
plt.ylabel('Tracer flux (g s⁻¹)',fontsize=60)
plt.title('South-Western boundary',fontsize=80)
plt.yticks(fontsize = 60)

plt.savefig('sw_tracer_fluxes.png')

