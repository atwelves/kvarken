# script to trim emodnet data to required size

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4 as nc
from netCDF4 import Dataset
import scipy
from scipy.interpolate import interpn

# select part of tile 
lat_min = 62
lat_max = 65

lon_min = 20
lon_max = 22.5

infile = 'C6_2020'

ds   = xr.open_dataset('{}.nc'.format(infile))
bath = ds.elevation.values
lat  = ds.lat.values
lon  = ds.lon.values
bath[np.isnan(bath)] = 0
lat = np.tile(lat,(len(lon),1))
lat = np.transpose(lat)

lon = np.tile(lon,(len(lat),1))

bath[lat<lat_min]   = np.nan
bath[lat>lat_max]   = np.nan
bath[lon<lon_min]   = np.nan
bath[lon>lon_max] = np.nan

lon[np.isnan(bath)] = np.nan
lat[np.isnan(bath)] = np.nan

bath[-1,:] = np.nan
bath[:,-1] = np.nan
lat[-1,:] = np.nan
lat[:,-1] = np.nan
lon[-1,:] = np.nan
lon[:,-1] = np.nan

bath = bath[~np.isnan(bath).all(axis=1)]
bath = np.transpose(bath)
bath = bath[~np.isnan(bath).all(axis=1)]
bath = np.transpose(bath)
bath = -bath

lat = lat[~np.isnan(lat).all(axis=1)]
lat = np.transpose(lat)
lat = lat[~np.isnan(lat).all(axis=1)]
lat = np.transpose(lat)

lon = lon[~np.isnan(lon).all(axis=1)]
lon = np.transpose(lon)
lon = lon[~np.isnan(lon).all(axis=1)]
lon = np.transpose(lon)

# add in sampling station
#Norr_lat = 63.12882
#Norr_lon = 21.31295

#stat_lat = np.abs(lat-Norr_lat)
#stat_lon = np.abs(lon-Norr_lon)
#stat_grid = stat_lat + np.cos(np.pi*lat/180)*stat_lon
#plt.figure()
#pcol=plt.pcolormesh(stat_grid)
#plt.show()

#print("station at:{}".format(np.argmin(stat_grid),axis=0))

#bath[bath>60] = -999

# minimum depth
bath[bath<1] = 0

bath[-1,:] = np.nan
bath[:,-1] = np.nan
lat[-1,:] = np.nan
lat[:,-1] = np.nan
lon[-1,:] = np.nan
lon[:,-1] = np.nan

bath = bath[~np.isnan(bath).all(axis=1)]
bath = np.transpose(bath)
bath = bath[~np.isnan(bath).all(axis=1)]
bath = np.transpose(bath)
bath = -bath

lat = lat[~np.isnan(lat).all(axis=1)]
lat = np.transpose(lat)
lat = lat[~np.isnan(lat).all(axis=1)]
lat = np.transpose(lat)

lon = lon[~np.isnan(lon).all(axis=1)]
lon = np.transpose(lon)
lon = lon[~np.isnan(lon).all(axis=1)]
lon = np.transpose(lon)

#bath[-1,:] = np.nan
#bath[:,-1] = np.nan

#bath = bath[~np.isnan(bath).all(axis=1)]

#bath = np.transpose(bath)

#bath = bath[~np.isnan(bath).all(axis=1)]

#bath = np.transpose(bath)

print(np.shape(bath))

### at 60', remove half of longitudinal points

bath_comp = np.zeros((2879,1199))

#for i in range(0,1199):
#    itr = 2*i
#    bath_comp[:,i] = bath[:,itr]

### convert longitudes from degrees to km (from 20'E)

lon_old= (lon-lon[0,0])*np.cos(np.pi*lat/180)*111
print(lat)
plt.figure()
pcol=plt.contour(lon_old)
plt.show()
print(lon_old[0,:])
#lat_km = 

lat_km = np.arange(0,2879,1)*0.115
print(lat_km)

lon_km = np.arange(0,2399,1)*0.115

# tile long_new over latitudinal direction

old_grid = (lat_km,lon_km,'dtype=object')
print(np.shape(old_grid))
# interpolate onto new grid

new_bath = np.zeros((2879,2399))

print(np.shape(lon_old))
print(np.shape(bath))

for j in range(0,2879):
    new_bath[j,:] = np.interp(lon_km,lon_old[j,:],bath[j,:])

new_bath=-new_bath
new_bath[new_bath==0] = np.nan
plt.figure()
pcol=plt.pcolormesh(lon_km,lat_km,new_bath)
cbar = plt.colorbar(pcol)
#plt.show()

# add in sampling station
Norr_lat = 63.12882
Norr_lon = 21.31295

conv_lon = (Norr_lon-lon[0,0])*np.cos(np.pi*Norr_lat/180)*111
conv_lat = (Norr_lat-lat[0,0])*111
print('convert station lon: {}'.format(conv_lon))
print('convert station lat: {}'.format(conv_lat))
diff_lon = np.abs(lon_km-conv_lon)
diff_lat = np.abs(lat_km-conv_lat)

diff_lon_1 = np.nanmin(diff_lon)
print(diff_lon_1)
diff_lat_1 = np.nanmin(diff_lat)
print(diff_lat_1)

stat_lon = np.argmin(diff_lon)
stat_lat = np.argmin(diff_lat)

print(stat_lon)
print(stat_lat)

# also need to set adjacent cell before rotation...
diff_lon[stat_lon]=np.nan
diff_lat[stat_lat]=np.nan

diff_lon_2 = np.nanmin(diff_lon)
print(diff_lon_2)
diff_lat_2 = np.nanmin(diff_lat)
print(diff_lat_2)

if (diff_lon_2 < diff_lat_2):
    print('here')
    alt_lon = np.argmin(diff_lon_2)
    alt_lat = stat_lat
else:
    print('there')
    alt_lon = stat_lon
    alt_lat = np.argmin(diff_lat_2)

#stat_lon = np.argmin(np.abs(lon_km-conv_lon))
#stat_lat = np.argmin(np.abs(lat_km-conv_lat))

print(stat_lon)
print(stat_lat)
print(alt_lon)
print(alt_lat)
###
#TEST!
new_bath[stat_lat,stat_lon]=-100
new_bath[alt_lat,alt_lon]=-100
plt.figure()
pcol=plt.pcolormesh(new_bath)
cbar = plt.colorbar(pcol)
#plt.show()
### interpolate bathymetry onto grid with 115m spacing in both latitude and longitude directions

plt.figure()
pcol = plt.pcolormesh(new_bath,cmap='coolwarm')
plt.clim(-60,60)
cbar = plt.colorbar(pcol)
plt.savefig('comp_quark.png')

flat_bath = np.ravel(new_bath)
flat_lat  = np.ravel(lat)
flat_lon  = np.ravel(lon)

print(np.shape(flat_bath))
#ny = 390
#nx = 525
ny  = 400
nx  = 700

rotd_bath = np.zeros((ny,nx))
#y_0 = 800
#x_0 = 500

y_0  = 1600
x_0  = 0

x_pts = 2399 

rotd_y = np.arange(0,ny,1)*np.sqrt(2)*0.115
rotd_x = np.arange(0,nx,1)*np.sqrt(2)*0.115

for j in range (0,ny):
    for i in range(0,nx):
        rotd_indx      = y_0*x_pts + x_0 + j*(x_pts+1) - i*(x_pts-1)
        rotd_bath[j,i] = flat_bath[rotd_indx]
#        rotd_bath[j,i]  = rotd_indx
#        print(rotd_indx)
 #       print(rotd_bath[j,i])

#plt.figure()
#pcol = plt.pcolormesh(bath,cmap='coolwarm')
#plt.clim(-60,60)
#cbar = plt.colorbar(pcol)
#plt.show()
#plt.savefig('quark.png')

#nbor_sea = np.zeros((ny,nx))

#for j in range (1,ny-1):
#    for i in range (1,nx-1):
#        nbor_sea[j,i] = rotd_bath[j-1,i] + rotd_bath[j+1,i] + rotd_bath[j,i-1] + rotd_bath[j,i+1]

#rotd_bath[nbor_sea==0]=0

#plt.figure()
#pcol = plt.pcolormesh(nbor_sea,cmap='winter')
#plt.gca().set_aspect('equal')
#cbar = plt.colorbar(pcol)
#plt.show()

# minimum depth
rotd_bath[rotd_bath<3]=3

rotd_bath[np.isnan(rotd_bath)]=0

plt.figure(figsize=(40,40))
pcol = plt.pcolormesh(rotd_x,rotd_y,rotd_bath,cmap='winter')
plt.gca().set_aspect('equal')
cbar = plt.colorbar(pcol,location='bottom')
plt.title('Bathymetry (m)',fontsize=40)
cbar.ax.tick_params(labelsize=35)
plt.yticks(fontsize=35)
plt.xticks(fontsize=35)
plt.xlabel('km',fontsize=40)
plt.ylabel('km',fontsize=40)
#plt.show()
plt.savefig('rotated_quark.png')
rotd_bath[np.isnan(rotd_bath)] = 0
#write out this quark bathymetry to a netcdf file
outfile = 'quark_bath_show_stations'
try: ncfile.close()
except: pass
ncfile = Dataset('{}.nc'.format(outfile), mode='w')
print(ncfile)

y = ncfile.createDimension('y',ny)
x = ncfile.createDimension('x',nx)

Bathymetry  = ncfile.createVariable('Bathymetry' ,np.float32,('y','x'))

Bathymetry[:,:]  = rotd_bath[:,:]

print(ncfile)
ncfile.close(); print('Dataset is closed')


# now select only half points


