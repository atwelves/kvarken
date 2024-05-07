### --- Libraries --- ### 

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4 as nc
from netCDF4 import Dataset

# read in netcdf cmems

temp = np.zeros((6,56,774,763))
salt = np.zeros((6,56,774,763))
vvel = np.zeros((6,56,774,763))

tmp_bnd_s = np.zeros((6,80,55))
slt_bnd_s = np.zeros((6,80,55))
vel_bnd_s = np.zeros((6,80,55))

tmp_bnd_n = np.zeros((6,80,55))
slt_bnd_n = np.zeros((6,80,55))
vel_bnd_n = np.zeros((6,80,55))

m=5
ds = xr.open_dataset("~/Downloads/BAL-MYP-NEMO_PHY-MonthlyMeans-2021{:02d}.nc".format(m))
lat = ds.lat.values
lon = ds.lon.values
dpt = ds.depth.values

depth_match = np.linspace(0.5,79.5,80)
for k in range(0,80):
    depth_match[k] = np.argmin(np.abs(k+0.5-dpt))

print(lat[680])
print(lat[590])
print(lon[395])
print(lon[480])

for m in range(5,11):
    ds = xr.open_dataset("~/Downloads/BAL-MYP-NEMO_PHY-MonthlyMeans-2021{:02d}.nc".format(m))
    t_in = ds.thetao.values
    temp[m-5,:,:,:] = np.squeeze(t_in)
    s_in = ds.so.values
    salt[m-5,:,:,:] = np.squeeze(s_in)
    v_in = ds.vo.values
    vvel[m-5,:,:,:] = np.squeeze(v_in)
    print('GOT HERE')
    for i in range (395,450):
        j = 645 - (i-395)
        for k in range(0,80):
            indz = np.int(depth_match[k])
            tmp_bnd_s[m-5,k,i-395] = temp[m-5,indz,j,i]
            slt_bnd_s[m-5,k,i-395] = salt[m-5,indz,j,i]
            vel_bnd_s[m-5,k,i-395] = vvel[m-5,indz,j,i]
        tmp_bnd_s[m-5,33:,i-395] = tmp_bnd_s[m-5,32,i-395]
        slt_bnd_s[m-5,33:,i-395] = slt_bnd_s[m-5,32,i-395]
        vel_bnd_s[m-5,33:,i-395] = vel_bnd_s[m-5,32,i-395]
    for i in range (425,480):
        j = 680 - (i-425)
        for k in range(0,80):
            indz = np.int(depth_match[k])
            tmp_bnd_n[m-5,k,i-425] = temp[m-5,indz,j,i]
            slt_bnd_n[m-5,k,i-425] = salt[m-5,indz,j,i]
            vel_bnd_n[m-5,k,i-425] = vvel[m-5,indz,j,i]

temp_obc_s = np.transpose(np.nanmean(tmp_bnd_s,2))
temp_obc_n = np.transpose(np.nanmean(tmp_bnd_n,2))

slt_obc_s = np.transpose(np.nanmean(slt_bnd_s,2))
slt_obc_n = np.transpose(np.nanmean(slt_bnd_n,2))

vel_obc_s = np.transpose(np.nanmean(vel_bnd_s,2))
vel_obc_n = np.transpose(np.nanmean(vel_bnd_n,2))

for m in range(5,11):
    temp_obc_north = np.squeeze(temp_obc_n[:,m-5])
    temp_obc_north = np.tile(temp_obc_north,(400,1))
    temp_obc_north = np.transpose(temp_obc_north)
    temp_obc_north = np.tile(temp_obc_north,(1,1,1))
    temp_obc_south = np.squeeze(temp_obc_s[:,m-5])
    temp_obc_south = np.tile(temp_obc_south,(400,1))
    temp_obc_south = np.transpose(temp_obc_south)
    temp_obc_south = np.tile(temp_obc_south,(1,1,1))

    salt_obc_north = np.squeeze(slt_obc_n[:,m-5])
    salt_obc_north = np.tile(salt_obc_north,(400,1))
    salt_obc_north = np.transpose(salt_obc_north)
    salt_obc_north = np.tile(salt_obc_north,(1,1,1))
    salt_obc_south = np.squeeze(slt_obc_s[:,m-5])
    salt_obc_south = np.tile(salt_obc_south,(400,1))
    salt_obc_south = np.transpose(salt_obc_south)
    salt_obc_south = np.tile(salt_obc_south,(1,1,1))

    vvel_obc_north = np.squeeze(vel_obc_n[:,m-5])
    vvel_obc_north = np.tile(vvel_obc_north,(400,1))
    vvel_obc_north = np.transpose(vvel_obc_north)
    vvel_obc_north = np.tile(vvel_obc_north,(1,1,1))
    vvel_obc_south = np.squeeze(vel_obc_s[:,m-5])
    vvel_obc_south = np.tile(vvel_obc_south,(400,1))
    vvel_obc_south = np.transpose(vvel_obc_south)
    vvel_obc_south = np.tile(vvel_obc_south,(1,1,1))

    outfile = 'kvarken_north_ts'

    try: ncfile.close()
    except: pass
    ncfile = Dataset('{}_y2020m{:02d}.nc'.format(outfile,m), mode='w')
    print(ncfile)
    
    t = ncfile.createDimension('time_counter',1)
    z = ncfile.createDimension('z',79)
    x = ncfile.createDimension('xb',400)

    temp = ncfile.createVariable('temp',np.float32,('time_counter','xb','z'))
    salt = ncfile.createVariable('salt',np.float32,('time_counter','xb','z'))
    vvel = ncfile.createVariable('vvel',np.float32,('time_counter','xb','z'))

    temp[:,:,:]  = temp_obc_north[:,:79,:]
    salt[:,:,:]  = salt_obc_north[:,:79,:]
    vvel[:,:,:]  = vvel_obc_north[:,:79,:]

    print(ncfile)
    ncfile.close(); print('Dataset is closed')

    outfile = 'kvarken_south_ts'

    try: ncfile.close()
    except: pass
    ncfile = Dataset('{}_y2020m{:02d}.nc'.format(outfile,m), mode='w')
    print(ncfile)

    t = ncfile.createDimension('time_counter',1)
    z = ncfile.createDimension('z',79)
    x = ncfile.createDimension('xb',400)

    temp = ncfile.createVariable('temp',np.float32,('time_counter','xb','z'))
    salt = ncfile.createVariable('salt',np.float32,('time_counter','xb','z'))
    vvel = ncfile.createVariable('vvel',np.float32,('time_counter','xb','z'))

    temp[:,:,:]  = temp_obc_south[:,:79,:]
    salt[:,:,:]  = salt_obc_south[:,:79,:]
    vvel[:,:,:]  = vvel_obc_south[:,:79,:]

    print(ncfile)
    ncfile.close(); print('Dataset is closed')

### extract atmospheric boundary conditions here somehow??
# condition for min, max of lat, lon
ds = xr.open_dataset("FORCE_y2020m01d10.nc")
lw_force = ds.LongWaveRadiationFlux.values
lat_force = ds.latitude.values
lat_force = np.tile(lat_force,(1560,1))
lat_force = np.transpose(lat_force)
lat_force = np.tile(lat_force,(24,1,1))
lon_force = ds.longitude.values
lon_force = np.tile(lon_force,(820,1))
lon_force = np.tile(lon_force,(24,1,1))
print(np.shape(lat_force))
print(np.shape(lon_force))
print(np.shape(lw_force))
# check that nearest (longitudinal) neighbour is not land
# then generate list of ATM_NORDIC indices corresponding to the points that I want
lw_force[lat_force>lat[680]] = np.nan
lw_force[lat_force<lat[590]] = np.nan
lw_force[lon_force>lon[490]] = np.nan
lw_force[lon_force<lon[395]] = np.nan



plt.figure()
pcol=plt.pcolormesh(np.nanmean(lw_force,0))
plt.show()
# average over all locations and hours for each daily file, then close file
# end with daily sampled time series for each atmospheric paramater
# implement this somehow on puhti???

plt.figure()
pcol=plt.pcolormesh(temp_obc_s)
plt.gca().invert_yaxis()
cbar=plt.colorbar(pcol)
plt.show()

plt.figure()
plin = plt.plot(np.nanmean(tmp_bnd_s,1),'r')
plin = plt.plot(np.nanmean(tmp_bnd_n,1),'b')
plt.show()

plt.figure()
plin = plt.plot(np.nanmean(slt_bnd_s,1),'r')
plin = plt.plot(np.nanmean(slt_bnd_n,1),'b')
plt.show()

outfile = 'test_3d_bdy_kvarken'

try: ncfile.close()
except: pass
ncfile = Dataset('{}.nc'.format(outfile), mode='w')
print(ncfile)

y = ncfile.createDimension('y',1400)
x = ncfile.createDimension('x',1)
t = ncfile.createDimension('t',24)

ssh = ncfile.createVariable('ssh',np.float32,('t','y','x'))
uos = ncfile.createVariable('uos',np.float32,('t','y','x'))
vos = ncfile.createVariable('vos',np.float32,('t','y','x'))

ssh[:,:,:]  = 0
uos[:,:,:]  = 0
vos[:,:,:]  = -0.1

print(ncfile)
ncfile.close(); print('Dataset is closed')
