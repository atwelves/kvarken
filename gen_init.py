### --- Libraries --- ### 

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4 as nc
from netCDF4 import Dataset

outfile = 'test_init_kvarken'

try: ncfile.close()
except: pass
ncfile = Dataset('{}.nc'.format(outfile), mode='w')
print(ncfile)

z = ncfile.createDimension('z',80)
y = ncfile.createDimension('y',400)
x = ncfile.createDimension('x',700)

init_temp  = ncfile.createVariable('init_temp' ,np.float32,('z','y','x'))
init_salt  = ncfile.createVariable('init_salt' ,np.float32,('z','y','x'))

for j in range(0,400):
    init_temp[:,j,:]  = 8-(j/400)*3
    init_salt[:,j,:]  = 6-(j/400)*3

print(ncfile)
ncfile.close(); print('Dataset is closed')

outfile = 'test_force_kvarken'

try: ncfile.close()
except: pass
ncfile = Dataset('{}.nc'.format(outfile), mode='w')
print(ncfile)

y = ncfile.createDimension('y',400)
x = ncfile.createDimension('x',700)
t = ncfile.createDimension('t',24)

u_wind  = ncfile.createVariable('u_wind' ,np.float32,('t','y','x'))
v_wind  = ncfile.createVariable('v_wind' ,np.float32,('t','y','x'))
t_air   = ncfile.createVariable('t_air' ,np.float32,('t','y','x'))
h_air   = ncfile.createVariable('h_air' ,np.float32,('t','y','x'))
p_air   = ncfile.createVariable('p_air' ,np.float32,('t','y','x'))
lw_dwn  = ncfile.createVariable('lw_dwn' ,np.float32,('t','y','x'))
sw_dwn  = ncfile.createVariable('sw_dwn' ,np.float32,('t','y','x'))
precip  = ncfile.createVariable('precip' ,np.float32,('t','y','x'))
snow    = ncfile.createVariable('snow' ,np.float32,('t','y','x'))
slp     = ncfile.createVariable('slp' ,np.float32,('t','y','x'))

u_wind[:,:,:]  =  0
v_wind[:,:,:]  = -5
t_air[:,:,:]   = 280
h_air[:,:,:]   = 0.003
p_air[:,:,:]   = 101000
sw_dwn[:,:,:]  = 400
lw_dwn[:,:,:]  = 200
precip[:,:,:]  = 0
snow[:,:,:]    = 0
slp[:,:,:]     = 0

print(ncfile)
ncfile.close(); print('Dataset is closed')

outfile = 'test_bdy_kvarken'

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

outfile = 'test_river_kvarken'

try: ncfile.close()
except: pass
ncfile = Dataset('{}.nc'.format(outfile), mode='w')
print(ncfile)

y = ncfile.createDimension('y',400)
x = ncfile.createDimension('x',700)
t = ncfile.createDimension('t',24)

rnf = ncfile.createVariable('rnf',np.float32,('t','y','x'))
rtem = ncfile.createVariable('rtem',np.float32,('t','y','x'))
rsal = ncfile.createVariable('rsal',np.float32,('t','y','x'))

rnf[:,:,:]  = 0
rtem[:,:,:]  = 0
rsal[:,:,:]  = 0

rnf[:,112,26]  = 0.001
rtem[:,112,26]  = 3
rsal[:,112,26]  = 0.1
