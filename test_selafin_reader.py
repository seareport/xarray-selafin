import xarray as xr
import matplotlib.pyplot as plt

file_3d = 'data/r3d_tidal_flats.slf'

# read test
ds = xr.open_dataset(file_3d, engine='selafin')

# print 
print(ds)

# plot
fig, ax = plt.subplots(1, 1, figsize=(20, 1))
ds = ds.isel(time=slice(0,3))
ax.scatter(ds.x, ds.y)
ax.triplot(ds.x, ds.y, ds.ikle[:,0:3])
plt.tight_layout()
plt.show()

# to netcdf -- NOT WORKING
ds.to_netcdf('test.nc')

# write -- NOT WORKING
ds.selafin.write('test.slf')
